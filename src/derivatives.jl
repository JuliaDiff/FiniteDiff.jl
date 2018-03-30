#=
Single-point derivatives of scalar->scalar maps.
=#
function finite_difference_derivative(f, x::T, ::Val{fdtype}=Val{:central}(),
    returntype::Type{T1}=eltype(x), f_x::Union{Nothing,T}=nothing) where {T<:Number,fdtype,T1}

    epsilon = compute_epsilon(Val{fdtype}(), x)
    if fdtype==:forward
        return (f(x+epsilon) - f(x)) / epsilon
    elseif fdtype==:central
        return (f(x+epsilon) - f(x-epsilon)) / (2*epsilon)
    elseif fdtype==:complex && returntype<:Real
        return imag(f(x+im*epsilon)) / epsilon
    end
    fdtype_error(returntype)
end
# Single point derivative implementations end here.


#=
Multi-point implementations of scalar derivatives for efficiency.
=#
struct DerivativeCache{CacheType1, CacheType2, fdtype, returntype}
    fx      :: CacheType1
    epsilon :: CacheType2
end
function DerivativeCache(
    x          :: AbstractArray{<:Number},
    fx         :: Union{Nothing,AbstractArray{<:Number}} = nothing,
    epsilon    :: Union{Nothing,AbstractArray{<:Real}} = nothing,
               :: Val{fdtype} = Val{:central}(),
    returntype :: Type{T} = eltype(x)) where {fdtype,T}

    if fdtype==:complex && !(eltype(returntype)<:Real)
        fdtype_error(returntype)
    end

    if fdtype!=:forward && typeof(fx)!=Nothing
        Compat.@warn "Pre-computed function values are only useful for fdtype==Val(:forward)."
        _fx = nothing
    else
        # more runtime sanity checks?
        _fx = fx
    end

    if typeof(epsilon)!=Nothing && typeof(x)<:StridedArray && typeof(fx)<:Union{Nothing,StridedArray} && 1==2
        Compat.@warn "StridedArrays don't benefit from pre-allocating epsilon."
        _epsilon = nothing
    elseif typeof(epsilon)!=Nothing && fdtype==:complex
        Compat.@warn "Val(:complex) makes the epsilon array redundant."
        _epsilon = nothing
    else
        if typeof(epsilon)==Nothing || eltype(epsilon)!=real(eltype(x))
            epsilon = zeros(real(eltype(x)), size(x))
        end
        _epsilon = epsilon
    end
    DerivativeCache{typeof(_fx),typeof(_epsilon),fdtype,returntype}(_fx,_epsilon)
end



#=
Compute the derivative df of a scalar-valued map f at a collection of points x.
=#
function finite_difference_derivative(
    f,
    x          :: AbstractArray{<:Number},
               :: Val{fdtype} = Val{:central}(),
    returntype :: Type{T1} = eltype(x),      # return type of f
    fx         :: Union{Nothing,AbstractArray{<:Number}} = nothing,
    epsilon    :: Union{Nothing,AbstractArray{<:Real}} = nothing) where {fdtype,T1}

    df = zeros(returntype, size(x))
    finite_difference_derivative!(df, f, x, Val{fdtype}(), returntype, fx, epsilon)
end

function finite_difference_derivative!(
    df         :: AbstractArray{<:Number},
    f,
    x          :: AbstractArray{<:Number},
               :: Val{fdtype} = Val{:central}(),
    returntype :: Type{T1} = eltype(x),
    fx         :: Union{Nothing,AbstractArray{<:Number}} = nothing,
    epsilon    :: Union{Nothing,AbstractArray{<:Real}}   = nothing) where {fdtype,T1}

    cache = DerivativeCache(x, fx, epsilon, Val{fdtype}(), returntype)
    finite_difference_derivative!(df, f, x, cache)
end

function finite_difference_derivative!(df::AbstractArray{<:Number}, f, x::AbstractArray{<:Number},
    cache::DerivativeCache{T1,T2,fdtype,returntype}) where {T1,T2,fdtype,returntype}

    fx, epsilon = cache.fx, cache.epsilon
    if typeof(epsilon) != Nothing
        epsilon_factor = compute_epsilon_factor(fdtype, eltype(x))
        @. epsilon = compute_epsilon(fdtype, x, epsilon_factor)
    end
    if fdtype == :forward
        if typeof(fx) == Nothing
            @. df = (f(x+epsilon) - f(x)) / epsilon
        else
            @. df = (f(x+epsilon) - fx) / epsilon
        end
    elseif fdtype == :central
        @. df = (f(x+epsilon) - f(x-epsilon)) / (2 * epsilon)
    elseif fdtype == :complex && returntype<:Real
        epsilon_complex = eps(eltype(x))
        @. df = imag(f(x+im*epsilon_complex)) / epsilon_complex
    else
        fdtype_error(returntype)
    end
    df
end

#=
Optimized implementations for StridedArrays.
Essentially, the only difference between these and the AbstractArray case
is that here we can compute the epsilon one by one in local variables and aNothing caching it.
=#
function finite_difference_derivative!(df::StridedArray, f, x::StridedArray,
    cache::DerivativeCache{T1,T2,fdtype,returntype}) where {T1,T2,fdtype,returntype}

    epsilon_factor = compute_epsilon_factor(Val{fdtype}(), eltype(x))
    if fdtype == :forward
        fx = cache.fx
        @inbounds for i ∈ eachindex(x)
            epsilon = compute_epsilon(Val{:forward}(), x[i], epsilon_factor)
            x_plus = x[i] + epsilon
            if typeof(fx) == Nothing
                df[i] = (f(x_plus) - f(x[i])) / epsilon
            else
                df[i] = (f(x_plus) - fx[i]) / epsilon
            end
        end
    elseif fdtype == :central
        @inbounds for i ∈ eachindex(x)
            epsilon = compute_epsilon(Val{:central}(), x[i], epsilon_factor)
            epsilon_double_inv = one(typeof(epsilon)) / (2*epsilon)
            x_plus, x_minus = x[i]+epsilon, x[i]-epsilon
            df[i] = (f(x_plus) - f(x_minus)) * epsilon_double_inv
        end
    elseif fdtype == :complex
        epsilon_complex = eps(eltype(x))
        @inbounds for i ∈ eachindex(x)
            df[i] = imag(f(x[i]+im*epsilon_complex)) / epsilon_complex
        end
    else
        fdtype_error(returntype)
    end
    df
end
