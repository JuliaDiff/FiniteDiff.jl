#=
Single-point derivatives of scalar->scalar maps.
=#
function finite_difference_derivative(f, x::T, fdtype::DataType=Val{:central},
    returntype::DataType=eltype(x), f_x::Union{Void,T}=nothing) where T<:Number

    epsilon = compute_epsilon(fdtype, x)
    if fdtype==Val{:forward}
        return (f(x+epsilon) - f(x)) / epsilon
    elseif fdtype==Val{:central}
        return (f(x+epsilon) - f(x-epsilon)) / (2*epsilon)
    elseif fdtype==Val{:complex} && returntype<:Real
        return imag(f(x+im*epsilon)) / epsilon
    end
    fdtype_error(returntype)
end

#=
Finite difference kernels for single point derivatives.
These are currently unused because of inlining / broadcast issues.
Revisit this in Julia v0.7 / 1.0.
=#
#=
@inline function _finite_difference_kernel(f, x::T, ::Type{Val{:forward}}, ::Type{Val{:Real}},
    epsilon::T, fx::Union{Void,T}=nothing) where T<:Real

    if typeof(fx) == Void
        return (f(x+epsilon) - f(x)) / epsilon
    else
        return (f(x+epsilon) - fx) / epsilon
    end
end

@inline function _finite_difference_kernel(f, x::T, ::Type{Val{:central}}, ::Type{Val{:Real}},
    epsilon::T, ::Union{Void,T}=nothing) where T<:Real

    (f(x+epsilon) - f(x-epsilon)) / (2 * epsilon)
end

@inline function _finite_difference_kernel(f, x::T, ::Type{Val{:complex}}, ::Type{Val{:Real}},
    epsilon::T, ::Union{Void,T}=nothing) where T<:Real

    imag(f(x+im*epsilon)) / epsilon
end

@inline function _finite_difference_kernel(f, x::Number, ::Type{Val{:forward}}, ::Type{Val{:Complex}},
    epsilon::Real, fx::Union{Void,<:Number}=nothing)

    if typeof(fx) == Void
        return real((f(x+epsilon) - f(x))) / epsilon + im*imag((f(x+im*epsilon) - f(x))) / epsilon
    else
        return real((f(x+epsilon) - fx)) / epsilon + im*imag((f(x+im*epsilon) - fx)) / epsilon
    end
end

@inline function _finite_difference_kernel(f, x::Number, ::Type{Val{:central}}, ::Type{Val{:Complex}},
    epsilon::Real, fx::Union{Void,<:Number}=nothing)

    real(f(x+epsilon) - f(x-epsilon)) / (2 * epsilon) + im*imag(f(x+im*epsilon) - f(x-im*epsilon)) / (2 * epsilon)
end
=#
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
    fx         :: Union{Void,AbstractArray{<:Number}} = nothing,
    epsilon    :: Union{Void,AbstractArray{<:Real}} = nothing,
    fdtype     :: DataType = Val{:central},
    returntype :: DataType = eltype(x))

    if fdtype==Val{:complex} && !(eltype(returntype)<:Real)
        fdtype_error(returntype)
    end

    if fdtype!=Val{:forward} && typeof(fx)!=Void
        warn("Pre-computed function values are only useful for fdtype==Val{:forward}.")
        _fx = nothing
    else
        # more runtime sanity checks?
        _fx = fx
    end

    if typeof(epsilon)!=Void && typeof(x)<:StridedArray && typeof(fx)<:Union{Void,StridedArray} && 1==2
        warn("StridedArrays don't benefit from pre-allocating epsilon.")
        _epsilon = nothing
    elseif typeof(epsilon)!=Void && fdtype==Val{:complex}
        warn("Val{:complex} makes the epsilon array redundant.")
        _epsilon = nothing
    else
        if typeof(epsilon)==Void || eltype(epsilon)!=real(eltype(x))
            epsilon = zeros(real(eltype(x)), size(x))
        end
        epsilon_factor = compute_epsilon_factor(fdtype, eltype(x))
        @. epsilon = compute_epsilon(fdtype, x, epsilon_factor)
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
    fdtype     :: DataType = Val{:central},
    returntype :: DataType = eltype(x),      # return type of f
    fx         :: Union{Void,AbstractArray{<:Number}} = nothing,
    epsilon    :: Union{Void,AbstractArray{<:Real}} = nothing)

    df = zeros(returntype, size(x))
    finite_difference_derivative!(df, f, x, fdtype, returntype, fx, epsilon)
end

function finite_difference_derivative!(
    df         :: AbstractArray{<:Number},
    f,
    x          :: AbstractArray{<:Number},
    fdtype     :: DataType = Val{:central},
    returntype :: DataType = eltype(x),
    fx         :: Union{Void,AbstractArray{<:Number}} = nothing,
    epsilon    :: Union{Void,AbstractArray{<:Real}}   = nothing)

    cache = DerivativeCache(x, fx, epsilon, fdtype, returntype)
    finite_difference_derivative!(df, f, x, cache)
end

function finite_difference_derivative!(df::AbstractArray{<:Number}, f, x::AbstractArray{<:Number},
    cache::DerivativeCache{T1,T2,fdtype,returntype}) where {T1,T2,fdtype,returntype}

    fx, epsilon = cache.fx, cache.epsilon
    if fdtype == Val{:forward}
        if typeof(fx) == Void
            @. df = (f(x+epsilon) - f(x)) / epsilon
        else
            @. df = (f(x+epsilon) - fx) / epsilon
        end
    elseif fdtype == Val{:central}
        @. df = (f(x+epsilon) - f(x-epsilon)) / (2 * epsilon)
    elseif fdtype == Val{:complex} && returntype<:Real
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
is that here we can compute the epsilon one by one in local variables and avoid caching it.
=#
function finite_difference_derivative!(df::StridedArray, f, x::StridedArray,
    cache::DerivativeCache{T1,T2,fdtype,returntype}) where {T1,T2,fdtype,returntype}

    epsilon_factor = compute_epsilon_factor(fdtype, eltype(x))
    if fdtype == Val{:forward}
        fx = cache.fx
        @inbounds for i ∈ eachindex(x)
            epsilon = compute_epsilon(Val{:forward}, x[i], epsilon_factor)
            x_plus = x[i] + epsilon
            if typeof(fx) == Void
                df[i] = (f(x_plus) - f(x[i])) / epsilon
            else
                df[i] = (f(x_plus) - fx[i]) / epsilon
            end
        end
    elseif fdtype == Val{:central}
        @inbounds for i ∈ eachindex(x)
            epsilon = compute_epsilon(Val{:central}, x[i], epsilon_factor)
            epsilon_double_inv = one(typeof(epsilon)) / (2*epsilon)
            x_plus, x_minus = x[i]+epsilon, x[i]-epsilon
            df[i] = (f(x_plus) - f(x_minus)) * epsilon_double_inv
        end
    elseif fdtype == Val{:complex}
        epsilon_complex = eps(eltype(x))
        @inbounds for i ∈ eachindex(x)
            df[i] = imag(f(x[i]+im*epsilon_complex)) / epsilon_complex
        end
    else
        fdtype_error(returntype)
    end
    df
end
