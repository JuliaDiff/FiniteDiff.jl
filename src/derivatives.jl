#=
Single-point derivatives of scalar->scalar maps.
=#
function finite_difference_derivative(
    f,
    x::T,
    fdtype=Val(:central),
    returntype=eltype(x),
    f_x=nothing;
    relstep=default_relstep(fdtype, T),
    absstep=relstep,
    dir=true) where {T<:Number}

    fdtype isa Type && (fdtype = fdtype())
    epsilon = compute_epsilon(fdtype, x, relstep, absstep, dir)
    if fdtype==Val(:forward)
        return (f(x+epsilon) - f(x)) / epsilon
    elseif fdtype==Val(:central)
        return (f(x+epsilon) - f(x-epsilon)) / (2*epsilon)
    elseif fdtype==Val(:complex) && returntype<:Real
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
    fdtype     :: Type{T1} = Val(:central),
    returntype :: Type{T2} = eltype(x)) where {T1,T2}

    fdtype isa Type && (fdtype = fdtype())
    if fdtype==Val(:complex) && !(eltype(returntype)<:Real)
        fdtype_error(returntype)
    end

    if fdtype!=Val(:forward) && typeof(fx)!=Nothing
        @warn("Pre-computed function values are only useful for fdtype==Val(:forward).")
        _fx = nothing
    else
        # more runtime sanity checks?
        _fx = fx
    end

    if typeof(epsilon)!=Nothing && typeof(x)<:StridedArray && typeof(fx)<:Union{Nothing,StridedArray} && 1==2
        @warn("StridedArrays don't benefit from pre-allocating epsilon.")
        _epsilon = nothing
    elseif typeof(epsilon)!=Nothing && fdtype==Val(:complex)
        @warn("Val(:complex) makes the epsilon array redundant.")
        _epsilon = nothing
    else
        if typeof(epsilon)==Nothing || eltype(epsilon)!=real(eltype(x))
          epsilon = fill(zero(real(eltype(x))), size(x))
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
    x,
    fdtype = Val(:central),
    returntype = eltype(x),      # return type of f
    fx = nothing,
    epsilon = nothing;
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep)

    df = fill(zero(returntype), size(x))
    finite_difference_derivative!(df, f, x, fdtype, returntype, fx, epsilon; relstep=relstep, absstep=absstep)
end

function finite_difference_derivative!(
    df,
    f,
    x,
    fdtype = Val(:central),
    returntype = eltype(x),
    fx = nothing,
    epsilon = nothing;
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep)

    cache = DerivativeCache(x, fx, epsilon, fdtype, returntype)
    finite_difference_derivative!(df, f, x, cache; relstep=relstep, absstep=absstep)
end

function finite_difference_derivative!(
    df,
    f,
    x,
    cache::DerivativeCache{T1,T2,fdtype,returntype};
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    dir=true) where {T1,T2,fdtype,returntype}

    fx, epsilon = cache.fx, cache.epsilon
    if typeof(epsilon) != Nothing
        @. epsilon = compute_epsilon(fdtype, x, relstep, absstep, dir)
    end
    if fdtype == Val(:forward)
        if typeof(fx) == Nothing
            @. df = (f(x+epsilon) - f(x)) / epsilon
        else
            @. df = (f(x+epsilon) - fx) / epsilon
        end
    elseif fdtype == Val(:central)
        @. df = (f(x+epsilon) - f(x-epsilon)) / (2 * epsilon)
    elseif fdtype == Val(:complex) && returntype<:Real
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
function finite_difference_derivative!(
    df::StridedArray,
    f,
    x::StridedArray,
    cache::DerivativeCache{T1,T2,fdtype,returntype};
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    dir=true) where {T1,T2,fdtype,returntype}

    if fdtype == Val(:forward)
        fx = cache.fx
        @inbounds for i ∈ eachindex(x)
            epsilon = compute_epsilon(Val(:forward), x[i], relstep, absstep, dir)
            x_plus = x[i] + epsilon
            if typeof(fx) == Nothing
                df[i] = (f(x_plus) - f(x[i])) / epsilon
            else
                df[i] = (f(x_plus) - fx[i]) / epsilon
            end
        end
    elseif fdtype == Val(:central)
        @inbounds for i ∈ eachindex(x)
            epsilon = compute_epsilon(Val(:central), x[i], relstep, absstep, dir)
            epsilon_double_inv = one(typeof(epsilon)) / (2*epsilon)
            x_plus, x_minus = x[i]+epsilon, x[i]-epsilon
            df[i] = (f(x_plus) - f(x_minus)) * epsilon_double_inv
        end
    elseif fdtype == Val(:complex)
        epsilon_complex = eps(eltype(x))
        @inbounds for i ∈ eachindex(x)
            df[i] = imag(f(x[i]+im*epsilon_complex)) / epsilon_complex
        end
    else
        fdtype_error(returntype)
    end
    df
end
