#=
Derivative of f : R -> R or f : C -> C at a single point x.
=#
function finite_difference_derivative(f, x::T, fdtype::DataType, funtype::DataType=Val{:Real},
    f_x::Union{Void,T}=nothing) where T<:Number

    if funtype == Val{:Real}
        if fdtype == Val{:complex}
            epsilon = eps(T)
        else
            epsilon = compute_epsilon(fdtype, x)
        end
    elseif funtype == Val{:Complex}
        epsilon = compute_epsilon(fdtype, real(x))
    else
        fdtype_error(funtype)
    end

    _finite_difference_kernel(f, x, fdtype, funtype, epsilon, f_x)
end

#=
Finite difference kernels for single point derivatives of f : R -> R.
These are currently underused because of inlining / broadcast issues.
Revisit this in Julia v0.7 / 1.0.
=#
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
# Single point derivative implementations end here.


#=
Multi-point implementations of scalar derivatives for efficiency.
=#
struct DerivativeCache{CacheType1, CacheType2, fdtype, RealOrComplex}
    fx      :: CacheType1
    epsilon :: CacheType2
end

function DerivativeCache(
    x       :: AbstractArray{<:Number},
    fx      :: Union{Void,AbstractArray{<:Number}} = nothing,
    epsilon :: Union{Void,AbstractArray{<:Number}} = nothing,
    fdtype  :: DataType = Val{:central},
    RealOrComplex :: DataType =
        fdtype==Val{:complex} ? Val{:Real} : eltype(x) <: Complex ?
        Val{:Complex} : Val{:Real}
    )

    if typeof(x)<:StridedArray && typeof(fx)<:Union{Void,StridedArray}
        if typeof(epsilon)!=Void
            warn("StridedArrays don't benefit from pre-allocating epsilon.")
            epsilon = nothing
        end
    end
    if fdtype == Val{:complex}
        if RealOrComplex == Val{:Complex}
            fdtype_error(Val{:Complex})
        end
        if typeof(fx) != Void || typeof(epsilon) != Void
            warn("Val{:complex} doesn't benefit from cache arrays.")
        end
        return DerivativeCache{Void,Void,fdtype,RealOrComplex}(nothing, nothing)
    else
        if !(typeof(x)<:StridedArray && typeof(fx)<:Union{Void,StridedArray})
            epsilon_elemtype = compute_epsilon_elemtype(epsilon, x)
            if typeof(epsilon) == Void || eltype(epsilon) != epsilon_elemtype
                epsilon = zeros(epsilon_elemtype, size(x))
            end
            epsilon_factor = compute_epsilon_factor(fdtype, real(eltype(x)))
            @. epsilon = compute_epsilon(fdtype, real(x), epsilon_factor)
        end
        if fdtype != Val{:forward}
            if typeof(fx) != Void
                warn("Pre-computed function values are only useful for fdtype == Val{:forward}.")
            end
            return DerivativeCache{Void,typeof(epsilon),fdtype,RealOrComplex}(nothing,epsilon)
        else
            return DerivativeCache{typeof(fx),typeof(epsilon),fdtype,RealOrComplex}(fx,epsilon)
        end
    end
end

#=
Compute the derivative df of a scalar-valued map f at a collection of points x.
=#
function finite_difference_derivative(f, x::AbstractArray{<:Number}, fdtype::DataType=Val{:central},
    RealOrComplex :: DataType =
        fdtype==Val{:complex} ? Val{:Real} : eltype(x) <: Complex ?
        Val{:Complex} : Val{:Real},
    fx :: Union{Void,AbstractArray{<:Number}}=nothing,
    epsilon :: Union{Void,AbstractArray{<:Real}}=nothing,
    return_type :: DataType=eltype(x))

    df = zeros(return_type, size(x))
    finite_difference_derivative!(df, f, x, fdtype, RealOrComplex, fx, epsilon, return_type)
end

function finite_difference_derivative!(df::AbstractArray{<:Number}, f,
    x::AbstractArray{<:Number}, fdtype::DataType=Val{:central},
    RealOrComplex :: DataType =
        fdtype==Val{:complex} ? Val{:Real} : eltype(x) <: Complex ?
        Val{:Complex} : Val{:Real},
    fx::Union{Void,AbstractArray{<:Number}}=nothing,
    epsilon::Union{Void,AbstractArray{<:Real}}=nothing, return_type::DataType=eltype(x))

    cache = DerivativeCache(x, fx, epsilon, fdtype, RealOrComplex)
    _finite_difference_derivative!(df, f, x, cache)
end

function finite_difference_derivative!(df::AbstractArray{<:Number}, f, x::AbstractArray{<:Number},
    cache::DerivativeCache{T1,T2,fdtype,RealOrComplex}) where {T1,T2,fdtype,RealOrComplex}

    _finite_difference_derivative!(df, f, x, cache)
end

function _finite_difference_derivative!(df::AbstractArray{<:Real}, f, x::AbstractArray{<:Real},
    cache::DerivativeCache{T1,T2,fdtype,Val{:Real}}) where {T1,T2,fdtype}

    fx, epsilon = cache.fx, cache.epsilon
    if fdtype == Val{:forward}
        if typeof(fx) == Void
            @. df = (f(x+epsilon) - f(x)) / epsilon
        else
            @. df = (f(x+epsilon) - fx) / epsilon
        end
    elseif fdtype == Val{:central}
        @. df = (f(x+epsilon) - f(x-epsilon)) / (2 * epsilon)
    elseif fdtype == Val{:complex}
        epsilon_elemtype = compute_epsilon_elemtype(nothing, x)
        epsilon_complex = eps(epsilon_elemtype)
        @. df = imag(f(x+im*epsilon_complex)) / epsilon_complex
    else
        fdtype_error(Val{:Real})
    end
    df
end

function _finite_difference_derivative!(df::AbstractArray{<:Number}, f, x::AbstractArray{<:Number},
    cache::DerivativeCache{T1,T2,fdtype,Val{:Complex}}) where {T1,T2,fdtype}

    fx, epsilon = cache.fx, cache.epsilon
    if fdtype == Val{:forward}
        if typeof(fx) == Void
            fx = f.(x)
        end
        @. df = real((f(x+epsilon) - fx)) / epsilon + im*imag((f(x+im*epsilon) - fx)) / epsilon
    elseif fdtype == Val{:central}
        @. df = real(f(x+epsilon) - f(x-epsilon)) / (2 * epsilon) + im*imag(f(x+im*epsilon) - f(x-im*epsilon)) / (2 * epsilon)
    else
        fdtype_error(Val{:Complex})
    end
    df
end

#=
Optimized implementations for StridedArrays.
Essentially, the only difference between these and the AbstractArray case
is that here we can compute the epsilon one by one in local variables and avoid caching it.
=#
function _finite_difference_derivative!(df::StridedArray{<:Real}, f, x::StridedArray{<:Real},
    cache::DerivativeCache{T1,T2,fdtype,Val{:Real}}) where {T1,T2,fdtype}

    epsilon_elemtype = compute_epsilon_elemtype(nothing, x)
    if fdtype == Val{:forward}
        fx = cache.fx
        epsilon_factor = compute_epsilon_factor(Val{:forward}, epsilon_elemtype)
        @inbounds for i in 1 : length(x)
            epsilon = compute_epsilon(Val{:forward}, x[i], epsilon_factor)
            x_plus = x[i] + epsilon
            if typeof(fx) == Void
                df[i] = (f(x_plus) - f(x[i])) / epsilon
            else
                df[i] = (f(x_plus) - fx[i]) / epsilon
            end
        end
    elseif fdtype == Val{:central}
        epsilon_factor = compute_epsilon_factor(Val{:central}, epsilon_elemtype)
        @inbounds for i in 1 : length(x)
            epsilon = compute_epsilon(Val{:central}, x[i], epsilon_factor)
            epsilon_double_inv = one(typeof(epsilon)) / (2*epsilon)
            x_plus, x_minus = x[i]+epsilon, x[i]-epsilon
            df[i] = (f(x_plus) - f(x_minus)) * epsilon_double_inv
        end
    elseif fdtype == Val{:complex}
        epsilon_complex = eps(eltype(x))
        @inbounds for i in 1 : length(x)
            df[i] = imag(f(x[i]+im*epsilon_complex)) / epsilon_complex
        end
    else
        fdtype_error(Val{:Real})
    end
    df
end

function _finite_difference_derivative!(df::StridedArray{<:Number}, f, x::StridedArray{<:Number},
    cache::DerivativeCache{T1,T2,fdtype,Val{:Complex}}) where {T1,T2,fdtype}

    epsilon_elemtype = compute_epsilon_elemtype(nothing, x)
    if fdtype == Val{:forward}
        fx = cache.fx
        epsilon_factor = compute_epsilon_factor(Val{:forward}, epsilon_elemtype)
        @inbounds for i in 1 : length(x)
            epsilon = compute_epsilon(Val{:forward}, real(x[i]), epsilon_factor)
            if typeof(fx) == Void
                fxi = f(x[i])
            else
                fxi = fx[i]
            end
            df[i] = ( real( f(x[i]+epsilon) - fxi ) + im*imag( f(x[i]+im*epsilon) - fxi ) ) / epsilon
        end
    elseif fdtype == Val{:central}
        epsilon_factor = compute_epsilon_factor(Val{:central}, epsilon_elemtype)
        @inbounds for i in 1 : length(x)
            epsilon = compute_epsilon(Val{:central}, real(x[i]), epsilon_factor)
            df[i] = (real(f(x[i]+epsilon) - f(x[i]-epsilon)) + im*imag(f(x[i]+im*epsilon) - f(x[i]-im*epsilon))) / (2 * epsilon)
        end
    else
        fdtype_error(Val{:Complex})
    end
    df
end
