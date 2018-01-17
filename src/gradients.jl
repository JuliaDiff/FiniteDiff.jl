struct GradientCache{CacheType1, CacheType2, CacheType3, fdtype, RealOrComplex}
    fx :: CacheType1
    c1 :: CacheType2
    c2 :: CacheType3
end

function GradientCache(
    df     :: AbstractArray{<:Number},
    x      :: Union{<:Number, AbstractArray{<:Number}},
    fx     :: Union{Void,<:Number,AbstractArray{<:Number}} = nothing,
    c1     :: Union{Void,AbstractArray{<:Number}} = nothing,
    c2     :: Union{Void,AbstractArray{<:Number}} = nothing,
    fdtype :: DataType = Val{:central},
    RealOrComplex :: DataType =
        fdtype==Val{:complex} ? Val{:Real} : eltype(x) <: Complex ? Val{:Complex} : Val{:Real}
    )

    if fdtype != Val{:forward} && typeof(fx) != Void
        warn("Pre-computed function values are only useful for fdtype == Val{:forward}.")
        _fx = nothing
    else
        # more runtime sanity checks?
        _fx = fx
    end

    if typeof(x) <: AbstractArray # the f:R^n->R case
        # need cache arrays for epsilon (c1) and x1 (c2)
        if fdtype != Val{:complex} # complex-mode FD only needs one cache, for x+eps*im
            epsilon_elemtype = compute_epsilon_elemtype(nothing, x)
            if typeof(c1) == Void || eltype(c1) != epsilon_elemtype
                _c1 = zeros(epsilon_elemtype, size(x))
            else
                _c1 = c1
            end
            epsilon_factor = compute_epsilon_factor(fdtype, real(eltype(x)))
            @. _c1 = compute_epsilon(fdtype, real(x), epsilon_factor)

            if typeof(c2) != typeof(x) || size(c2) != size(x)
                _c2 = copy(x)
            else
                copy!(_c2, x)
            end
        else
            _c1 = x + 0*im
            _c2 = nothing
        end

    else # the f:R->R^n case
        # need cache arrays for fx1 and fx2, except in complex mode
        if fdtype != Val{:complex}
            if typeof(c1) != typeof(df) || size(c1) != size(df)
                _c1 = similar(df)
            else
                _c1 = c1
            end
            if fdtype == Val{:forward} && typeof(fx) != Void
                _c2 = nothing
            else
                if typeof(c2) != typeof(df) || size(c2) != size(df)
                    _c2 = similar(df)
                else
                    _c2 = c2
                end
            end
        else
            _c1 = nothing
            _c2 = nothing
        end
    end
    GradientCache{typeof(_fx),typeof(_c1),typeof(_c2),fdtype,RealOrComplex}(_fx,_c1,_c2)
end

function finite_difference_gradient(f, x, fdtype::DataType=Val{:central},
    RealOrComplex::DataType =
        fdtype==Val{:complex} ? Val{:Real} : eltype(x) <: Complex ? Val{:Complex} : Val{:Real},
    fx::Union{Void,AbstractArray{<:Number}}=nothing,
    c1::Union{Void,AbstractArray{<:Number}}=nothing,
    c2::Union{Void,AbstractArray{<:Number}}=nothing,
    )

    if typeof(x) <: AbstractArray
        df = similar(x)
    else
        df = similar(f(x))  # can we get rid of this by requesting more information?
    end
    cache = GradientCache(df,x,fx,c1,c2,fdtype,RealOrComplex)
    finite_difference_gradient!(df,f,x,cache)
end

function finite_difference_gradient!(df, f, x, fdtype::DataType=Val{:central},
    RealOrComplex::DataType =
        fdtype==Val{:complex} ? Val{:Real} : eltype(x) <: Complex ? Val{:Complex} : Val{:Real},
    fx::Union{Void,AbstractArray{<:Number}}=nothing,
    c1::Union{Void,AbstractArray{<:Number}}=nothing,
    c2::Union{Void,AbstractArray{<:Number}}=nothing,
    )

    cache = GradientCache(df,x,fx,c1,c2,fdtype,RealOrComplex)
    finite_difference_gradient!(df,f,x,cache)
end

function finite_difference_gradient(f,x,cache::GradientCache)
    if typeof(x) <: AbstractArray
        df = similar(x)
    else
        df = similar(cache.c1)
    end
    finite_difference_gradient!(df,f,x,cache)
    df
end

# vector of derivatives of f : R^n -> R by each component of a vector x
function finite_difference_gradient!(df::AbstractArray{<:Number}, f, x::AbstractArray{<:Number},
    cache::GradientCache{T1,T2,T3,fdtype,Val{:Real}}) where {T1,T2,T3,fdtype}

    # NOTE: in this case epsilon is a vector, we need two arrays for epsilon and x1
    # c1 denotes epsilon (pre-computed by the cache constructor),
    # c2 is x1, pre-set to the values of x by the cache constructor
    fx, c1, c2 = cache.fx, cache.c1, cache.c2
    if fdtype == Val{:forward}
        @inbounds for i ∈ eachindex(x)
            c2[i] += c1[i]
            if typeof(fx) != Void
                df[i] = (f(c2) - fx) / c1[i]
            else
                df[i]  = (f(c2) - f(x)) / c1[i]
            end
            c2[i] -= c1[i]
        end
    elseif fdtype == Val{:central}
        @inbounds for i ∈ eachindex(x)
            c2[i] += c1[i]
            x[i]  -= c1[i]
            df[i]  = (f(c2) - f(x)) / (2*c1[i])
            c2[i] -= c1[i]
            x[i]  += c1[i]
        end
    elseif fdtype == Val{:complex}
        epsilon_elemtype = compute_epsilon_elemtype(nothing, x)
        epsilon_complex = eps(epsilon_elemtype)
        # we use c1 here to avoid typing issues with x
        @inbounds for i ∈ eachindex(x)
            c1[i] += im*epsilon_complex
            df[i]  = imag(f(c1)) / epsilon_complex
            c1[i] -= im*epsilon_complex
        end
    end
    df
end

# vector of derivatives of f : R -> R^n
# this is effectively a vector of partial derivatives, but we still call it a gradient
function finite_difference_gradient!(df::AbstractArray{<:Number}, f, x::Number,
    cache::GradientCache{T1,T2,T3,fdtype,Val{:Real}}) where {T1,T2,T3,fdtype}

    # NOTE: in this case epsilon is a scalar, we need two arrays for fx1 and fx2
    # c1 denotes fx1, c2 is fx2, sizes guaranteed by the cache constructor
    fx, c1, c2 = cache.fx, cache.c1, cache.c2

    epsilon_elemtype = compute_epsilon_elemtype(nothing, x)
    if fdtype == Val{:forward}
        epsilon_factor = compute_epsilon_factor(fdtype, real(eltype(x)))
        epsilon = compute_epsilon(Val{:forward}, real(x), epsilon_factor)
        c1 .= f(x+epsilon)
        if typeof(fx) != Void
            @. df = (c1 - fx) / epsilon
        else
            c2 .= f(x)
            @. df = (c1 - c2) / epsilon
        end
    elseif fdtype == Val{:central}
        epsilon_factor = compute_epsilon_factor(fdtype, real(eltype(x)))
        epsilon = compute_epsilon(Val{:central}, real(x), epsilon_factor)
        c1 .= f(x+epsilon)
        c2 .= f(x-epsilon)
        @. df = (c1 - c2) / (2*epsilon)
    elseif fdtype == Val{:complex}
        epsilon_complex = eps(epsilon_elemtype)
        df .= imag.(f(x+im*epsilon_complex)) ./ epsilon_complex
    end
    df
end

# vector of derivatives of f : C^n -> C by each component of a vector x
function finite_difference_gradient!(df::AbstractArray{<:Number}, f, x::AbstractArray{<:Number},
    cache::GradientCache{T1,T2,T3,fdtype,Val{:Complex}}) where {T1,T2,T3,fdtype}

    # NOTE: in this case epsilon is a vector, we need two arrays for epsilon and x1
    # c1 denotes epsilon (pre-computed by the cache constructor),
    # c2 is x1, pre-set to the values of x by the cache constructor
    fx, c1, c2 = cache.fx, cache.c1, cache.c2
    if fdtype == Val{:forward}
        @inbounds for i ∈ eachindex(x)
            epsilon = c1[i]
            c2[i] += epsilon
            if typeof(fx) == Void
                df[i]  = real(f(c2) - f(x)) / epsilon
                df[i] += im*imag(f(c2) - f(x)) / epsilon
            else
                df[i]  = real(f(c2) - fx) / epsilon
                df[i] += im*imag(f(c2) - fx) / epsilon
            end
            c2[i] -= epsilon
        end
    elseif fdtype == Val{:central}
        @inbounds for i ∈ eachindex(x)
            epsilon = c1[i]
            c2[i]  += epsilon
            x[i]   -= epsilon
            df[i]   = real(f(c2) - f(x)) / (2*epsilon)
            df[i]  += im*imag(f(c2) - f(x)) / (2*epsilon)
            c2[i]  -= c1[i]
            x[i]   += c1[i]            
        end
    else
        fdtype_error(Val{:complex})
    end
    df
end

# vector of derivatives of f : C -> C^n
# this is effectively a vector of partial derivatives, but we still call it a gradient
function finite_difference_gradient!(df::AbstractArray{<:Number}, f, x::Number,
    cache::GradientCache{T1,T2,T3,fdtype,Val{:Complex}}) where {T1,T2,T3,fdtype}

    # NOTE: in this case epsilon is a scalar, we need two arrays for fx1 and fx2
    # c1 denotes fx1, c2 is fx2, sizes guaranteed by the cache constructor
    fx, c1, c2 = cache.fx, cache.c1, cache.c2

    epsilon_elemtype = compute_epsilon_elemtype(nothing, x)
    if fdtype == Val{:forward}
        epsilon_factor = compute_epsilon_factor(fdtype, real(eltype(x)))
        epsilon = compute_epsilon(Val{:forward}, real(x), epsilon_factor)
        c1 .= f(x+epsilon)
        if typeof(fx) != Void
            @. df  = real(c1 - fx) / epsilon
            @. df += im*imag(c1 - fx) / epsilon
        else
            c2 .= f(x)
            @. df  = real(c1 - c2) / epsilon
            @. df += im*imag(c1 - c2) / epsilon
        end
    elseif fdtype == Val{:central}
        epsilon_factor = compute_epsilon_factor(fdtype, real(eltype(x)))
        epsilon = compute_epsilon(Val{:central}, real(x), epsilon_factor)
        c1 .= f(x+epsilon)
        c2 .= f(x-epsilon)
        @. df  = real(c1 - c2) / (2*epsilon)
        #c1 .= im*imag(f(x+im*epsilon))
        #c2 .= im*imag(f(x-im*epsilon))
        @. df += im*imag(c1 - c2) / (2*epsilon)
    elseif fdtype == Val{:complex}
        fdtype_error(Val{:complex})
    end
    df
end
