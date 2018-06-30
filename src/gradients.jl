struct GradientCache{CacheType1, CacheType2, CacheType3, fdtype, returntype, inplace}
    fx :: CacheType1
    c1 :: CacheType2
    c2 :: CacheType3
end

function GradientCache(
    df         :: Union{<:Number,AbstractArray{<:Number}},
    x          :: Union{<:Number, AbstractArray{<:Number}},
    fdtype     :: Type{T1} = Val{:central},
    returntype :: Type{T2} = eltype(df),
    inplace    :: Type{Val{T3}} = Val{true}) where {T1,T2,T3}

    if typeof(x)<:AbstractArray # the vector->scalar case
        if fdtype!=Val{:complex} # complex-mode FD only needs one cache, for x+eps*im
            if typeof(x)<:StridedVector
                if eltype(df)<:Complex && !(eltype(x)<:Complex)
                    _c1 = fill(zero(Complex{eltype(x)}), size(x))
                    _c2 = nothing
                else
                    _c1 = nothing
                    _c2 = nothing
                end
            else
                _c1 = similar(x)
                _c2 = fill(zero(real(eltype(x))), size(x))
            end
        else
            if !(returntype<:Real)
                fdtype_error(returntype)
            else
                _c1 = x .+ 0*im
                _c2 = nothing
            end
        end
    else # the scalar->vector case
        # need cache arrays for fx1 and fx2, except in complex mode, which needs one complex array
        if fdtype != Val{:complex}
            _c1 = similar(df)
            _c2 = similar(df)
        else
            _c1 = fill(zero(Complex{eltype(x)}), size(df))
            _c2 = nothing
        end
    end

    GradientCache{Nothing,typeof(_c1),typeof(_c2),fdtype,
                  returntype,inplace}(nothing,_c1,_c2)

end

function GradientCache(
    c1         :: Union{Nothing,AbstractArray{<:Number}},
    c2         :: Union{Nothing,AbstractArray{<:Number}},
    fx         :: Union{Nothing,<:Number,AbstractArray{<:Number}} = nothing,
    fdtype     :: Type{T1} = Val{:central},
    returntype :: Type{T2} = eltype(c1),
    inplace    :: Type{Val{T3}} = Val{true}) where {T1,T2,T3}

    if fdtype!=Val{:forward} && typeof(fx)!=Nothing
        @warn("Pre-computed function values are only useful for fdtype == Val{:forward}.")
        _fx = nothing
    else
        # more runtime sanity checks?
        _fx = fx
    end

    if typeof(x)<:AbstractArray # the vector->scalar case
        # need cache arrays for x1 (c1) and epsilon (c2) (both only if non-StridedArray)
        if fdtype!=Val{:complex} # complex-mode FD only needs one cache, for x+eps*im
            if typeof(x)<:StridedVector
                if eltype(df)<:Complex && !(eltype(x)<:Complex)
                    _c1 = fill(zero(Complex{eltype(x)}), size(x))
                    _c2 = nothing
                else
                    _c1 = nothing
                    _c2 = nothing
                    if typeof(c1)!=Nothing || typeof(c2)!=Nothing
                        @warn("For StridedVectors, neither c1 nor c2 are necessary.")
                    end
                end
            else
                _c1 = c1
                _c2 = c2
            end
        else
            if !(returntype<:Real)
                fdtype_error(returntype)
            else
                _c1 = x + 0*im
                _c2 = nothing
            end
        end

    else # the scalar->vector case
        # need cache arrays for fx1 and fx2, except in complex mode, which needs one complex array
        if fdtype != Val{:complex}
            _c1 = c1
            _c2 = c2
        else
            _c1 = c1
            _c2 = nothing
        end
    end
    GradientCache{typeof(_fx),typeof(_c1),typeof(_c2),fdtype,returntype,inplace}(_fx,_c1,_c2)
end

function finite_difference_gradient(f, x, fdtype::Type{T1}=Val{:central},
    returntype::Type{T2}=eltype(x), inplace::Type{Val{T3}}=Val{true},
    fx::Union{Nothing,AbstractArray{<:Number}}=nothing,
    c1::Union{Nothing,AbstractArray{<:Number}}=nothing,
    c2::Union{Nothing,AbstractArray{<:Number}}=nothing) where {T1,T2,T3}

    if typeof(x) <: AbstractArray
        df = fill(zero(returntype), size(x))
    else
        if inplace == Val{true}
            if typeof(fx)==Nothing && typeof(c1)==Nothing && typeof(c2)==Nothing
                error("In the scalar->vector in-place map case, at least one of fx, c1 or c2 must be provided, otherwise we cannot infer the return size.")
            else
                if     c1 != nothing    df = similar(c1)
                elseif fx != nothing    df = similar(fx)
                elseif c2 != nothing    df = similar(c2)
                end
            end
        else
            df = similar(f(x))
        end
    end
    cache = GradientCache(df,x,fdtype,returntype,inplace)
    finite_difference_gradient!(df,f,x,cache)
end

function finite_difference_gradient!(df, f, x, fdtype::Type{T1}=Val{:central},
    returntype::Type{T2}=eltype(df), inplace::Type{Val{T3}}=Val{true},
    fx::Union{Nothing,AbstractArray{<:Number}}=nothing,
    c1::Union{Nothing,AbstractArray{<:Number}}=nothing,
    c2::Union{Nothing,AbstractArray{<:Number}}=nothing,
    ) where {T1,T2,T3}

    cache = GradientCache(df,x,fdtype,returntype,inplace)
    finite_difference_gradient!(df,f,x,cache)
end

function finite_difference_gradient(f,x,
    cache::GradientCache{T1,T2,T3,fdtype,returntype,inplace}) where {T1,T2,T3,fdtype,returntype,inplace}

    if typeof(x) <: AbstractArray
        df = fill(zero(returntype), size(x))
    else
        df = zero(cache.c1)
    end
    finite_difference_gradient!(df,f,x,cache)
    df
end

# vector of derivatives of a vector->scalar map by each component of a vector x
# this ignores the value of "inplace", because it doesn't make much sense
function finite_difference_gradient!(df::AbstractArray{<:Number}, f, x::AbstractArray{<:Number},
    cache::GradientCache{T1,T2,T3,fdtype,returntype,inplace}) where {T1,T2,T3,fdtype,returntype,inplace}

    # NOTE: in this case epsilon is a vector, we need two arrays for epsilon and x1
    # c1 denotes x1, c2 is epsilon
    fx, c1, c2 = cache.fx, cache.c1, cache.c2
    if fdtype != Val{:complex}
        epsilon_factor = compute_epsilon_factor(fdtype, eltype(x))
        @. c2 = compute_epsilon(fdtype, x, epsilon_factor)
        copyto!(c1,x)
    end
    if fdtype == Val{:forward}
        @inbounds for i ∈ eachindex(x)
            epsilon = c2[i]
            c1_old = c1[i]
            c1[i] += epsilon
            if typeof(fx) != Nothing
                dfi = (f(c1) - fx) / epsilon
            else
                fx0 = f(x)
                dfi = (f(c1) - fx0) / epsilon
            end
            df[i] = real(dfi)
            c1[i] = c1_old
            if eltype(df)<:Complex
                c1[i] += im * epsilon
                if typeof(fx) != Nothing
                    dfi = (f(c1) - fx) / (im*epsilon)
                else
                    dfi = (f(c1) - fx0) / (im*epsilon)
                end
                c1[i] = c1_old
                df[i] -= im * imag(dfi)
            end
        end
    elseif fdtype == Val{:central}
        @inbounds for i ∈ eachindex(x)
            epsilon = c2[i]
            c1_old = c1[i]
            c1[i] += epsilon
            x_old  = x[i]
            x[i]  -= epsilon
            df[i]  = real((f(c1) - f(x)) / (2*epsilon))
            c1[i]  = c1_old
            x[i]   = x_old
            if eltype(df)<:Complex
                c1[i] += im*epsilon
                x[i]  -= im*epsilon
                df[i] -= im*imag( (f(c1) - f(x)) / (2*im*epsilon) )
                c1[i] = c1_old
                x[i] = x_old
            end
        end
    elseif fdtype == Val{:complex} && returntype <: Real
        copyto!(c1,x)
        epsilon_complex = eps(real(eltype(x)))
        # we use c1 here to avoid typing issues with x
        @inbounds for i ∈ eachindex(x)
            c1_old = c1[i]
            c1[i] += im*epsilon_complex
            df[i]  = imag(f(c1)) / epsilon_complex
            c1[i]  = c1_old
        end
    else
        fdtype_error(returntype)
    end
    df
end

function finite_difference_gradient!(df::StridedVector{<:Number}, f, x::StridedVector{<:Number},
    cache::GradientCache{T1,T2,T3,fdtype,returntype,inplace}) where {T1,T2,T3,fdtype,returntype,inplace}

    # c1 is x1 if we need a complex copy of x, otherwise Nothing
    # c2 is Nothing
    fx, c1, c2 = cache.fx, cache.c1, cache.c2
    if fdtype != Val{:complex}
        epsilon_factor = compute_epsilon_factor(fdtype, eltype(x))
        if eltype(df)<:Complex && !(eltype(x)<:Complex)
            copyto!(c1,x)
        end
    end
    if fdtype == Val{:forward}
        for i ∈ eachindex(x)
            epsilon = compute_epsilon(fdtype, x[i], epsilon_factor)
            x_old = x[i]
            if typeof(fx) != Nothing
                x[i] += epsilon
                dfi = (f(x) - fx) / epsilon
                x[i] = x_old
            else
                fx0 = f(x)
                x[i] += epsilon
                dfi = (f(x) - fx0) / epsilon
                x[i] = x_old
            end

            df[i] = real(dfi)
            if eltype(df)<:Complex
                if eltype(x)<:Complex
                    x[i] += im * epsilon
                    if typeof(fx) != Nothing
                        dfi = (f(x) - fx) / (im*epsilon)
                    else
                        dfi = (f(x) - fx0) / (im*epsilon)
                    end
                    x[i] = x_old
                else
                    c1[i] += im * epsilon
                    if typeof(fx) != Nothing
                        dfi = (f(c1) - fx) / (im*epsilon)
                    else
                        dfi = (f(c1) - fx0) / (im*epsilon)
                    end
                    c1[i] = x_old
                end
                df[i] -= im * imag(dfi)
            end
        end
    elseif fdtype == Val{:central}
        @inbounds for i ∈ eachindex(x)
            epsilon = compute_epsilon(fdtype, x[i], epsilon_factor)
            x_old = x[i]
            x[i] += epsilon
            dfi = f(x)
            x[i] = x_old - epsilon
            dfi -= f(x)
            x[i] = x_old
            df[i] = real(dfi / (2*epsilon))
            if eltype(df)<:Complex
                if eltype(x)<:Complex
                    x[i] += im*epsilon
                    dfi = f(x)
                    x[i] = x_old - im*epsilon
                    dfi -= f(x)
                    x[i] = x_old
                else
                    c1[i] += im*epsilon
                    dfi = f(c1)
                    c1[i] = x_old - im*epsilon
                    dfi -= f(c1)
                    c1[i] = x_old
                end
                df[i] -= im*imag(dfi / (2*im*epsilon))
            end
        end
    elseif fdtype==Val{:complex} && returntype<:Real && eltype(df)<:Real && eltype(x)<:Real
        copyto!(c1,x)
        epsilon_complex = eps(real(eltype(x)))
        # we use c1 here to avoid typing issues with x
        @inbounds for i ∈ eachindex(x)
            c1_old = c1[i]
            c1[i] += im*epsilon_complex
            df[i]  = imag(f(c1)) / epsilon_complex
            c1[i]  = c1_old
        end
    else
        fdtype_error(returntype)
    end
    df
end

# vector of derivatives of a scalar->vector map
# this is effectively a vector of partial derivatives, but we still call it a gradient
function finite_difference_gradient!(df::AbstractArray{<:Number}, f, x::Number,
    cache::GradientCache{T1,T2,T3,fdtype,returntype,inplace}) where {T1,T2,T3,fdtype,returntype,inplace}

    # NOTE: in this case epsilon is a scalar, we need two arrays for fx1 and fx2
    # c1 denotes fx1, c2 is fx2, sizes guaranteed by the cache constructor
    fx, c1, c2 = cache.fx, cache.c1, cache.c2

    if fdtype == Val{:forward}
        epsilon_factor = compute_epsilon_factor(fdtype, eltype(x))
        epsilon = compute_epsilon(Val{:forward}, x, epsilon_factor)
        if inplace == Val{true}
            f(c1, x+epsilon)
        else
            c1 .= f(x+epsilon)
        end
        if typeof(fx) != Nothing
            @. df = (c1 - fx) / epsilon
        else
            if inplace == Val{true}
                f(c2, x)
            else
                c2 .= f(x)
            end
            @. df = (c1 - c2) / epsilon
        end
    elseif fdtype == Val{:central}
        epsilon_factor = compute_epsilon_factor(fdtype, eltype(x))
        epsilon = compute_epsilon(Val{:central}, x, epsilon_factor)
        if inplace == Val{true}
            f(c1, x+epsilon)
            f(c2, x-epsilon)
        else
            c1 .= f(x+epsilon)
            c2 .= f(x-epsilon)
        end
        @. df = (c1 - c2) / (2*epsilon)
    elseif fdtype == Val{:complex} && returntype <: Real
        epsilon_complex = eps(real(eltype(x)))
        if inplace == Val{true}
            f(c1, x+im*epsilon_complex)
        else
            c1 .= f(x+im*epsilon_complex)
        end
        @. df = imag(c1) / epsilon_complex
    else
        fdtype_error(returntype)
    end
    df
end
