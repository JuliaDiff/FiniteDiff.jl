struct GradientCache{CacheType1,CacheType2,CacheType3,CacheType4,fdtype,returntype,inplace}
    fx::CacheType1
    c1::CacheType2
    c2::CacheType3
    c3::CacheType4
end

function GradientCache(
    df,
    x,
    fdtype=Val(:central),
    returntype=eltype(df),
    inplace=Val(true))

    fdtype isa Type && (fdtype = fdtype())
    inplace isa Type && (inplace = inplace())
    if typeof(x) <: AbstractArray # the vector->scalar case
        if fdtype != Val(:complex) # complex-mode FD only needs one cache, for x+eps*im
            if typeof(x) <: StridedVector
                if eltype(df) <: Complex && !(eltype(x) <: Complex)
                    _c1 = zero(Complex{eltype(x)}) .* x
                    _c2 = nothing
                else
                    _c1 = nothing
                    _c2 = nothing
                end
            else
                _c1 = zero(x)
                _c2 = zero(real(eltype(x))) .* x
            end
        else
            if !(returntype <: Real)
                fdtype_error(returntype)
            else
                _c1 = x .+ zero(eltype(x)) .* im
                _c2 = nothing
            end
        end
        _c3 = zero(x)
    else # the scalar->vector case
        # need cache arrays for fx1 and fx2, except in complex mode, which needs one complex array
        if fdtype != Val(:complex)
            _c1 = zero(df)
            _c2 = zero(df)
        else
            _c1 = zero(Complex{eltype(x)}) .* df
            _c2 = nothing
        end
        _c3 = x
    end

    GradientCache{Nothing,typeof(_c1),typeof(_c2),typeof(_c3),fdtype,
        returntype,inplace}(nothing, _c1, _c2, _c3)

end

"""
    GradientCache(c1, c2, c3, fx, fdtype = Val(:central), returntype = eltype(fx), inplace = Val(false))

Construct a non-allocating gradient cache.

# Arguments 
- `c1`, `c2`, `c3`: (Non-aliased) caches for the input vector.
- `fx`: Cached function call.
- `fdtype = Val(:central)`: Method for cmoputing the finite difference.
- `returntype = eltype(fx)`: Element type for the returned function value.
- `inplace = Val(false)`: Whether the function is computed in-place or not.

# Output 
The output is a [`GradientCache`](@ref) struct.

```julia
julia> x = [1.0, 3.0]
2-element Vector{Float64}:
 1.0
 3.0

julia> _f = x -> x[1] + x[2]
#13 (generic function with 1 method)

julia> fx = _f(x)
4.0

julia> gradcache = GradientCache(copy(x), copy(x), copy(x), fx)
GradientCache{Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}, Val{:central}(), Float64, Val{false}()}(4.0, [1.0, 3.0], [1.0, 3.0], [1.0, 3.0])
```
"""
function GradientCache(
    fx::Fx,# match order in struct for Setfield
    c1::T,
    c2::T,
    c3::T,
    fdtype=Val(:central),
    returntype=eltype(fx),
    inplace=Val(true)) where {T,Fx} # Val(false) isn't so important for vector -> scalar, it gets ignored in that case anyway.
    GradientCache{Fx,T,T,T,fdtype,returntype,inplace}(fx, c1, c2, c3)
end

function finite_difference_gradient(
    f,
    x,
    fdtype=Val(:central),
    returntype=eltype(x),
    inplace=Val(true),
    fx=nothing,
    c1=nothing,
    c2=nothing;
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    dir=true)

    inplace isa Type && (inplace = inplace())
    if typeof(x) <: AbstractArray
        df = zero(returntype) .* x
    else
        if inplace == Val(true)
            if typeof(fx) == Nothing && typeof(c1) == Nothing && typeof(c2) == Nothing
                error("In the scalar->vector in-place map case, at least one of fx, c1 or c2 must be provided, otherwise we cannot infer the return size.")
            else
                if c1 != nothing
                    df = zero(c1)
                elseif fx != nothing
                    df = zero(fx)
                elseif c2 != nothing
                    df = zero(c2)
                end
            end
        else
            df = zero(f(x))
        end
    end
    cache = GradientCache(df, x, fdtype, returntype, inplace)
    finite_difference_gradient!(df, f, x, cache, relstep=relstep, absstep=absstep, dir=dir)
end

function finite_difference_gradient!(
    df,
    f,
    x,
    fdtype=Val(:central),
    returntype=eltype(df),
    inplace=Val(true),
    fx=nothing,
    c1=nothing,
    c2=nothing;
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep)

    cache = GradientCache(df, x, fdtype, returntype, inplace)
    finite_difference_gradient!(df, f, x, cache, relstep=relstep, absstep=absstep)
end

function finite_difference_gradient(
    f,
    x,
    cache::GradientCache{T1,T2,T3,T4,fdtype,returntype,inplace};
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    dir=true) where {T1,T2,T3,T4,fdtype,returntype,inplace}

    if typeof(x) <: AbstractArray
        df = zero(returntype) .* x
    else
        df = zero(cache.c1)
    end
    finite_difference_gradient!(df, f, x, cache, relstep=relstep, absstep=absstep, dir=dir)
    df
end

# vector of derivatives of a vector->scalar map by each component of a vector x
# this ignores the value of "inplace", because it doesn't make much sense
function finite_difference_gradient!(
    df,
    f,
    x,
    cache::GradientCache{T1,T2,T3,T4,fdtype,returntype,inplace};
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    dir=true) where {T1,T2,T3,T4,fdtype,returntype,inplace}

    # NOTE: in this case epsilon is a vector, we need two arrays for epsilon and x1
    # c1 denotes x1, c2 is epsilon
    fx, c1, c2, c3 = cache.fx, cache.c1, cache.c2, cache.c3
    if fdtype != Val(:complex) && ArrayInterface.fast_scalar_indexing(c2)
        @. c2 = compute_epsilon(fdtype, x, relstep, absstep, dir)
        copyto!(c1, x)
    end
    copyto!(c3, x)
    if fdtype == Val(:forward)
        @inbounds for i ∈ eachindex(x)
            if ArrayInterface.fast_scalar_indexing(c2)
                epsilon = ArrayInterface.allowed_getindex(c2, i) * dir
            else
                epsilon = compute_epsilon(fdtype, x, relstep, absstep, dir) * dir
            end
            c1_old = ArrayInterface.allowed_getindex(c1, i)
            ArrayInterface.allowed_setindex!(c1, c1_old + epsilon, i)
            if typeof(fx) != Nothing
                dfi = (f(c1) - fx) / epsilon
            else
                fx0 = f(x)
                dfi = (f(c1) - fx0) / epsilon
            end
            df_tmp = real(dfi)
            if eltype(df) <: Complex
                ArrayInterface.allowed_setindex!(c1, c1_old + im * epsilon, i)
                if typeof(fx) != Nothing
                    dfi = (f(c1) - fx) / (im * epsilon)
                else
                    dfi = (f(c1) - fx0) / (im * epsilon)
                end
                ArrayInterface.allowed_setindex!(c1, c1_old, i)
                ArrayInterface.allowed_setindex!(df, df_tmp - im * imag(dfi), i)
            else
                ArrayInterface.allowed_setindex!(df, df_tmp, i)
                ArrayInterface.allowed_setindex!(c1, c1_old, i)
            end
        end
    elseif fdtype == Val(:central)
        @inbounds for i ∈ eachindex(x)
            if ArrayInterface.fast_scalar_indexing(c2)
                epsilon = ArrayInterface.allowed_getindex(c2, i) * dir
            else
                epsilon = compute_epsilon(fdtype, x, relstep, absstep, dir) * dir
            end
            c1_old = ArrayInterface.allowed_getindex(c1, i)
            ArrayInterface.allowed_setindex!(c1, c1_old + epsilon, i)
            x_old = ArrayInterface.allowed_getindex(x, i)
            ArrayInterface.allowed_setindex!(c3, x_old - epsilon, i)
            df_tmp = real((f(c1) - f(c3)) / (2 * epsilon))
            if eltype(df) <: Complex
                ArrayInterface.allowed_setindex!(c1, c1_old + im * epsilon, i)
                ArrayInterface.allowed_setindex!(c3, x_old - im * epsilon, i)
                df_tmp2 = im * imag((f(c1) - f(c3)) / (2 * im * epsilon))
                ArrayInterface.allowed_setindex!(df, df_tmp - df_tmp2, i)
            else
                ArrayInterface.allowed_setindex!(df, df_tmp, i)
            end
            ArrayInterface.allowed_setindex!(c1, c1_old, i)
            ArrayInterface.allowed_setindex!(c3, x_old, i)
        end
    elseif fdtype == Val(:complex) && returntype <: Real
        copyto!(c1, x)
        epsilon_complex = eps(real(eltype(x)))
        # we use c1 here to avoid typing issues with x
        @inbounds for i ∈ eachindex(x)
            c1_old = ArrayInterface.allowed_getindex(c1, i)
            ArrayInterface.allowed_setindex!(c1, c1_old + im * epsilon_complex, i)
            ArrayInterface.allowed_setindex!(df, imag(f(c1)) / epsilon_complex, i)
            ArrayInterface.allowed_setindex!(c1, c1_old, i)
        end
    else
        fdtype_error(returntype)
    end
    df
end

function finite_difference_gradient!(
    df::StridedVector{<:Number},
    f,
    x::StridedVector{<:Number},
    cache::GradientCache{T1,T2,T3,T4,fdtype,returntype,inplace};
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    dir=true) where {T1,T2,T3,T4,fdtype,returntype,inplace}

    # c1 is x1 if we need a complex copy of x, otherwise Nothing
    # c2 is Nothing
    fx, c1, c2, c3 = cache.fx, cache.c1, cache.c2, cache.c3
    if fdtype != Val(:complex)
        if eltype(df) <: Complex && !(eltype(x) <: Complex)
            copyto!(c1, x)
        end
    end
    copyto!(c3, x)
    if fdtype == Val(:forward)
        for i ∈ eachindex(x)
            epsilon = compute_epsilon(fdtype, x[i], relstep, absstep, dir)
            x_old = x[i]
            if typeof(fx) != Nothing
                c3[i] += epsilon
                dfi = (f(c3) - fx) / epsilon
                c3[i] = x_old
            else
                fx0 = f(x)
                c3[i] += epsilon
                dfi = (f(c3) - fx0) / epsilon
                c3[i] = x_old
            end

            df[i] = real(dfi)
            if eltype(df) <: Complex
                if eltype(x) <: Complex
                    c3[i] += im * epsilon
                    if typeof(fx) != Nothing
                        dfi = (f(c3) - fx) / (im * epsilon)
                    else
                        dfi = (f(c3) - fx0) / (im * epsilon)
                    end
                    c3[i] = x_old
                else
                    c1[i] += im * epsilon
                    if typeof(fx) != Nothing
                        dfi = (f(c1) - fx) / (im * epsilon)
                    else
                        dfi = (f(c1) - fx0) / (im * epsilon)
                    end
                    c1[i] = x_old
                end
                df[i] -= im * imag(dfi)
            end
        end
    elseif fdtype == Val(:central)
        @inbounds for i ∈ eachindex(x)
            epsilon = compute_epsilon(fdtype, x[i], relstep, absstep, dir)
            x_old = x[i]
            c3[i] += epsilon
            dfi = f(c3)
            c3[i] = x_old - epsilon
            dfi -= f(c3)
            c3[i] = x_old
            df[i] = real(dfi / (2 * epsilon))
            if eltype(df) <: Complex
                if eltype(x) <: Complex
                    c3[i] += im * epsilon
                    dfi = f(c3)
                    c3[i] = x_old - im * epsilon
                    dfi -= f(c3)
                    c3[i] = x_old
                else
                    c1[i] += im * epsilon
                    dfi = f(c1)
                    c1[i] = x_old - im * epsilon
                    dfi -= f(c1)
                    c1[i] = x_old
                end
                df[i] -= im * imag(dfi / (2 * im * epsilon))
            end
        end
    elseif fdtype == Val(:complex) && returntype <: Real && eltype(df) <: Real && eltype(x) <: Real
        copyto!(c1, x)
        epsilon_complex = eps(real(eltype(x)))
        # we use c1 here to avoid typing issues with x
        @inbounds for i ∈ eachindex(x)
            c1_old = c1[i]
            c1[i] += im * epsilon_complex
            df[i] = imag(f(c1)) / epsilon_complex
            c1[i] = c1_old
        end
    else
        fdtype_error(returntype)
    end
    df
end

# vector of derivatives of a scalar->vector map
# this is effectively a vector of partial derivatives, but we still call it a gradient
function finite_difference_gradient!(
    df,
    f,
    x::Number,
    cache::GradientCache{T1,T2,T3,T4,fdtype,returntype,inplace};
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    dir=true) where {T1,T2,T3,T4,fdtype,returntype,inplace}

    # NOTE: in this case epsilon is a scalar, we need two arrays for fx1 and fx2
    # c1 denotes fx1, c2 is fx2, sizes guaranteed by the cache constructor
    fx, c1, c2 = cache.fx, cache.c1, cache.c2

    if inplace == Val(true)
        _c1, _c2 = c1, c2
    end

    if fdtype == Val(:forward)
        epsilon = compute_epsilon(Val(:forward), x, relstep, absstep, dir)
        if inplace == Val(true)
            f(c1, x + epsilon)
        else
            _c1 = f(x + epsilon)
        end
        if typeof(fx) != Nothing
            @. df = (_c1 - fx) / epsilon
        else
            if inplace == Val(true)
                f(c2, x)
            else
                _c2 = f(x)
            end
            @. df = (_c1 - _c2) / epsilon
        end
    elseif fdtype == Val(:central)
        epsilon = compute_epsilon(Val(:central), x, relstep, absstep, dir)
        if inplace == Val(true)
            f(c1, x + epsilon)
            f(c2, x - epsilon)
        else
            _c1 = f(x + epsilon)
            _c2 = f(x - epsilon)
        end
        @. df = (_c1 - _c2) / (2 * epsilon)
    elseif fdtype == Val(:complex) && returntype <: Real
        epsilon_complex = eps(real(eltype(x)))
        if inplace == Val(true)
            f(c1, x + im * epsilon_complex)
        else
            _c1 = f(x + im * epsilon_complex)
        end
        @. df = imag(_c1) / epsilon_complex
    else
        fdtype_error(returntype)
    end
    df
end
