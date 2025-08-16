struct GradientCache{
    CacheType1, CacheType2, CacheType3, CacheType4, fdtype, returntype, inplace}
    fx::CacheType1
    c1::CacheType2
    c2::CacheType3
    c3::CacheType4
end

"""
    FiniteDiff.GradientCache(
        df         :: Union{<:Number,AbstractArray{<:Number}},
        x          :: Union{<:Number, AbstractArray{<:Number}},
        fdtype     :: Type{T1} = Val{:central},
        returntype :: Type{T2} = eltype(df),
        inplace    :: Type{Val{T3}} = Val{true})

Allocating Cache Constructor
"""
function GradientCache(
        df,
        x,
        fdtype = Val(:central),
        returntype = eltype(df),
        inplace = Val(true))
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

    GradientCache{Nothing, typeof(_c1), typeof(_c2), typeof(_c3), fdtype,
        returntype, inplace}(nothing, _c1, _c2, _c3)
end

"""
    FiniteDiff.GradientCache(
        fx         :: Union{Nothing,<:Number,AbstractArray{<:Number}},
        c1         :: Union{Nothing,AbstractArray{<:Number}},
        c2         :: Union{Nothing,AbstractArray{<:Number}},
        c3         :: Union{Nothing,AbstractArray{<:Number}},
        fdtype     :: Type{T1} = Val{:central},
        returntype :: Type{T2} = eltype(fx),
        inplace    :: Type{Val{T3}} = Val{true})

Non-Allocating Cache Constructor

# Arguments 
- `fx`: Cached function call.
- `c1`, `c2`, `c3`: (Non-aliased) caches for the input vector.
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
        fdtype = Val(:central),
        returntype = eltype(fx),
        inplace = Val(true)) where {T, Fx} # Val(false) isn't so important for vector -> scalar, it gets ignored in that case anyway.
    GradientCache{Fx, T, T, T, fdtype, returntype, inplace}(fx, c1, c2, c3)
end

"""
    FiniteDiff.finite_difference_gradient(
        f,
        x,
        fdtype::Type{T1}=Val{:central},
        returntype::Type{T2}=eltype(x),
        inplace::Type{Val{T3}}=Val{true};
        relstep=default_relstep(fdtype, eltype(x)),
        absstep=relstep,
        dir=true)

Compute the gradient of function `f` at point `x` using finite differences.

This is the cache-less version that allocates temporary arrays internally.
Supports both vector→scalar maps `f(x) → scalar` and scalar→vector maps depending
on the `inplace` parameter and function signature.

# Arguments
- `f`: Function to differentiate
  - If `typeof(x) <: AbstractArray`: `f(x)` should return a scalar (vector→scalar gradient)
  - If `typeof(x) <: Number` and `inplace=Val(true)`: `f(fx, x)` modifies `fx` in-place (scalar→vector gradient)
  - If `typeof(x) <: Number` and `inplace=Val(false)`: `f(x)` returns a vector (scalar→vector gradient)
- `x`: Point at which to evaluate the gradient (vector or scalar)
- `fdtype::Type{T1}=Val{:central}`: Finite difference method (`:forward`, `:central`, `:complex`)
- `returntype::Type{T2}=eltype(x)`: Element type of gradient components
- `inplace::Type{Val{T3}}=Val{true}`: Whether to use in-place function evaluation

# Keyword Arguments
- `relstep`: Relative step size (default: method-dependent optimal value)
- `absstep=relstep`: Absolute step size fallback
- `dir=true`: Direction for step size (typically ±1)

# Returns
- Gradient vector `∇f` where `∇f[i] = ∂f/∂x[i]`

# Examples
```julia
# Vector→scalar gradient
f(x) = x[1]^2 + x[2]^2
x = [1.0, 2.0]
grad = finite_difference_gradient(f, x)  # [2.0, 4.0]

# Scalar→vector gradient (out-of-place)
g(t) = [t^2, t^3]
t = 2.0
grad = finite_difference_gradient(g, t, Val(:central), eltype(t), Val(false))
```

# Notes
- Forward differences: `O(n)` function evaluations, `O(h)` accuracy
- Central differences: `O(2n)` function evaluations, `O(h²)` accuracy
- Complex step: `O(n)` function evaluations, machine precision accuracy
"""
function finite_difference_gradient(
        f,
        x,
        fdtype = Val(:central),
        returntype = eltype(x),
        inplace = Val(true),
        fx = nothing,
        c1 = nothing,
        c2 = nothing;
        relstep = default_relstep(fdtype, eltype(x)),
        absstep = relstep,
        dir = true)
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
    finite_difference_gradient!(
        df, f, x, cache, relstep = relstep, absstep = absstep, dir = dir)
end

"""
    FiniteDiff.finite_difference_gradient!(
        df,
        f,
        x,
        fdtype::Type{T1}=Val{:central},
        returntype::Type{T2}=eltype(df),
        inplace::Type{Val{T3}}=Val{true};
        relstep=default_relstep(fdtype, eltype(x)),
        absstep=relstep)

Gradients are either a vector->scalar map `f(x)`, or a scalar->vector map `f(fx,x)` if `inplace=Val{true}` and `fx=f(x)` if `inplace=Val{false}`.

Cache-less.
"""
function finite_difference_gradient!(
        df,
        f,
        x,
        fdtype = Val(:central),
        returntype = eltype(df),
        inplace = Val(true),
        fx = nothing,
        c1 = nothing,
        c2 = nothing;
        relstep = default_relstep(fdtype, eltype(x)),
        absstep = relstep)
    cache = GradientCache(df, x, fdtype, returntype, inplace)
    finite_difference_gradient!(df, f, x, cache, relstep = relstep, absstep = absstep)
end

"""
    FiniteDiff.finite_difference_gradient!(
        df::AbstractArray{<:Number},
        f,
        x::AbstractArray{<:Number},
        cache::GradientCache;
        relstep=default_relstep(fdtype, eltype(x)),
        absstep=relstep
        dir=true)

Gradients are either a vector->scalar map `f(x)`, or a scalar->vector map `f(fx,x)` if `inplace=Val{true}` and `fx=f(x)` if `inplace=Val{false}`.

Cached.
"""
function finite_difference_gradient(
        f,
        x,
        cache::GradientCache{T1, T2, T3, T4, fdtype, returntype, inplace};
        relstep = default_relstep(fdtype, eltype(x)),
        absstep = relstep,
        dir = true) where {T1, T2, T3, T4, fdtype, returntype, inplace}
    if typeof(x) <: AbstractArray
        df = zero(returntype) .* x
    else
        df = zero(cache.c1)
    end
    finite_difference_gradient!(
        df, f, x, cache, relstep = relstep, absstep = absstep, dir = dir)
    df
end

# vector of derivatives of a vector->scalar map by each component of a vector x
# this ignores the value of "inplace", because it doesn't make much sense
#=
function finite_difference_gradient!(
        df,
        f,
        x,
        cache::GradientCache{T1, T2, T3, T4, fdtype, returntype, inplace};
        relstep = default_relstep(fdtype, eltype(x)),
        absstep = relstep,
        dir = true) where {T1, T2, T3, T4, fdtype, returntype, inplace}

    # NOTE: in this case epsilon is a vector, we need two arrays for epsilon and x1
    # c1 denotes x1, c2 is epsilon
    fx, c1, c2, c3 = cache.fx, cache.c1, cache.c2, cache.c3
    if fdtype != Val(:complex) && ArrayInterface.fast_scalar_indexing(c2)
        @. c2 = compute_epsilon(fdtype, one(eltype(x)), relstep, absstep, dir)
    end
    copyto!(c1, x)
    if fdtype == Val(:forward)
        @inbounds for i in eachindex(x)
            if ArrayInterface.fast_scalar_indexing(c2)
                epsilon = ArrayInterface.allowed_getindex(c2, i) * dir
            else
                epsilon = compute_epsilon(fdtype, one(eltype(x)), relstep, absstep, dir) *
                          dir
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
        copyto!(c3, x)
        @inbounds for i in eachindex(x)
            if ArrayInterface.fast_scalar_indexing(c2)
                epsilon = ArrayInterface.allowed_getindex(c2, i) * dir
            else
                epsilon = compute_epsilon(fdtype, one(eltype(x)), relstep, absstep, dir) *
                          dir
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
        # we use c1 here to avoid typing issues with x
        epsilon_complex = eps(real(eltype(x)))
        @inbounds for i in eachindex(x)
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
=#

function finite_difference_gradient!(
        df::StridedVector{<:Number},
        f,
        x::StridedVector{<:Number},
        cache::GradientCache{T1, T2, T3, T4, fdtype, returntype, inplace};
        relstep = default_relstep(fdtype, eltype(x)),
        absstep = relstep,
        dir = true) where {T1, T2, T3, T4, fdtype, returntype, inplace}

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
        fx0 = fx !== nothing ? fx : f(x)
        for i in eachindex(x)
            epsilon = compute_epsilon(fdtype, x[i], relstep, absstep, dir)
            x_old = x[i]
            c3[i] += epsilon
            dfi = (f(c3) - fx0) / epsilon
            c3[i] = x_old

            df[i] = real(dfi)
            if eltype(df) <: Complex
                if eltype(x) <: Complex
                    c3[i] += im * epsilon
                    dfi = (f(c3) - fx0) / (im * epsilon)
                    c3[i] = x_old
                else
                    c1[i] += im * epsilon
                    dfi = (f(c1) - fx0) / (im * epsilon)
                    c1[i] = x_old
                end
                df[i] -= im * imag(dfi)
            end
        end
    elseif fdtype == Val(:central)
        @inbounds for i in eachindex(x)
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
    elseif fdtype == Val(:complex) && returntype <: Real && eltype(df) <: Real &&
           eltype(x) <: Real
        copyto!(c1, x)
        epsilon_complex = eps(real(eltype(x)))
        # we use c1 here to avoid typing issues with x
        @inbounds for i in eachindex(x)
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
        cache::GradientCache{T1, T2, T3, T4, fdtype, returntype, inplace};
        relstep = default_relstep(fdtype, eltype(x)),
        absstep = relstep,
        dir = true) where {T1, T2, T3, T4, fdtype, returntype, inplace}

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
