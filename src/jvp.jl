"""
    JVPCache{X1, FX1, FDType}

Cache structure for Jacobian-vector product (JVP) computations.

Stores temporary arrays needed for efficient JVP computation without repeated allocations.
The JVP computes `J(x) * v` where `J(x)` is the Jacobian of function `f` at point `x` 
and `v` is a vector.

# Fields
- `x1::X1`: Temporary array for perturbed input values
- `fx1::FX1`: Temporary array for function evaluations
"""
mutable struct JVPCache{X1, FX1, FDType}
    x1::X1
    fx1::FX1
end

"""
    FiniteDiff.JVPCache(
        x,
        fdtype::Type{T1} = Val{:forward})

Allocating cache constructor for Jacobian-vector product computations.

Creates a `JVPCache` by allocating temporary arrays with the same structure as `x`.
This constructor is convenient but allocates memory for the cache arrays.

# Arguments
- `x`: Input vector whose structure determines the cache array sizes
- `fdtype::Type{T1} = Val{:forward}`: Finite difference method type

# Returns
- `JVPCache` with allocated temporary arrays for JVP computation

# Examples
```julia
x = [1.0, 2.0, 3.0]
cache = JVPCache(x, Val(:forward))
```
"""
function JVPCache(
        x,
        fdtype::Union{Val{FD}, Type{FD}} = Val(:forward)) where {FD}
    fdtype isa Type && (fdtype = fdtype())
    JVPCache{typeof(x), typeof(x), fdtype}(copy(x), copy(x))
end

"""
    FiniteDiff.JVPCache(
        x,
        fx1,
        fdtype::Type{T1} = Val{:forward})

Non-allocating cache constructor for Jacobian-vector product computations.

Creates a `JVPCache` using pre-allocated arrays `x` and `fx1`. This constructor
is memory-efficient as it reuses existing arrays without additional allocation.

# Arguments
- `x`: Pre-allocated array for perturbed input values
- `fx1`: Pre-allocated array for function evaluations
- `fdtype::Type{T1} = Val{:forward}`: Finite difference method type

# Returns
- `JVPCache` using the provided arrays as cache storage

# Examples
```julia
x = [1.0, 2.0, 3.0]
fx1 = similar(x)
cache = JVPCache(x, fx1, Val(:forward))
```

# Notes
The arrays `x` and `fx1` will be modified during JVP computations. Ensure they
are not used elsewhere if their values need to be preserved.
"""
function JVPCache(
        x,
        fx,
        fdtype::Union{Val{FD}, Type{FD}} = Val(:forward)) where {FD}
    fdtype isa Type && (fdtype = fdtype())
    JVPCache{typeof(x), typeof(fx), fdtype}(x, fx)
end

"""
    FiniteDiff.finite_difference_jvp(
        f,
        x::AbstractArray{<:Number},
        v::AbstractArray{<:Number},
        fdtype::Type{T1}=Val{:forward},
        f_in=nothing;
        relstep=default_relstep(fdtype, eltype(x)),
        absstep=relstep)

Compute the Jacobian-vector product `J(x) * v` using finite differences.

This function computes the directional derivative of `f` at `x` in direction `v`
without explicitly forming the Jacobian matrix. This is more efficient than
computing the full Jacobian when only `J*v` is needed.

# Arguments
- `f`: Function to differentiate (vector→vector map)
- `x::AbstractArray{<:Number}`: Point at which to evaluate the Jacobian
- `v::AbstractArray{<:Number}`: Direction vector for the product
- `fdtype::Type{T1}=Val{:forward}`: Finite difference method (`:forward`, `:central`)
- `f_in=nothing`: Pre-computed `f(x)` value (if available)

# Keyword Arguments
- `relstep`: Relative step size (default: method-dependent optimal value)
- `absstep=relstep`: Absolute step size fallback

# Returns
- Vector `J(x) * v` representing the Jacobian-vector product

# Examples
```julia
f(x) = [x[1]^2 + x[2], x[1] * x[2], x[2]^3]
x = [1.0, 2.0]
v = [1.0, 0.0]  # Direction vector
jvp = finite_difference_jvp(f, x, v)  # Directional derivative
```

# Mathematical Background
The JVP is computed using the finite difference approximation:
- Forward: `J(x) * v ≈ (f(x + h*v) - f(x)) / h`
- Central: `J(x) * v ≈ (f(x + h*v) - f(x - h*v)) / (2h)`

where `h` is the step size and `v` is the direction vector.

# Notes
- Requires only `O(1)` or `O(2)` function evaluations (vs `O(n)` for full Jacobian)
- Forward differences: 1 extra function evaluation, `O(h)` accuracy
- Central differences: 2 extra function evaluations, `O(h²)` accuracy
- Particularly efficient when `v` is sparse or when only one directional derivative is needed
"""
function finite_difference_jvp(f, x, v,
        fdtype = Val(:forward),
        f_in = nothing;
        relstep = default_relstep(fdtype, eltype(x)),
        absstep = relstep,
        dir = true)
    if f_in isa Nothing
        fx = f(x)
    else
        fx = f_in
    end
    cache = JVPCache(x, fx, fdtype)
    finite_difference_jvp(f, x, v, cache, fx; relstep, absstep, dir)
end

"""
    FiniteDiff.finite_difference_jvp(
        f,
        x,
        v,
        cache::JVPCache;
        relstep=default_relstep(fdtype, eltype(x)),
        absstep=relstep,

Cached.
"""
function finite_difference_jvp(
        f,
        x,
        v,
        cache::JVPCache{X1, FX1, fdtype},
        f_in = nothing;
        relstep = default_relstep(fdtype, eltype(x)),
        absstep = relstep,
        dir = true) where {X1, FX1, fdtype}
    if fdtype == Val(:complex)
        ArgumentError("finite_difference_jvp doesn't support :complex-mode finite diff")
    end

    tmp = sqrt(abs(dot(_vec(x), _vec(v))))
    epsilon = compute_epsilon(fdtype, tmp, relstep, absstep, dir)
    if fdtype == Val(:forward)
        fx = f_in isa Nothing ? f(x) : f_in
        x1 = @. x + epsilon * v
        fx1 = f(x1)
        fx1 = @. (fx1-fx)/epsilon
    elseif fdtype == Val(:central)
        x1 = @. x + epsilon * v
        fx1 = f(x1)
        x1 = @. x - epsilon * v
        fx = f(x1)
        fx1 = @. (fx1-fx)/(2epsilon)
    else
        fdtype_error(eltype(x))
    end
    fx1
end

"""
    finite_difference_jvp!(
        jvp::AbstractArray{<:Number},
        f,
        x::AbstractArray{<:Number},
        v::AbstractArray{<:Number},
        fdtype     :: Type{T1}=Val{:forward},
        returntype :: Type{T2}=eltype(x),
        f_in       :: Union{T2,Nothing}=nothing;
        relstep=default_relstep(fdtype, eltype(x)),
        absstep=relstep)

Cache-less.
"""
function finite_difference_jvp!(jvp,
        f,
        x,
        v,
        fdtype = Val(:forward),
        f_in = nothing;
        relstep = default_relstep(fdtype, eltype(x)),
        absstep = relstep)
    if !isnothing(f_in)
        cache = JVPCache(x, f_in, fdtype)
    elseif fdtype == Val(:forward)
        fx = zero(x)
        f(fx, x)
        cache = JVPCache(x, fx, fdtype)
    else
        cache = JVPCache(x, fdtype)
    end
    finite_difference_jvp!(jvp, f, x, v, cache, cache.fx1; relstep, absstep)
end

"""
    FiniteDiff.finite_difference_jvp!(
        jvp::AbstractArray{<:Number},
        f,
        x::AbstractArray{<:Number},
        v::AbstractArray{<:Number},
        cache::JVPCache;
        relstep=default_relstep(fdtype, eltype(x)),
        absstep=relstep,
        dir=true)

Cached.
"""
function finite_difference_jvp!(
        jvp,
        f,
        x,
        v,
        cache::JVPCache{X1, FX1, fdtype},
        f_in = nothing;
        relstep = default_relstep(fdtype, eltype(x)),
        absstep = relstep,
        dir = true) where {X1, FX1, fdtype}
    if fdtype == Val(:complex)
        ArgumentError("finite_difference_jvp doesn't support :complex-mode finite diff")
    end

    (; x1, fx1) = cache
    tmp = sqrt(abs(dot(_vec(x), _vec(v))))
    epsilon = compute_epsilon(fdtype, tmp, relstep, absstep, dir)
    if fdtype == Val(:forward)
        if f_in isa Nothing
            f(fx1, x)
        else
            fx1 = f_in
        end
        @. x1 = x + epsilon * v
        f(jvp, x1)
        @. jvp = (jvp-fx1)/epsilon
    elseif fdtype == Val(:central)
        @. x1 = x - epsilon * v
        f(fx1, x1)
        @. x1 = x + epsilon * v
        f(jvp, x1)
        @. jvp = (jvp-fx1)/(2epsilon)
    else
        fdtype_error(eltype(x))
    end
    nothing
end

function resize!(cache::JVPCache, i::Int)
    resize!(cache.x1, i)
    cache.fx1 !== nothing && resize!(cache.fx1, i)
    nothing
end
