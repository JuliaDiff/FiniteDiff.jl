mutable struct JVPCache{X1, FX1, FDType}
    x1  :: X1
    fx1 :: FX1
end

"""
    FiniteDiff.JVPCache(
        x,
        fdtype     :: Type{T1} = Val{:forward})

Allocating Cache Constructor.
"""
function JVPCache(
    x,
    fdtype::Union{Val{FD},Type{FD}} = Val(:forward)) where {FD}
    fdtype isa Type && (fdtype = fdtype())
    JVPCache{typeof(x), typeof(x), fdtype}(copy(x), copy(x))
end

"""
    FiniteDiff.JVPCache(
        x,
        fx1,
        fdtype     :: Type{T1} = Val{:forward},

Non-Allocating Cache Constructor.
"""
function JVPCache(
    x,
    fx,
    fdtype::Union{Val{FD},Type{FD}} = Val(:forward)) where {FD}
    fdtype isa Type && (fdtype = fdtype())
    JVPCache{typeof(x), typeof(fx), fdtype}(copy(x),fx)
end

"""
    FiniteDiff.finite_difference_jvp(
        f,
        x          :: AbstractArray{<:Number},
        v          :: AbstractArray{<:Number},
        fdtype     :: Type{T1}=Val{:central},
        relstep=default_relstep(fdtype, eltype(x)),
        absstep=relstep)

Cache-less.
"""
function finite_difference_jvp(f, x, v,
    fdtype     = Val(:forward),
    f_in       = nothing;
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    dir=true)

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
    f_in=nothing;
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    dir=true) where {X1, FX1, fdtype}

    if fdtype == Val(:complex)
        ArgumentError("finite_difference_jvp doesn't support :complex-mode finite diff")
    end

    tmp = sqrt(abs(dot(_vec(x), _vec(v))))
    epsilon = compute_epsilon(fdtype, tmp, relstep, absstep, dir)
    if fdtype == Val(:forward)
        fx = f_in isa Nothing ? f(x) : f_in
        x1 =  @. x + epsilon * v
        fx1 = f(x1)
        fx1 = @. (fx1-fx)/epsilon
    elseif fdtype == Val(:central)
        x1 = @. x + epsilon * v
        fx1 = f(x1)
        x1 = @. x - epsilon * v
        fx = f(x1)
        fx1 = @. (fx1-fx)/2epsilon
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
    fdtype     = Val(:forward),
    f_in       = nothing;
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep)
    if !isnothing(f_in)
        cache = JVPCache(x, f_in, fdtype)
    elseif fdtype == Val(:forward)
        fx = zero(x)
        f(fx,x)
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
        absstep=relstep,)

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

    (;x1, fx1) = cache
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
    resize!(cache.x1,  i)
    cache.fx1 !== nothing && resize!(cache.fx1, i)
    nothing
end
