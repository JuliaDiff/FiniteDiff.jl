"""
    FiniteDiff.finite_difference_jvp(
        f,
        x,
        v,
        fdtype = Val(:forward),
        f_in=nothing;
        relstep=default_relstep(fdtype, eltype(x))
        absstep=relstep)
"""
function finite_difference_jvp(
    f,
    x,
    v
    fdtype = Val(:forward),
    f_in = nothing;
    relstep=default_relstep(eltype(x), eltype(x)),
    absstep=relstep,
    dir=true)
    if fdtype == Val(:complex)
        ArgumentError("finite_difference_jvp doesn't support :complex-mode finite diff")
    end
    vecx = _vec(x)
    vecv = _vec(v)

    tmp = sqrt(dot(vecx, vecv))
    epsilon = compute_epsilon(fdtype, sqrt(tmp), relstep, absstep, dir)
    if fdtype == Val(:forward)
        fx = f_in isa Nothing ? f(x) : f_in
        _x = @. x + epsilon * v
        fx1 = f(_x)
        return @. (fx1-fx)/epsilon
    elseif fdtype == Val(:central)
        _x = @. x + epsilon * v
        fx1 = f(_x)
        _x = @. x - epsilon * v
        fx = f(_x)
        return @. (fx1-fx)/(2epsilon)
    else
        fdtype_error(eltype(x))
    end
end

"""
    FiniteDiff.finite_difference_jvp!(
        jvp::AbstractArray{<:Number},
        f,
        x::AbstractArray{<:Number},
        v,
        fdtype = Val(:forward),
        f_in=nothing,
        fx1 = nothing;
        relstep=default_relstep(fdtype, eltype(x))
        absstep=relstep)
"""
function finite_difference_jvp!(
    jvp,
    f,
    x,
    v,
    fdtype = Val(:forward),
    f_in = nothing,
    fx1 = nothing;
    relstep = default_relstep(eltype(x), eltype(x)),
    absstep = relstep,
    dir = true)
    if fdtype == Val(:complex)
        ArgumentError("finite_difference_jvp doesn't support :complex-mode finite diff")
    end
    vecx = _vec(x)
    vecv = _vec(v)

    tmp = sqrt(dot(vecx, vecv))
    epsilon = compute_epsilon(fdtype, sqrt(tmp), relstep, absstep, dir)
    if fdtype == Val(:forward)
        if f_in isa Nothing
            fx1 = copy(jvp)
            f(fx1, x)
        else
            fx1 = f_in
        end
        @. x = x + epsilon * v
        f(jvp, x)
        @. jvp = (jvp-fx)/epsilon
    elseif fdtype == Val(:central)
        @. x = x - epsilon * v
        if fx1 isa Nothing
            fx1 = copy(jvp)
        end
        f(fx1, x)
        @. x = x + epsilon * v
        f(jvp, x)
        @. jvp = (jvp-fx1)/(2epsilon)
    else
        fdtype_error(eltype(x))
    end
    nothing
end
