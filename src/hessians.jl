struct HessianCache{T,fdtype,inplace}
    xpp::T
    xpm::T
    xmp::T
    xmm::T
end

function HessianCache(xpp,xpm,xmp,xmm,
                      fdtype=Val(:hcentral),
                      inplace = x isa StaticArray ? Val(false) : Val(true))
    fdtype isa Type && (fdtype = fdtype())
    inplace isa Type && (inplace = inplace())
    HessianCache{typeof(xpp),fdtype,inplace}(xpp,xpm,xmp,xmm)
end

function HessianCache(x, fdtype=Val(:hcentral),
                      inplace = x isa StaticArray ? Val(false) : Val(true))
    cx = copy(x)
    fdtype isa Type && (fdtype = fdtype())
    inplace isa Type && (inplace = inplace())
    HessianCache{typeof(cx),fdtype,inplace}(cx, copy(x), copy(x), copy(x))
end

function finite_difference_hessian(f, x,
    fdtype  = Val(:hcentral),
    inplace = x isa StaticArray ? Val(false) : Val(true);
    relstep = default_relstep(fdtype, eltype(x)),
    absstep = relstep)

    cache = HessianCache(x, fdtype, inplace)
    finite_difference_hessian(f, x, cache; relstep=relstep, absstep=absstep)
end

function finite_difference_hessian(
    f,x,
    cache::HessianCache{T,fdtype,inplace};
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep) where {T,fdtype,inplace}
    _H = false .* x .* x'
    _H isa SMatrix ? H = MArray(_H) : H = _H
    finite_difference_hessian!(H, f, x, cache; relstep=relstep, absstep=absstep)
    Symmetric(_H isa SMatrix ? SArray(H) : H)
end

function finite_difference_hessian!(H,f,
    x,
    fdtype  = Val(:hcentral),
    inplace = x isa StaticArray ? Val(false) : Val(true);
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep)

    cache = HessianCache(x,fdtype,inplace)
    finite_difference_hessian!(H, f, x, cache; relstep=relstep, absstep=absstep)
end

function finite_difference_hessian!(H,f,x,
                                    cache::HessianCache{T,fdtype,inplace};
                                    relstep = default_relstep(fdtype, eltype(x)),
                                    absstep = relstep) where {T,fdtype,inplace}

    @assert fdtype == Val(:hcentral)
    n = length(x)
    xpp, xpm, xmp, xmm = cache.xpp, cache.xpm, cache.xmp, cache.xmm
    fx = f(x)

    if inplace === Val(true)
        _xpp, _xpm, _xmp, _xmm = xpp, xpm, xmp, xmm
    end

    for i = 1:n
        xi = ArrayInterface.allowed_getindex(x,i)
        epsilon = compute_epsilon(Val(:hcentral), xi, relstep, absstep)

        if inplace === Val(true)
            ArrayInterface.allowed_setindex!(xpp,xi + epsilon,i)
            ArrayInterface.allowed_setindex!(xmm,xi - epsilon,i)
        else
            _xpp = setindex(xpp,xi + epsilon, i)
            _xmm = setindex(xmm,xi - epsilon, i)
        end

        ArrayInterface.allowed_setindex!(H,(f(_xpp) - 2*fx + f(_xmm)) / epsilon^2,i,i)
        epsiloni = compute_epsilon(Val(:central), xi, relstep, absstep)
        xp = xi + epsiloni
        xm = xi - epsiloni

        if inplace === Val(true)
            ArrayInterface.allowed_setindex!(xpp,xp,i)
            ArrayInterface.allowed_setindex!(xpm,xp,i)
            ArrayInterface.allowed_setindex!(xmp,xm,i)
            ArrayInterface.allowed_setindex!(xmm,xm,i)
        else
            _xpp = setindex(xpp,xp,i)
            _xpm = setindex(xpm,xp,i)
            _xmp = setindex(xmp,xm,i)
            _xmm = setindex(xmm,xm,i)
        end

        for j = i+1:n
            xj = ArrayInterface.allowed_getindex(x,j)
            epsilonj = compute_epsilon(Val(:central), xj, relstep, absstep)
            xp = xj + epsilonj
            xm = xj - epsilonj

            if inplace === Val(true)
                ArrayInterface.allowed_setindex!(xpp,xp,j)
                ArrayInterface.allowed_setindex!(xpm,xm,j)
                ArrayInterface.allowed_setindex!(xmp,xp,j)
                ArrayInterface.allowed_setindex!(xmm,xm,j)
            else
                _xpp = setindex(_xpp,xp,j)
                _xpm = setindex(_xpm,xm,j)
                _xmp = setindex(_xmp,xp,j)
                _xmm = setindex(_xmm,xm,j)
            end

            ArrayInterface.allowed_setindex!(H,(f(_xpp) - f(_xpm) - f(_xmp) + f(_xmm))/(4*epsiloni*epsilonj),i,j)

            if inplace === Val(true)
                ArrayInterface.allowed_setindex!(xpp,xj,j)
                ArrayInterface.allowed_setindex!(xpm,xj,j)
                ArrayInterface.allowed_setindex!(xmp,xj,j)
                ArrayInterface.allowed_setindex!(xmm,xj,j)
            else
                _xpp = setindex(_xpp,xj,j)
                _xpm = setindex(_xpm,xj,j)
                _xmp = setindex(_xmp,xj,j)
                _xmm = setindex(_xmm,xj,j)
            end
        end

        if inplace === Val(true)
            ArrayInterface.allowed_setindex!(xpp,xi,i)
            ArrayInterface.allowed_setindex!(xpm,xi,i)
            ArrayInterface.allowed_setindex!(xmp,xi,i)
            ArrayInterface.allowed_setindex!(xmm,xi,i)
        end
    end
    LinearAlgebra.copytri!(H,'U')
end
