struct HessianCache{T,fdtype,inplace}
    xpp::T
    xpm::T
    xmp::T
    xmm::T
end

function HessianCache(xpp,xpm,xmp,xmm,
                      fdtype::Type{T1}=Val{:hcentral},
                      inplace::Type{Val{T2}} = x isa StaticArray ? Val{true} : Val{false}) where {T1,T2}
    HessianCache{typeof(xpp),fdtype,inplace}(xpp,xpm,xmp,xmm)
end

function HessianCache(x,fdtype::Type{T1}=Val{:hcentral},
                        inplace::Type{Val{T2}} = x isa StaticArray ? Val{true} : Val{false}) where {T1,T2}
    HessianCache{typeof(x),fdtype,inplace}(copy(x),copy(x),copy(x),copy(x))
end

function finite_difference_hessian(f, x::AbstractArray{<:Number},
    fdtype     :: Type{T1}=Val{:hcentral},
    inplace    :: Type{Val{T2}} = x isa StaticArray ? Val{true} : Val{false};
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep) where {T1,T2}

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
    _H isa SMatrix ? SArray(H) : H
end

function finite_difference_hessian!(H::AbstractMatrix,f,
    x::AbstractArray{<:Number},
    fdtype     :: Type{T1}=Val{:hcentral},
    inplace    :: Type{Val{T2}} = x isa StaticArray ? Val{true} : Val{false};
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep) where {T1,T2}

    cache = HessianCache(x,fdtype,inplace)
    finite_difference_hessian!(H, f, x, cache; relstep=relstep, absstep=absstep)
end

function finite_difference_hessian!(H,f,x,
                                    cache::HessianCache{T,fdtype,inplace};
                                    relstep = default_relstep(fdtype, eltype(x)),
                                    absstep = relstep) where {T,fdtype,inplace}

    @assert fdtype == Val{:hcentral}
    n = length(x)
    xpp, xpm, xmp, xmm = cache.xpp, cache.xpm, cache.xmp, cache.xmm
    fx = f(x)

    for i = 1:n
        xi = x[i]
        epsilon = compute_epsilon(Val{:hcentral}, xi, relstep, absstep)

        if inplace === Val{true}
            xpp[i], xmm[i] = xi + epsilon, xi - epsilon
        else
            xpp = Base.setindex(xpp,xi + epsilon, i)
            xmm = Base.setindex(xmm,xi - epsilon, i)
        end

        H[i, i] = (f(xpp) - 2*fx + f(xmm)) / epsilon^2
        epsiloni = compute_epsilon(Val{:central}, xi, relstep, absstep)
        xp = xi + epsiloni
        xm = xi - epsiloni

        if inplace === Val{true}
            xpp[i], xpm[i], xmp[i], xmm[i] = xp, xp, xm, xm
        else
            xpp = Base.setindex(xpp,xp,i)
            xpm = Base.setindex(xpm,xp,i)
            xmp = Base.setindex(xmp,xm,i)
            xmm = Base.setindex(xmm,xm,i)
        end

        for j = i+1:n
            xj = x[j]
            epsilonj = compute_epsilon(Val{:central}, xj, relstep, absstep)
            xp = xj + epsilonj
            xm = xj - epsilonj

            if inplace === Val{true}
                xpp[j], xpm[j], xmp[j], xmm[j] = xp, xm, xp, xm
            else
                xpp = Base.setindex(xpp,xp,j)
                xpm = Base.setindex(xpm,xm,j)
                xmp = Base.setindex(xmp,xp,j)
                xmm = Base.setindex(xmm,xm,j)
            end

            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm))/(4*epsiloni*epsilonj)

            if inplace === Val{true}
                xpp[j], xpm[j], xmp[j], xmm[j] = xj, xj, xj, xj
            else
                xpp = Base.setindex(xpp,xj,j)
                xpm = Base.setindex(xpm,xj,j)
                xmp = Base.setindex(xmp,xj,j)
                xmm = Base.setindex(xmm,xj,j)
            end
        end

        if inplace === Val{true}
            xpp[i], xpm[i], xmp[i], xmm[i] = xi, xi, xi, xi
        else
            xpp = Base.setindex(xpp,xi,i)
            xpm = Base.setindex(xpm,xi,i)
            xmp = Base.setindex(xmp,xi,i)
            xmm = Base.setindex(xmm,xi,i)
        end

    end
    Symmetric(LinearAlgebra.copytri!(H,'U'))
end
