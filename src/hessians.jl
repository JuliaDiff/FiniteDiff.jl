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
    Symmetric(_H isa SMatrix ? SArray(H) : H)
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
        xi = ArrayInterface.allowed_getindex(x,i)
        epsilon = compute_epsilon(Val{:hcentral}, xi, relstep, absstep)

        if inplace === Val{true}
            ArrayInterface.allowed_setindex!(xpp,xi + epsilon,i)
            ArrayInterface.allowed_setindex!(xmm,xi - epsilon,i)
        else
            xpp = Base.setindex(xpp,xi + epsilon, i)
            xmm = Base.setindex(xmm,xi - epsilon, i)
        end

        ArrayInterface.allowed_setindex!(H,(f(xpp) - 2*fx + f(xmm)) / epsilon^2,i,i)
        epsiloni = compute_epsilon(Val{:central}, xi, relstep, absstep)
        xp = xi + epsiloni
        xm = xi - epsiloni

        if inplace === Val{true}
            ArrayInterface.allowed_setindex!(xpp,xp,i)
            ArrayInterface.allowed_setindex!(xpm,xp,i)
            ArrayInterface.allowed_setindex!(xmp,xm,i)
            ArrayInterface.allowed_setindex!(xmm,xm,i)
        else
            xpp = Base.setindex(xpp,xp,i)
            xpm = Base.setindex(xpm,xp,i)
            xmp = Base.setindex(xmp,xm,i)
            xmm = Base.setindex(xmm,xm,i)
        end

        for j = i+1:n
            xj = ArrayInterface.allowed_getindex(x,j)
            epsilonj = compute_epsilon(Val{:central}, xj, relstep, absstep)
            xp = xj + epsilonj
            xm = xj - epsilonj

            if inplace === Val{true}
                ArrayInterface.allowed_setindex!(xpp,xp,i)
                ArrayInterface.allowed_setindex!(xpm,xm,i)
                ArrayInterface.allowed_setindex!(xmp,xp,i)
                ArrayInterface.allowed_setindex!(xmm,xm,i)
            else
                xpp = Base.setindex(xpp,xp,j)
                xpm = Base.setindex(xpm,xm,j)
                xmp = Base.setindex(xmp,xp,j)
                xmm = Base.setindex(xmm,xm,j)
            end

            ArrayInterface.allowed_setindex!(H,(f(xpp) - f(xpm) - f(xmp) + f(xmm))/(4*epsiloni*epsilonj),i,j)

            if inplace === Val{true}
                ArrayInterface.allowed_setindex!(xpp,xj,j)
                ArrayInterface.allowed_setindex!(xpm,xj,j)
                ArrayInterface.allowed_setindex!(xmp,xj,j)
                ArrayInterface.allowed_setindex!(xmm,xj,j)
            else
                xpp = Base.setindex(xpp,xj,j)
                xpm = Base.setindex(xpm,xj,j)
                xmp = Base.setindex(xmp,xj,j)
                xmm = Base.setindex(xmm,xj,j)
            end
        end

        if inplace === Val{true}
            ArrayInterface.allowed_setindex!(xpp,xi,i)
            ArrayInterface.allowed_setindex!(xpm,xi,i)
            ArrayInterface.allowed_setindex!(xmp,xi,i)
            ArrayInterface.allowed_setindex!(xmm,xi,i)
        else
            xpp = Base.setindex(xpp,xi,i)
            xpm = Base.setindex(xpm,xi,i)
            xmp = Base.setindex(xmp,xi,i)
            xmm = Base.setindex(xmm,xi,i)
        end
    end
    LinearAlgebra.copytri!(H,'U')
end
