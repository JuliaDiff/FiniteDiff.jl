struct HessianCache{T,fdtype,inplace}
    xpp::T
    xpm::T
    xmp::T
    xmm::T
end

"""
    _hessian_inplace(::Type{T}) where T
    _hessian_inplace(x)

Internal function to determine if Hessian computation should be performed in-place.

Returns `Val(true)` if the array type is mutable and supports in-place operations,
`Val(false)` otherwise. Used to dispatch on StaticArrays vs mutable arrays.

# Arguments
- `::Type{T}` or `x`: Array type or array instance

# Returns
- `Val(true)` if the array type supports in-place mutation
- `Val(false)` if the array type is immutable (e.g., StaticArray)
"""
_hessian_inplace(::Type{T}) where T = Val(ArrayInterface.ismutable(T))
_hessian_inplace(x) = _hessian_inplace(typeof(x))

"""
    __Symmetric(x)

Internal utility function that wraps a matrix in a `Symmetric` view.

# Arguments
- `x`: Matrix to be wrapped

# Returns
- `Symmetric(x)`: Symmetric view of the matrix
"""
__Symmetric(x) = Symmetric(x)

"""
    mutable_zeromatrix(x)

Internal utility function to create a mutable zero matrix with the same structure as `x`.

Creates a zero matrix compatible with `x` and ensures it's mutable for in-place operations.
If the created matrix is immutable, it converts it to a mutable copy.

# Arguments
- `x`: Array whose structure should be matched

# Returns
- Mutable zero matrix with the same dimensions and compatible type as `x`
"""
function mutable_zeromatrix(x)
    A = ArrayInterface.zeromatrix(x)
    ArrayInterface.ismutable(A) ? A : Base.copymutable(A)
end

"""
    HessianCache(
        xpp,
        xpm,
        xmp,
        xmm,
        fdtype::Type{T1}=Val{:hcentral},
        inplace::Type{Val{T2}} = x isa StaticArray ? Val{true} : Val{false})

Non-allocating cache constructor.
"""
function HessianCache(xpp,xpm,xmp,xmm,
                      fdtype=Val(:hcentral),
                      inplace = _hessian_inplace(x))
    fdtype isa Type && (fdtype = fdtype())
    inplace isa Type && (inplace = inplace())
    HessianCache{typeof(xpp),fdtype,inplace}(xpp,xpm,xmp,xmm)
end

"""
    HessianCache(
        x,
        fdtype::Type{T1}=Val{:hcentral},
        inplace::Type{Val{T2}} = x isa StaticArray ? Val{true} : Val{false})

Allocating cache constructor.
"""
function HessianCache(x, fdtype=Val(:hcentral),
                      inplace = _hessian_inplace(x))
    cx = copy(x)
    fdtype isa Type && (fdtype = fdtype())
    inplace isa Type && (inplace = inplace())
    HessianCache{typeof(cx),fdtype,inplace}(cx, copy(x), copy(x), copy(x))
end

"""
    finite_difference_hessian(
        f,
        x::AbstractArray{<:Number},
        fdtype     :: Type{T1}=Val{:hcentral},
        inplace    :: Type{Val{T2}} = x isa StaticArray ? Val{true} : Val{false};
        relstep=default_relstep(fdtype, eltype(x)),
        absstep=relstep)

Cache-less.
"""
function finite_difference_hessian(f, x,
    fdtype  = Val(:hcentral),
    inplace = _hessian_inplace(x);
    relstep = default_relstep(fdtype, eltype(x)),
    absstep = relstep)

    cache = HessianCache(x, fdtype, inplace)
    finite_difference_hessian(f, x, cache; relstep=relstep, absstep=absstep)
end

"""
    finite_difference_hessian(
        f,
        x,
        cache::HessianCache{T,fdtype,inplace};
        relstep=default_relstep(fdtype, eltype(x)),
        absstep=relstep)

Cached.
"""
function finite_difference_hessian(
    f,x,
    cache::HessianCache{T,fdtype,inplace};
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep) where {T,fdtype,inplace}
    H = mutable_zeromatrix(x)
    finite_difference_hessian!(H, f, x, cache; relstep=relstep, absstep=absstep)
    __Symmetric(H)
end

"""
    finite_difference_hessian!(
        H::AbstractMatrix,
        f,
        x::AbstractArray{<:Number},
        fdtype     :: Type{T1}=Val{:hcentral},
        inplace    :: Type{Val{T2}} = x isa StaticArray ? Val{true} : Val{false};
        relstep=default_relstep(fdtype, eltype(x)),
        absstep=relstep)

Cache-less.
"""
function finite_difference_hessian!(H,f,
    x,
    fdtype  = Val(:hcentral),
    inplace = _hessian_inplace(x);
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep)

    cache = HessianCache(x,fdtype,inplace)
    finite_difference_hessian!(H, f, x, cache; relstep=relstep, absstep=absstep)
end

"""
    finite_difference_hessian!(
        H,
        f,
        x,
        cache::HessianCache{T,fdtype,inplace};
        relstep = default_relstep(fdtype, eltype(x)),
        absstep = relstep)

Cached.
"""
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
        copyto!(xpp,x)
        copyto!(xpm,x)
        copyto!(xmp,x)
        copyto!(xmm,x)
    else # ignore the cache since immutable
        xpp, xpm, xmp, xmm = copy(x), copy(x), copy(x), copy(x)
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
