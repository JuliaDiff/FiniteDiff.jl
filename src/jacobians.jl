mutable struct JacobianCache{CacheType1,CacheType2,CacheType3,ColorType,SparsityType,fdtype,returntype}
    x1  :: CacheType1
    fx  :: CacheType2
    fx1 :: CacheType3
    colorvec :: ColorType
    sparsity :: SparsityType
end

function JacobianCache(
    x,
    fdtype     :: Type{T1} = Val{:forward},
    returntype :: Type{T2} = eltype(x);
    inplace    :: Type{Val{T3}} = Val{true},
    colorvec = 1:length(x),
    sparsity = nothing) where {T1,T2,T3}

    if eltype(x) <: Real && fdtype==Val{:complex}
        x1  = false .* im .* x
        _fx = false .* im .* x
    else
        x1 = copy(x)
        _fx = copy(x)
    end

    if fdtype==Val{:complex}
        _fx1  = nothing
    else
        _fx1 = copy(x)
    end

    JacobianCache(x1,_fx,_fx1,fdtype,returntype;colorvec=colorvec,sparsity=sparsity)
end

function JacobianCache(
    x ,
    fx,
    fdtype     :: Type{T1} = Val{:forward},
    returntype :: Type{T2} = eltype(x);
    inplace    :: Type{Val{T3}} = Val{true},
    colorvec = 1:length(x),
    sparsity = nothing) where {T1,T2,T3}

    if eltype(x) <: Real && fdtype==Val{:complex}
        x1  = false .* im .* x
    else
        x1 = copy(x)
    end

    if eltype(fx) <: Real && fdtype==Val{:complex}
        _fx = false .* im .* fx
    else
        _fx = copy(fx)
    end

    if fdtype==Val{:complex}
        _fx1  = nothing
    else
        _fx1 = copy(fx)
    end

    JacobianCache(x1,_fx,_fx1,fdtype,returntype;colorvec=colorvec,sparsity=sparsity)
end

function JacobianCache(
    x1 ,
    fx ,
    fx1,
    fdtype     :: Type{T1} = Val{:forward},
    returntype :: Type{T2} = eltype(fx);
    inplace    :: Type{Val{T3}} = Val{true},
    colorvec = 1:length(x1),
    sparsity = nothing) where {T1,T2,T3}

    if fdtype==Val{:complex}
        !(returntype<:Real) && fdtype_error(returntype)

        if eltype(fx) <: Real
            _fx  = false .* im .* fx
        else
            _fx = fx
        end
        if eltype(x1) <: Real
            _x1  = false .* im .* x1
        else
            _x1 = x1
        end
    else
        _x1 = x1
        @assert eltype(fx) == T2
        @assert eltype(fx1) == T2
        _fx = fx
    end
    JacobianCache{typeof(_x1),typeof(_fx),typeof(fx1),typeof(colorvec),typeof(sparsity),fdtype,returntype}(_x1,_fx,fx1,colorvec,sparsity)
end

function _make_Ji(::SparseMatrixCSC, rows_index,cols_index,dx,colorvec,color_i,nrows,ncols)
    pick_inds = [i for i in 1:length(rows_index) if colorvec[cols_index[i]] == color_i]
    rows_index_c = rows_index[pick_inds]
    cols_index_c = cols_index[pick_inds]
    Ji = sparse(rows_index_c, cols_index_c, dx[rows_index_c],nrows,ncols)
    Ji
end

function _make_Ji(::AbstractArray, rows_index,cols_index,dx,colorvec,color_i,nrows,ncols)
    pick_inds = [i for i in 1:length(rows_index) if colorvec[cols_index[i]] == color_i]
    rows_index_c = rows_index[pick_inds]
    cols_index_c = cols_index[pick_inds]
    len_rows = length(pick_inds)
    unused_rows = setdiff(1:nrows,rows_index_c)
    perm_rows = sortperm(vcat(rows_index_c,unused_rows))
    cols_index_c = vcat(cols_index_c,zeros(Int,nrows-len_rows))[perm_rows]
    Ji = [j==cols_index_c[i] ? dx[i] : false for i in 1:nrows, j in 1:ncols]
    Ji
end

function _make_Ji(::SparseMatrixCSC, xtype, dx, color_i, nrows, ncols)
    Ji = sparse(1:nrows,fill(color_i,nrows),dx,nrows,ncols)
    Ji
end


function _make_Ji(::AbstractArray, xtype, dx, color_i, nrows, ncols)
    Ji = mapreduce(i -> i==color_i ? dx : zero(dx), hcat, 1:ncols)
    size(Ji)!=(nrows, ncols) ? reshape(Ji,(nrows,ncols)) : Ji #branch when size(dx) == (1,) => size(Ji) == (1,) while size(J) == (1,1)
end

function finite_difference_jacobian(f, x,
    fdtype     = Val{:forward},
    returntype = eltype(x),
    f_in       = nothing;
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    colorvec = 1:length(x),
    sparsity = nothing,
    jac_prototype = nothing,
    dir=true)

    if f_in isa Nothing
        fx = f(x)
    else
        fx = f_in
    end
    cache = JacobianCache(x, fx, fdtype, returntype)
    finite_difference_jacobian(f, x, cache, fx; relstep=relstep, absstep=absstep, colorvec=colorvec, sparsity=sparsity, jac_prototype=jac_prototype, dir=dir)
end

void_setindex!(args...) = (setindex!(args...); return)

function finite_difference_jacobian(
    f,
    x,
    cache::JacobianCache{T1,T2,T3,cType,sType,fdtype,returntype},
    f_in=nothing;
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    colorvec = cache.colorvec,
    sparsity = cache.sparsity,
    jac_prototype = nothing,
    dir=true) where {T1,T2,T3,cType,sType,fdtype,returntype}

    x1, fx, fx1 = cache.x1, cache.fx, cache.fx1

    if !(f_in isa Nothing)
        vecfx = _vec(f_in)
    elseif fdtype == Val{:forward}
        vecfx = _vec(f(x))
    elseif fdtype == Val{:complex} && returntype <: Real
        vecfx = real(fx)
    else
        vecfx = _vec(fx)
    end
    vecx = _vec(x)
    vecx1 = _vec(x1)
    nrows, ncols = length(vecfx), length(vecx)

    if !(sparsity isa Nothing)
        rows_index, cols_index = ArrayInterface.findstructralnz(sparsity)
        rows_index = [rows_index[i] for i in 1:length(rows_index)]
        cols_index = [cols_index[i] for i in 1:length(cols_index)]
    end

    if sparsity isa Nothing
        J = mapreduce(hcat,1:maximum(colorvec)) do color_i
            if fdtype == Val{:forward}
                x_save = ArrayInterface.allowed_getindex(vecx,color_i)
                epsilon = compute_epsilon(Val{:forward}, x_save, relstep, absstep, dir)
                _vecx1 = Base.setindex(vecx,x_save+epsilon,color_i)
                _x1 = reshape(_vecx1,size(x))
                vecfx1 = _vec(f(_x1))
                dx = (vecfx1-vecfx)/epsilon
            elseif fdtype == Val{:central}
                x1_save = ArrayInterface.allowed_getindex(vecx1,color_i)
                x_save = ArrayInterface.allowed_getindex(vecx,color_i)
                epsilon = compute_epsilon(Val{:forward}, x1_save, relstep, absstep, dir)
                _vecx1 = Base.setindex(vecx1,x1_save+epsilon,color_i)
                _vecx = Base.setindex(vecx,x_save-epsilon,color_i)
                _x1 = reshape(_vecx1,size(x))
                _x = reshape(_vecx,size(x))
                vecfx1 = _vec(f(_x1))
                vecfx = _vec(f(_x))
                dx = (vecfx1-vecfx)/(2epsilon)
            elseif fdtype == Val{:complex} && returntype <: Real
                epsilon = eps(eltype(x))
                x_save = ArrayInterface.allowed_getindex(vecx,color_i)
                _vecx = Base.setindex(complex.(vecx),x_save+im*epsilon,color_i)
                _x = reshape(_vecx,size(x))
                vecfx = _vec(f(_x))
                dx = imag(vecfx)/epsilon
            else
                fdtype_error(returntype)
            end
        end
    else
        cols = map(hcat,1:maximum(colorvec)) do color_i
            if fdtype == Val{:forward}
                tmp = norm(vecx .* (colorvec .== color_i))
                epsilon = compute_epsilon(Val{:forward}, sqrt(tmp), relstep, absstep, dir)
                _vecx = @. vecx + epsilon * (colorvec == color_i)
                _x = reshape(_vecx,size(x))
                vecfx1 = _vec(f(_x))
                dx = (vecfx1-vecfx)/epsilon
            elseif fdtype == Val{:central}
                tmp = norm(vecx1 .* (colorvec .== color_i))
                epsilon = compute_epsilon(Val{:forward}, sqrt(tmp), relstep, absstep, dir)
                _vecx1 = @. vecx1 + epsilon * (colorvec == color_i)
                _vecx = @. vecx - epsilon * (colorvec == color_i)
                _x1 = reshape(_vecx1,size(x))
                _x = reshape(_vecx, size(x))
                vecfx1 = _vec(f(_x1))
                vecfx = _vec(f(_x))
                dx = (vecfx1-vecfx)/(2epsilon)
            elseif fdtype == Val{:complex} && returntype <: Real
                epsilon = eps(eltype(x))
                _vecx = @. vecx + im * epsilon * (colorvec == color_i)
                _x = reshape(_vecx,size(x))
                vecfx = _vec(f(_x))
                dx = imag(vecfx)/epsilon
            else
                fdtype_error(returntype)
            end
        end
        J = jac_prototype isa Nothing ? (sparsity isa Nothing ? false.*vecfx.*vecx' : zeros(eltype(x),size(sparsity))) : zero(jac_prototype)
        for color_i in 1:maximum(colorvec)
            vfx1 = cols[color_i]
            if ArrayInterface.fast_scalar_indexing(x1)
                _colorediteration!(J,sparsity,rows_index,cols_index,vfx1,colorvec,color_i,n)
            else
                #=
                J.nzval[rows_index] .+= (colorvec[cols_index] .== color_i) .* vfx1[rows_index]
                or
                J[rows_index, cols_index] .+= (colorvec[cols_index] .== color_i) .* vfx1[rows_index]
                += means requires a zero'd out start
                =#
                if J isa SparseMatrixCSC
                    @. void_setindex!((J.nzval,),getindex((J.nzval,),rows_index) + (getindex((_color,),cols_index) == color_i) * getindex((vfx1,),rows_index),rows_index)
                else
                    @. void_setindex!((J,),getindex((J,),rows_index, cols_index) + (getindex((_color,),cols_index) == color_i) * getindex((vfx1,),rows_index),rows_index, cols_index)
                end
            end
        end
    end
    J
end

function finite_difference_jacobian!(J,
    f,
    x,
    fdtype     = Val{:forward},
    returntype = eltype(x),
    f_in       = nothing;
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    colorvec = 1:length(x),
    sparsity = ArrayInterface.has_sparsestruct(J) ? J : nothing)
    if f_in isa Nothing && fdtype == Val{:forward}
        if size(J,1) == length(x)
            fx = zero(x)
        else
            fx = zeros(returntype,size(J,1))
        end
        f(fx,x)
        cache = JacobianCache(x, fx, fdtype, returntype)
    elseif f_in isa Nothing
        cache = JacobianCache(x, fdtype, returntype)
    else
        cache = JacobianCache(x, f_in, fdtype, returntype)
    end
    finite_difference_jacobian!(J, f, x, cache, cache.fx; relstep=relstep, absstep=absstep, colorvec=colorvec, sparsity=sparsity)
end

function finite_difference_jacobian!(
    J,
    f,
    x,
    cache::JacobianCache{T1,T2,T3,cType,sType,fdtype,returntype},
    f_in = nothing;
    relstep = default_relstep(fdtype, eltype(x)),
    absstep = relstep,
    colorvec = cache.colorvec,
    sparsity = cache.sparsity,
    dir = true) where {T1,T2,T3,cType,sType,fdtype,returntype}

    m, n = size(J)
    _color = reshape(colorvec,size(x)...)

    x1, fx, fx1 = cache.x1, cache.fx, cache.fx1
    copyto!(x1, x)
    vfx = _vec(fx)

    rows_index = nothing
    cols_index = nothing
    if _use_findstructralnz(sparsity)
        rows_index, cols_index = ArrayInterface.findstructralnz(sparsity)
    end

    if sparsity !== nothing
        fill!(J,false)
    end

    if fdtype == Val{:forward}
        vfx1 = _vec(fx1)

        if f_in isa Nothing
            f(fx, x)
            vfx = _vec(fx)
        else
            vfx = _vec(f_in)
        end

        @inbounds for color_i ∈ 1:maximum(colorvec)
            if sparsity isa Nothing
                x1_save = ArrayInterface.allowed_getindex(x1,color_i)
                epsilon = compute_epsilon(Val{:forward}, x1_save, relstep, absstep, dir)
                ArrayInterface.allowed_setindex!(x1,x1_save + epsilon,color_i)
                f(fx1, x1)
                # J is dense, so either it is truly dense or this is the
                # compressed form of the coloring, so write into it.
                @. J[:,color_i] = (vfx1 - vfx) / epsilon
                # Now return x1 back to its original value
                ArrayInterface.allowed_setindex!(x1,x1_save,color_i)
            else # Perturb along the colorvec vector
                @. fx1 = x1 * (_color == color_i)
                tmp = norm(fx1)
                epsilon = compute_epsilon(Val{:forward}, sqrt(tmp), relstep, absstep, dir)
                @. x1 = x1 + epsilon * (_color == color_i)
                f(fx1, x1)
                # J is a sparse matrix, so decompress on the fly
                @. vfx1 = (vfx1 - vfx) / epsilon
                if ArrayInterface.fast_scalar_indexing(x1)
                    _colorediteration!(J,sparsity,rows_index,cols_index,vfx1,colorvec,color_i,n)
                else
                    #=
                    J.nzval[rows_index] .+= (colorvec[cols_index] .== color_i) .* vfx1[rows_index]
                    or
                    J[rows_index, cols_index] .+= (colorvec[cols_index] .== color_i) .* vfx1[rows_index]
                    += means requires a zero'd out start
                    =#
                    if J isa SparseMatrixCSC
                        @. void_setindex!((J.nzval,),getindex((J.nzval,),rows_index) + (getindex((_color,),cols_index) == color_i) * getindex((vfx1,),rows_index),rows_index)
                    else
                        @. void_setindex!((J,),getindex((J,),rows_index, cols_index) + (getindex((_color,),cols_index) == color_i) * getindex((vfx1,),rows_index),rows_index, cols_index)
                    end
                end
                # Now return x1 back to its original value
                @. x1 = x1 - epsilon * (_color == color_i)
            end
        end #for ends here
    elseif fdtype == Val{:central}
        vfx1 = _vec(fx1)
        @inbounds for color_i ∈ 1:maximum(colorvec)
            if sparsity isa Nothing
                x_save = ArrayInterface.allowed_getindex(x,color_i)
                x1_save = ArrayInterface.allowed_getindex(x1,color_i)
                epsilon = compute_epsilon(Val{:central}, x_save, relstep, absstep, dir)
                ArrayInterface.allowed_setindex!(x1,x1_save+epsilon,color_i)
                ArrayInterface.allowed_setindex!(x,x_save-epsilon,color_i)
                f(fx1, x1)
                f(fx, x)
                @. J[:,color_i] = (vfx1 - vfx) / 2epsilon
                ArrayInterface.allowed_setindex!(x1,x1_save,color_i)
                ArrayInterface.allowed_setindex!(x,x_save,color_i)
            else # Perturb along the colorvec vector
                @. fx1 = x1 * (_color == color_i)
                tmp = norm(fx1)
                epsilon = compute_epsilon(Val{:central}, sqrt(tmp), relstep, absstep, dir)
                @. x1 = x1 + epsilon * (_color == color_i)
                @. x  = x  - epsilon * (_color == color_i)
                f(fx1, x1)
                f(fx, x)
                @. vfx1 = (vfx1 - vfx) / 2epsilon
                if ArrayInterface.fast_scalar_indexing(x1)
                    _colorediteration!(J,sparsity,rows_index,cols_index,vfx1,colorvec,color_i,n)
                else
                    if J isa SparseMatrixCSC
                        @. void_setindex!((J.nzval,),getindex((J.nzval,),rows_index) + (getindex((_color,),cols_index) == color_i) * getindex((vfx1,),rows_index),rows_index)
                    else
                        @. void_setindex!((J,),getindex((J,),rows_index, cols_index) + (getindex((_color,),cols_index) == color_i) * getindex((vfx1,),rows_index),rows_index, cols_index)
                    end
                end
                @. x1 = x1 - epsilon * (_color == color_i)
                @. x  = x  + epsilon * (_color == color_i)
            end
        end
    elseif fdtype==Val{:complex} && returntype<:Real
        epsilon = eps(eltype(x))
        @inbounds for color_i ∈ 1:maximum(colorvec)
            if sparsity isa Nothing
                x1_save = ArrayInterface.allowed_getindex(x1,color_i)
                ArrayInterface.allowed_setindex!(x1,x1_save + im*epsilon, color_i)
                f(fx,x1)
                @. J[:,color_i] = imag(vfx) / epsilon
                ArrayInterface.allowed_setindex!(x1,x1_save,color_i)
            else # Perturb along the colorvec vector
                @. x1 = x1 + im * epsilon * (_color == color_i)
                f(fx,x1)
                @. vfx = imag(vfx) / epsilon
                if ArrayInterface.fast_scalar_indexing(x1)
                    _colorediteration!(J,sparsity,rows_index,cols_index,vfx,colorvec,color_i,n)
                else
                   if J isa SparseMatrixCSC
                        @. void_setindex!((J.nzval,),getindex((J.nzval,),rows_index) + (getindex((_color,),cols_index) == color_i) * getindex((vfx,),rows_index),rows_index)
                    else
                        @. void_setindex!((J,),getindex((J,),rows_index, cols_index) + (getindex((_color,),cols_index) == color_i) * getindex((vfx,),rows_index),rows_index, cols_index)
                    end
                end
                @. x1 = x1 - im * epsilon * (_color == color_i)
            end
        end
    else
        fdtype_error(returntype)
    end
    nothing
end

function resize!(cache::JacobianCache, i::Int)
    resize!(cache.x1,  i)
    resize!(cache.fx,  i)
    cache.fx1 != nothing && resize!(cache.fx1, i)
    cache.colorvec = 1:i
    nothing
end
