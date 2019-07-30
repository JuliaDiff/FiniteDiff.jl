mutable struct JacobianCache{CacheType1,CacheType2,CacheType3,ColorType,SparsityType,fdtype,returntype,inplace}
    x1  :: CacheType1
    fx  :: CacheType2
    fx1 :: CacheType3
    color :: ColorType
    sparsity :: SparsityType
end

function JacobianCache(
    x,
    fdtype     :: Type{T1} = Val{:forward},
    returntype :: Type{T2} = eltype(x),
    inplace    :: Type{Val{T3}} = Val{true};
    color = eachindex(x),
    sparsity = nothing) where {T1,T2,T3}

    if eltype(x) <: Real && fdtype==Val{:complex}
        if x isa StaticArray
            x1  = zeros(SVector{length(x),Complex{eltype(x)}})
            _fx = zeros(SVector{length(x),Complex{eltype(x)}})
        else
            x1 = fill(zero(Complex{eltype(x)}), size(x))
            _fx = fill(zero(Complex{eltype(x)}), size(x))
        end
    else
        x1 = copy(x)
        _fx = copy(x)
    end

    if fdtype==Val{:complex}
        _fx1  = nothing
    else
        _fx1 = copy(x)
    end

    JacobianCache(x1,_fx,_fx1,fdtype,returntype,inplace;color=color,sparsity=sparsity)
end

function JacobianCache(
    x ,
    fx,
    fdtype     :: Type{T1} = Val{:forward},
    returntype :: Type{T2} = eltype(x),
    inplace    :: Type{Val{T3}} = Val{true};
    color = eachindex(x),
    sparsity = nothing) where {T1,T2,T3}

    if eltype(x) <: Real && fdtype==Val{:complex}
        if x1 isa StaticArray
            x1 = zeros(SVector{length(x),Complex{eltype(x)}})
        else
            x1 = fill(zero(Complex{eltype(x)}), size(x))
        end
    else
        x1 = copy(x)
    end

    if eltype(fx) <: Real && fdtype==Val{:complex}
        if x1 isa StaticArray
            _fx = zeros(SVector{length(fx),Complex{eltype(x)}})
        else
            _fx = fill(zero(Complex{eltype(x)}), size(fx))
        end
    else
        _fx = copy(fx)
    end

    if fdtype==Val{:complex}
        _fx1  = nothing
    else
        _fx1 = copy(fx)
    end

    JacobianCache(x1,_fx,_fx1,fdtype,returntype,inplace;color=color,sparsity=sparsity)
end

function JacobianCache(
    x1 ,
    fx ,
    fx1,
    fdtype     :: Type{T1} = Val{:forward},
    returntype :: Type{T2} = eltype(fx),
    inplace    :: Type{Val{T3}} = Val{true};
    color = 1:length(x1),
    sparsity = nothing) where {T1,T2,T3}

    if fdtype==Val{:complex}
        !(returntype<:Real) && fdtype_error(returntype)

        if eltype(fx) <: Real
            if x1 isa StaticArray
                _fx = zeros(SVector{length(fx),Complex{eltype(x1)}})
            else
                _fx = fill(zero(Complex{eltype(x1)}), size(fx))
            end
        else
            _fx = fx
        end
        if eltype(x1) <: Real
            if x1 isa StaticArray
                _x1 = zeros(SVector{length(x1),Complex{eltype(x1)}})
            else
                _x1 = fill(zero(Complex{eltype(x1)}), size(x1))
            end
        else
            _x1 = x1
        end
    else
        _x1 = x1
        @assert eltype(fx) == T2
        @assert eltype(fx1) == T2
        _fx = fx
    end
    JacobianCache{typeof(_x1),typeof(_fx),typeof(fx1),typeof(color),typeof(sparsity),fdtype,returntype,inplace}(_x1,_fx,fx1,color,sparsity)
end

function finite_difference_jacobian!(J::AbstractMatrix,
    f,
    x::AbstractArray{<:Number},
    fdtype     :: Type{T1}=Val{:forward},
    returntype :: Type{T2}=eltype(x),
    inplace    :: Type{Val{T3}}=Val{true},
    f_in       :: Union{T2,Nothing}=nothing;
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    color = eachindex(x),
    sparsity = ArrayInterface.has_sparsestruct(J) ? J : nothing) where {T1,T2,T3}

    cache = JacobianCache(x, fdtype, returntype, inplace)
    finite_difference_jacobian!(J, f, x, cache, f_in; relstep=relstep, absstep=absstep, color=color, sparsity=sparsity)
end

function finite_difference_jacobian(f, x::AbstractArray{<:Number},
    fdtype     :: Type{T1}=Val{:forward},
    returntype :: Type{T2}=eltype(x),
    inplace    :: Type{Val{T3}}=Val{true},
    f_in       :: Union{T2,Nothing}=nothing;
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    color = eachindex(x),
    sparsity = nothing,
    dir=true) where {T1,T2,T3}

    cache = JacobianCache(x, fdtype, returntype, inplace)
    finite_difference_jacobian(f, x, cache, f_in; relstep=relstep, absstep=absstep, color=color, sparsity=sparsity, dir=dir)
end

function finite_difference_jacobian(
    f,
    x,
    cache::JacobianCache{T1,T2,T3,cType,sType,fdtype,returntype,inplace},
    f_in=nothing;
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    color = cache.color,
    sparsity = cache.sparsity,
    dir=true) where {T1,T2,T3,cType,sType,fdtype,returntype,inplace}
    _J = false .* x .* x'
    _J isa SMatrix ? J = MArray(_J) : J = _J
    finite_difference_jacobian!(J, f, x, cache, f_in; relstep=relstep, absstep=absstep, color=color, sparsity=sparsity, dir=dir)
    _J isa SMatrix ? SArray(J) : J
end

function finite_difference_jacobian!(
    J::AbstractMatrix{<:Number},
    f,
    x::AbstractArray{<:Number},
    cache::JacobianCache{T1,T2,T3,cType,sType,fdtype,returntype,inplace},
    f_in::Union{T2,Nothing}=nothing;
    relstep = default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    color = cache.color,
    sparsity::Union{AbstractArray,Nothing} = cache.sparsity,
    dir = true) where {T1,T2,T3,cType,sType,fdtype,returntype,inplace}

    m, n = size(J)
    x1, fx, fx1 = cache.x1, cache.fx, cache.fx1
    if inplace == Val{true}
        copyto!(x1, x)
    end
    vfx = vec(fx)

    if ArrayInterface.has_sparsestruct(sparsity)
        rows_index, cols_index = ArrayInterface.findstructralnz(sparsity)
    end

    if sparsity !== nothing && !ArrayInterface.fast_scalar_indexing(x1)
        fill!(J,false)
    end

    if fdtype == Val{:forward}
        vfx1 = vec(fx1)

        if f_in isa Nothing
            if inplace == Val{true}
                f(fx, x)
            else
                fx = f(x)
                vfx = vec(fx)
            end
        else
            vfx = vec(f_in)
        end

        @inbounds for color_i ∈ 1:maximum(color)

            if color isa Base.OneTo || color isa UnitRange || color isa StaticArrays.SOneTo # Dense matrix
                x1_save = ArrayInterface.allowed_getindex(x1,color_i)
                epsilon = compute_epsilon(Val{:forward}, x1_save, relstep, absstep, dir)
                if inplace == Val{true}
                    ArrayInterface.allowed_setindex!(x1,x1_save + epsilon,color_i)
                else
                    _x1 = Base.setindex(x1,x1_save+epsilon,color_i)
                end
            else # Perturb along the color vector
                @.. fx1 = x1 * (color == color_i)
                tmp = norm(fx1)
                epsilon = compute_epsilon(Val{:forward}, sqrt(tmp), relstep, absstep, dir)

                if inplace == Val{true}
                    @.. x1 = x1 + epsilon * (color == color_i)
                else
                    _x1 = @.. _x1 + epsilon * (color == color_i)
                end
            end

            if inplace == Val{true}
                f(fx1, x1)

                if sparsity isa Nothing
                    # J is dense, so either it is truly dense or this is the
                    # compressed form of the coloring, so write into it.
                    @.. J[:,color_i] = (vfx1 - vfx) / epsilon
                else
                    # J is a sparse matrix, so decompress on the fly
                    @.. vfx1 = (vfx1 - vfx) / epsilon

                    if ArrayInterface.fast_scalar_indexing(x1)
                        for i in 1:length(cols_index)
                            if color[cols_index[i]] == color_i
                                J[rows_index[i],cols_index[i]] = vfx1[rows_index[i]]
                            end
                        end
                    else
                        #=
                        J[rows_index, cols_index] .+= (color[cols_index] .== color_i) .* vfx1[rows_index]
                        += means requires a zero'd out start
                        =#
                        @.. setindex!((J,),getindex((J,),rows_index, cols_index) + (getindex((color,),cols_index) == color_i) * getindex((vfx1,),rows_index),rows_index, cols_index)
                    end
                end
            else
                fx1 = f(_x1)
                vfx1 = vec(fx1)
                if sparsity isa Nothing
                    # J is dense, so either it is truly dense or this is the
                    # compressed form of the coloring, so write into it.
                    J[:,color_i] = (vfx1 - vfx) / epsilon
                else
                    # J is a sparse matrix, so decompress on the fly
                    _vfx1 = (vfx1 - vfx) / epsilon

                    if ArrayInterface.fast_scalar_indexing(x1)
                        for i in 1:length(cols_index)
                            if color[cols_index[i]] == color_i
                                J[rows_index[i],cols_index[i]] = vfx1[rows_index[i]]
                            end
                        end
                    else
                        #=
                        J[rows_index, cols_index] .+= (color[cols_index] .== color_i) .* vfx1[rows_index]
                        += means requires a zero'd out start
                        =#
                        @.. setindex!((J,),getindex((J,),rows_index, cols_index) + (getindex((color,),cols_index) == color_i) * getindex((vfx1,),rows_index),rows_index, cols_index)
                    end
                end
            end

            # Now return x1 back to its original value
            if inplace == Val{true}
                if color isa Base.OneTo || color isa UnitRange || color isa StaticArrays.SOneTo #Dense matrix
                    ArrayInterface.allowed_setindex!(x1,x1_save,color_i)
                else
                    @.. x1 = x1 - epsilon * (color == color_i)
                end
            end

        end #for ends here
    elseif fdtype == Val{:central}
        vfx1 = vec(fx1)

        @inbounds for color_i ∈ 1:maximum(color)

            if color isa Base.OneTo || color isa UnitRange || color isa StaticArrays.SOneTo # Dense matrix
                x_save = ArrayInterface.allowed_getindex(x,color_i)
                x1_save = ArrayInterface.allowed_getindex(x1,color_i)
                epsilon = compute_epsilon(Val{:central}, x_save, relstep, absstep, dir)
                if inplace == Val{true}
                    ArrayInterface.allowed_setindex!(x1,x1_save+epsilon,color_i)
                    ArrayInterface.allowed_setindex!(x,x_save-epsilon,color_i)
                else
                    _x1 = Base.setindex(x1,x1_save+epsilon,color_i)
                    _x  = Base.setindex(x, x_save-epsilon, color_i)
                end
            else # Perturb along the color vector
                @.. fx1 = x1 * (color == color_i)
                tmp = norm(fx1)
                epsilon = compute_epsilon(Val{:central}, sqrt(tmp), relstep, absstep, dir)
                if inplace == Val{true}
                    @.. x1 = x1 + epsilon * (color == color_i)
                    @.. x  = x  - epsilon * (color == color_i)
                else
                    _x1 = @.. _x1 + epsilon * (color == color_i)
                    _x  = @.. _x  - epsilon * (color == color_i)
                end
            end

            if inplace == Val{true}
                f(fx1, x1)
                f(fx, x)

                if sparsity isa Nothing
                    # J is dense, so either it is truly dense or this is the
                    # compressed form of the coloring, so write into it.
                    @.. J[:,color_i] = (vfx1 - vfx) / 2epsilon
                else
                    # J is a sparse matrix, so decompress on the fly
                    @.. vfx1 = (vfx1 - vfx) / 2epsilon

                    if ArrayInterface.fast_scalar_indexing(x1)
                        for i in 1:length(cols_index)
                            if color[cols_index[i]] == color_i
                                J[rows_index[i],cols_index[i]] = vfx1[rows_index[i]]
                            end
                        end
                    else
                        #=
                        J[rows_index, cols_index] .+= (color[cols_index] .== color_i) .* vfx1[rows_index]
                        += means requires a zero'd out start
                        =#
                        @.. setindex!((J,),getindex((J,),rows_index, cols_index) + (getindex((color,),cols_index) == color_i) * getindex((vfx1,),rows_index),rows_index, cols_index)
                    end
                end

            else
                fx1 = f(_x1)
                fx = f(_x)
                vfx1 = vec(fx1)
                vfx  = vec(fx)

                if sparsity isa Nothing
                    # J is dense, so either it is truly dense or this is the
                    # compressed form of the coloring, so write into it.
                    J[:,color_i] = (vfx1 - vfx) / 2epsilon
                else
                    # J is a sparse matrix, so decompress on the fly
                    vfx1 = (vfx1 - vfx) / 2epsilon
                    # vfx1 is the compressed Jacobian column

                    if ArrayInterface.fast_scalar_indexing(x1)
                        for i in 1:length(cols_index)
                            if color[cols_index[i]] == color_i
                                J[rows_index[i],cols_index[i]] = vfx1[rows_index[i]]
                            end
                        end
                    else
                        #=
                        J[rows_index, cols_index] .+= (color[cols_index] .== color_i) .* vfx1[rows_index]
                        += means requires a zero'd out start
                        =#
                        @.. setindex!((J,),getindex((J,),rows_index, cols_index) + (getindex((color,),cols_index) == color_i) * getindex((vfx1,),rows_index),rows_index, cols_index)
                    end
                end
            end

            # Now return x1 back to its original value
            if inplace == Val{true}
                if color isa Base.OneTo || color isa UnitRange || color isa StaticArrays.SOneTo #Dense matrix
                    ArrayInterface.allowed_setindex!(x1,x1_save,color_i)
                    ArrayInterface.allowed_setindex!(x,x_save,color_i)
                else
                    @.. x1 = x1 - epsilon * (color == color_i)
                    @.. x  = x  + epsilon * (color == color_i)
                end
            end
        end
    elseif fdtype==Val{:complex} && returntype<:Real
        epsilon = eps(eltype(x))
        @inbounds for color_i ∈ 1:maximum(color)

            if color isa Base.OneTo || color isa UnitRange || color isa StaticArrays.SOneTo # Dense matrix
                x1_save = ArrayInterface.allowed_getindex(x1,color_i)
                if inplace == Val{true}
                    ArrayInterface.allowed_setindex!(x1,x1_save + im*epsilon, color_i)
                else
                    _x1 = setindex(x1,x1_save+im*epsilon,color_i)
                end
            else # Perturb along the color vector
                if inplace == Val{true}
                    @.. x1 = x1 + im * epsilon * (color == color_i)
                else
                    _x1 = @.. x1 + im * epsilon * (color == color_i)
                end
            end

            if inplace == Val{true}
                f(fx,x1)
                if sparsity isa Nothing
                    # J is dense, so either it is truly dense or this is the
                    # compressed form of the coloring, so write into it.
                    @.. J[:,color_i] = imag(vfx) / epsilon
                else
                    # J is a sparse matrix, so decompress on the fly
                    @.. vfx = imag(vfx) / epsilon

                    if ArrayInterface.fast_scalar_indexing(x1)
                        for i in 1:length(cols_index)
                            if color[cols_index[i]] == color_i
                                J[rows_index[i],cols_index[i]] = vfx1[rows_index[i]]
                            end
                        end
                    else
                        #=
                        J[rows_index, cols_index] .+= (color[cols_index] .== color_i) .* vfx1[rows_index]
                        += means requires a zero'd out start
                        =#
                        @.. setindex!((J,),getindex((J,),rows_index, cols_index) + (getindex((color,),cols_index) == color_i) * getindex((vfx1,),rows_index),rows_index, cols_index)
                    end
                end

            else
                fx = f(_x1)
                vfx = vec(fx)
                if sparsity isa Nothing
                    # J is dense, so either it is truly dense or this is the
                    # compressed form of the coloring, so write into it.
                    J[:,color_i] = imag(vfx) / epsilon
                else
                    # J is a sparse matrix, so decompress on the fly
                    vfx = imag(vfx) / epsilon

                    if ArrayInterface.fast_scalar_indexing(x1)
                        for i in 1:length(cols_index)
                            if color[cols_index[i]] == color_i
                                J[rows_index[i],cols_index[i]] = vfx1[rows_index[i]]
                            end
                        end
                    else
                        #=
                        J[rows_index, cols_index] .+= (color[cols_index] .== color_i) .* vfx1[rows_index]
                        += means requires a zero'd out start
                        =#
                        @.. setindex!((J,),getindex((J,),rows_index, cols_index) + (getindex((color,),cols_index) == color_i) * getindex((vfx1,),rows_index),rows_index, cols_index)
                    end
                end
            end

            if inplace == Val{true}
                # Now return x1 back to its original value
                if color isa Base.OneTo || color isa StaticArrays.SOneTo #Dense matrix
                    ArrayInterface.allowed_setindex!(x1,x1_save,color_i)
                else
                    @.. x1 = x1 - im * epsilon * (color == color_i)
                end
            end
        end
    else
        fdtype_error(returntype)
    end
    J
end

function resize!(cache::JacobianCache, i::Int)
    resize!(cache.x1,  i)
    resize!(cache.fx,  i)
    cache.fx1 != nothing && resize!(cache.fx1, i)
    cache.color = 1:i
    nothing
end
