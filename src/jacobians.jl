mutable struct JacobianCache{CacheType1,CacheType2,CacheType3,ColorType,fdtype,returntype,inplace}
    x1  :: CacheType1
    fx  :: CacheType2
    fx1 :: CacheType3
    color :: ColorType
end

function JacobianCache(
    x,
    fdtype     :: Type{T1} = Val{:forward},
    returntype :: Type{T2} = eltype(x),
    inplace    :: Type{Val{T3}} = Val{true};
    color = eachindex(x)) where {T1,T2,T3}

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

    JacobianCache(x1,_fx,_fx1,fdtype,returntype,inplace;color=color)
end

function JacobianCache(
    x ,
    fx,
    fdtype     :: Type{T1} = Val{:forward},
    returntype :: Type{T2} = eltype(x),
    inplace    :: Type{Val{T3}} = Val{true};
    color = eachindex(x)) where {T1,T2,T3}

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

    JacobianCache(x1,_fx,_fx1,fdtype,returntype,inplace;color=color)
end

function JacobianCache(
    x1 ,
    fx ,
    fx1,
    fdtype     :: Type{T1} = Val{:forward},
    returntype :: Type{T2} = eltype(fx),
    inplace    :: Type{Val{T3}} = Val{true};
    color = 1:length(x1)) where {T1,T2,T3}

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
    JacobianCache{typeof(_x1),typeof(_fx),typeof(fx1),typeof(color),fdtype,returntype,inplace}(_x1,_fx,fx1,color)
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
    color = eachindex(x)) where {T1,T2,T3}

    cache = JacobianCache(x, fdtype, returntype, inplace)
    finite_difference_jacobian!(J, f, x, cache, f_in; relstep=relstep, absstep=absstep, color=color)
end

function finite_difference_jacobian(f, x::AbstractArray{<:Number},
    fdtype     :: Type{T1}=Val{:forward},
    returntype :: Type{T2}=eltype(x),
    inplace    :: Type{Val{T3}}=Val{true},
    f_in       :: Union{T2,Nothing}=nothing;
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    color = eachindex(x)) where {T1,T2,T3}

    cache = JacobianCache(x, fdtype, returntype, inplace)
    finite_difference_jacobian(f, x, cache, f_in; relstep=relstep, absstep=absstep, color=color)
end

function finite_difference_jacobian(
    f,
    x,
    cache::JacobianCache{T1,T2,T3,cType,fdtype,returntype,inplace},
    f_in=nothing;
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    color = cache.color) where {T1,T2,T3,cType,fdtype,returntype,inplace}
    _J = false .* x .* x'
    _J isa SMatrix ? J = MArray(_J) : J = _J
    finite_difference_jacobian!(J, f, x, cache, f_in; relstep=relstep, absstep=absstep, color=color)
    _J isa SMatrix ? SArray(J) : J
end

function finite_difference_jacobian!(
    J::AbstractMatrix{<:Number},
    f,
    x::AbstractArray{<:Number},
    cache::JacobianCache{T1,T2,T3,cType,fdtype,returntype,inplace},
    f_in::Union{T2,Nothing}=nothing;
    relstep = default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    color = cache.color) where {T1,T2,T3,cType,fdtype,returntype,inplace}

    m, n = size(J)
    x1, fx, fx1 = cache.x1, cache.fx, cache.fx1
    if inplace == Val{true}
        copyto!(x1, x)
    end
    vfx = vec(fx)

    if J isa SparseMatrixCSC
        (rows_index, cols_index, val) = findnz(J)
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

            if color isa Base.OneTo || color isa StaticArrays.SOneTo # Dense matrix
                epsilon = compute_epsilon(Val{:forward}, x1[color_i], relstep, absstep)
                x1_save = x1[color_i]
                if inplace == Val{true}
                    x1[color_i] += epsilon
                else
                    _x1 = Base.setindex(x1,x1[color_i]+epsilon,color_i)
                end
            else # Perturb along the color vector
                tmp = zero(x[1])
                for i in 1:n
                    if color[i] == color_i
                        tmp += abs2(x1[i])
                    end
                end
                epsilon = compute_epsilon(Val{:forward}, sqrt(tmp), relstep, absstep)

                if inplace != Val{true}
                    _x1 = copy(x1)
                end

                for i in 1:n
                    if color[i] == color_i
                        if inplace == Val{true}
                            x1[i] += epsilon
                        else
                            _x1 = Base.setindex(_x1,_x1[i]+epsilon,i)
                        end
                    end
                end
            end

            if inplace == Val{true}
                f(fx1, x1)

                if J isa Matrix || J isa MMatrix
                    # J is dense, so either it is truly dense or this is the
                    # compressed form of the coloring, so write into it.
                    @. J[:,color_i] = (vfx1 - vfx) / epsilon
                else
                    # J is a sparse matrix, so decompress on the fly
                    @. vfx1 = (vfx1 - vfx) / epsilon

                    for i in 1:length(cols_index)
                        if color[cols_index[i]] == color_i
                            J[rows_index[i],cols_index[i]] = vfx1[rows_index[i]]
                        end
                    end
                end
            else
                fx1 = f(_x1)
                vfx1 = vec(fx1)
                if J isa Matrix || J isa MMatrix
                    # J is dense, so either it is truly dense or this is the
                    # compressed form of the coloring, so write into it.
                    J[:,color_i] = (vfx1 - vfx) / epsilon
                else
                    # J is a sparse matrix, so decompress on the fly
                    _vfx1 = (vfx1 - vfx) / epsilon

                    for i in 1:length(cols_index)
                        if color[cols_index[i]] == color_i
                            J[rows_index[i],cols_index[i]] = _vfx1[rows_index[i]]
                        end
                    end
                end
            end

            # Now return x1 back to its original value
            if inplace == Val{true}
                if color isa Base.OneTo || color isa StaticArrays.SOneTo #Dense matrix
                    x1[color_i] = x1_save
                else
                    for i in 1:n
                        color[i] == color_i && (x1[i] -= epsilon)
                    end
                end
            end

        end #for ends here
    elseif fdtype == Val{:central}
        vfx1 = vec(fx1)

        @inbounds for color_i ∈ 1:maximum(color)

            if color isa Base.OneTo || color isa StaticArrays.SOneTo # Dense matrix
                epsilon = compute_epsilon(Val{:central}, x[color_i], relstep, absstep)
                x1_save = x1[color_i]
                x_save = x[color_i]
                if inplace == Val{true}
                    x1[color_i] += epsilon
                    x[color_i]  -= epsilon
                else
                    _x1 = Base.setindex(x1,x1[color_i]+epsilon,color_i)
                    _x  = Base.setindex(x, x[color_i]-epsilon, color_i)
                end
            else # Perturb along the color vector
                tmp = zero(x[1])
                for i in 1:n
                    if color[i] == color_i
                        tmp += abs2(x1[i])
                    end
                end
                epsilon = compute_epsilon(Val{:central}, sqrt(tmp), relstep, absstep)

                if inplace != Val{true}
                    _x1 = copy(x1)
                    _x  = copy(x)
                end

                for i in 1:n
                    if color[i] == color_i
                        if inplace == Val{true}
                            x1[i] += epsilon
                            x[i]  -= epsilon
                        else
                            _x1 = Base.setindex(_x1,_x1[i]+epsilon,i)
                            _x  = Base.setindex(_x,_x[i]-epsilon,i)
                        end
                    end
                end
            end

            if inplace == Val{true}
                f(fx1, x1)
                f(fx, x)

                if J isa Matrix || J isa MMatrix
                    # J is dense, so either it is truly dense or this is the
                    # compressed form of the coloring, so write into it.
                    @. J[:,color_i] = (vfx1 - vfx) / 2epsilon
                else
                    # J is a sparse matrix, so decompress on the fly
                    @. vfx1 = (vfx1 - vfx) / 2epsilon

                    for i in 1:length(cols_index)
                        if color[cols_index[i]] == color_i
                            J[rows_index[i],cols_index[i]] = vfx1[rows_index[i]]
                        end
                    end
                end

            else
                fx1 = f(_x1)
                fx = f(_x)
                vfx1 = vec(fx1)
                vfx  = vec(fx)

                if J isa Matrix || J isa MMatrix
                    # J is dense, so either it is truly dense or this is the
                    # compressed form of the coloring, so write into it.
                    J[:,color_i] = (vfx1 - vfx) / 2epsilon
                else
                    # J is a sparse matrix, so decompress on the fly
                    vfx1 = (vfx1 - vfx) / 2epsilon
                    # vfx1 is the compressed Jacobian column

                    for i in 1:length(cols_index)
                        if color[cols_index[i]] == color_i
                            J[rows_index[i],cols_index[i]] = vfx1[rows_index[i]]
                        end
                    end
                end
            end

            # Now return x1 back to its original value
            if inplace == Val{true}
                if color isa Base.OneTo || color isa StaticArrays.SOneTo #Dense matrix
                    x1[color_i] = x1_save
                    x[color_i]  = x_save
                else
                    for i in 1:n
                        if color[i] == color_i
                            x1[i] -= epsilon
                            x[i]  += epsilon
                        end
                    end
                end
            end
        end
    elseif fdtype==Val{:complex} && returntype<:Real
        epsilon = eps(eltype(x))
        @inbounds for color_i ∈ 1:maximum(color)

            if color isa Base.OneTo || color isa StaticArrays.SOneTo # Dense matrix
                x1_save = x1[color_i]
                if inplace == Val{true}
                    x1[color_i] += im*epsilon
                else
                    _x1 = setindex(x1,x1[color_i]+im*epsilon,color_i)
                end
            else # Perturb along the color vector
                if inplace != Val{true}
                    _x1 = copy(x1)
                end
                for i in 1:n
                    if color[i] == color_i
                        if inplace == Val{true}
                            x1[i] += im*epsilon
                        else
                            _x1 = setindex(_x1,_x1[i]+im*epsilon,i)
                        end
                    end
                end
            end

            if inplace == Val{true}
                f(fx,x1)
                if J isa Matrix || J isa MMatrix
                    # J is dense, so either it is truly dense or this is the
                    # compressed form of the coloring, so write into it.
                    @. J[:,color_i] = imag(vfx) / epsilon
                else
                    # J is a sparse matrix, so decompress on the fly
                    @. vfx = imag(vfx) / epsilon

                    for i in 1:length(cols_index)
                        if color[cols_index[i]] == color_i
                            J[rows_index[i],cols_index[i]] = vfx[rows_index[i]]
                        end
                    end
                end

            else
                fx = f(_x1)
                vfx = vec(fx)
                if J isa Matrix || J isa MMatrix
                    # J is dense, so either it is truly dense or this is the
                    # compressed form of the coloring, so write into it.
                    J[:,color_i] = imag(vfx) / epsilon
                else
                    # J is a sparse matrix, so decompress on the fly
                    vfx = imag(vfx) / epsilon

                    for i in 1:length(cols_index)
                        if color[cols_index[i]] == color_i
                            J[rows_index[i],cols_index[i]] = vfx[rows_index[i]]
                        end
                    end
                end
            end

            if inplace == Val{true}
                # Now return x1 back to its original value
                if color isa Base.OneTo || color isa StaticArrays.SOneTo #Dense matrix
                    x1[color_i] = x1_save
                else
                    for i in 1:n
                        color[i] == color_i && (x1[i] -= im*epsilon)
                    end
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
