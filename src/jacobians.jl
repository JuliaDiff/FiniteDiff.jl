struct JacobianCache{CacheType1,CacheType2,CacheType3,ColorType,fdtype,returntype,inplace}
    x1  :: CacheType1
    fx  :: CacheType2
    fx1 :: CacheType3
    color :: ColorType
end

function JacobianCache(
    x,
    fdtype     :: Type{T1} = Val{:central},
    returntype :: Type{T2} = eltype(x),
    inplace    :: Type{Val{T3}} = Val{true};
    color = eachindex(x)) where {T1,T2,T3}

    if eltype(x) <: Real && fdtype==Val{:complex}
        x1 = fill(zero(Complex{eltype(x)}), size(x))
        _fx = fill(zero(Complex{eltype(x)}), size(x))
    else
        x1 = similar(x)
        _fx = similar(x)
    end

    if fdtype==Val{:complex}
        _fx1  = nothing
    else
        _fx1 = similar(x)
    end

    JacobianCache(x1,_fx,_fx1,fdtype,returntype,inplace;color=color)
end

function JacobianCache(
    x ,
    fx,
    fdtype     :: Type{T1} = Val{:central},
    returntype :: Type{T2} = eltype(x),
    inplace    :: Type{Val{T3}} = Val{true};
    color = 1:length(x)) where {T1,T2,T3}

    if eltype(x) <: Real && fdtype==Val{:complex}
        x1 = fill(zero(Complex{eltype(x)}), size(x))
    else
        x1 = similar(x)
    end

    if eltype(fx) <: Real && fdtype==Val{:complex}
        _fx = fill(zero(Complex{eltype(x)}), size(fx))
    else
        _fx = similar(fx)
    end

    if fdtype==Val{:complex}
        _fx1  = nothing
    else
        _fx1 = similar(fx)
    end

    JacobianCache(x1,_fx,_fx1,fdtype,returntype,inplace;color=color)
end

function JacobianCache(
    x1 ,
    fx ,
    fx1,
    fdtype     :: Type{T1} = Val{:central},
    returntype :: Type{T2} = eltype(fx),
    inplace    :: Type{Val{T3}} = Val{true};
    color = 1:length(x1)) where {T1,T2,T3}

    if fdtype==Val{:complex}
        !(returntype<:Real) && fdtype_error(returntype)

        if eltype(fx) <: Real
            _fx = fill(zero(Complex{eltype(x1)}), size(fx))
        else
            _fx = fx
        end
        if eltype(x1) <: Real
            _x1 = fill(zero(Complex{eltype(x1)}), size(x1))
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

function finite_difference_jacobian(f, x::AbstractArray{<:Number},
    fdtype     :: Type{T1}=Val{:central},
    returntype :: Type{T2}=eltype(x),
    inplace    :: Type{Val{T3}}=Val{true},
    f_in       :: Union{T2,Nothing}=nothing;
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    color = 1:length(x)) where {T1,T2,T3}

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

    J = fill(zero(eltype(x)), length(x), length(x))
    finite_difference_jacobian!(J, f, x, cache, f_in; relstep=relstep, absstep=absstep, color=color)
    J
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
    copyto!(x1, x)
    vfx = vec(fx)
    if fdtype == Val{:forward}
        vfx1 = vec(fx1)
        @inbounds for color_i ∈ 1:maximum(color)

            if color isa UnitRange # Dense matrix
                epsilon = compute_epsilon(Val{:forward}, x1[color_i], relstep, absstep)
                save_x1 = x1[i]
                x1[i] += epsilon
            else # Perturb along the color vector
                tmp = zero(x[1])
                for i in 1:n
                    if color[i] == color_i
                        tmp += abs2(x1[i])
                    end
                end
                epsilon = compute_epsilon(Val{:forward}, sqrt(tmp), relstep, absstep)

                for i in 1:n
                    color[i] == color_i && (x1[i] += epsilon)
                end
            end

            if inplace == Val{true}
                f(fx1, x1)
                if f_in isa Nothing
                    f(fx, x)
                else
                    vfx = vec(f_in)
                end

                if J isa Matrix
                    # J is dense, so either it is truly dense or this is the
                    # compressed form of the coloring, so write into it.
                    @. J[:,color_i] = (vfx1 - vfx) / epsilon
                else
                    # J is a sparse matrix, so decompress on the fly
                    @. vfx1 = (vfx1 - vfx) / epsilon
                    # vfx1 is the compressed Jacobian column
                    # TODO
                end
            else
                fx1 .= f(x1)
                if f_in isa Nothing
                    fx .= f(x)
                else
                    vfx = vec(f_in)
                end
                if J isa Matrix
                    # J is dense, so either it is truly dense or this is the
                    # compressed form of the coloring, so write into it.
                    J[:,color_i] = (vfx1 - vfx) / epsilon
                else
                    # J is a sparse matrix, so decompress on the fly
                    vfx1 = (vfx1 - vfx) / epsilon
                    # vfx1 is the compressed Jacobian column
                    # TODO
                end
            end

            # Now return x1 back to its original value
            if color isa UnitRange #Dense matrix
                x1[i] = save_x1
            else
                for i in 1:n
                    color[i] == color_i && (x1[i] -= epsilon)
                end
            end

        end
    elseif fdtype == Val{:central}
        vfx1 = vec(fx1)
        @inbounds for i ∈ 1:n
            epsilon = compute_epsilon(Val{:central}, x[i], relstep, absstep)
            x1_save = x1[i]
            x_save = x[i]
            x1[i] += epsilon
            x[i]  -= epsilon
            if inplace == Val{true}
                f(fx1, x1)
                f(fx, x)
                @. J[:,i] = (vfx1 - vfx) / (2*epsilon)
            else
                fx1 .= f(x1)
                fx .= f(x)
                J[:,i] = (vfx1 - vfx) / (2*epsilon)
            end
            x1[i] = x1_save
            x[i]  = x_save
        end
    elseif fdtype==Val{:complex} && returntype<:Real
        epsilon = eps(eltype(x))
        @inbounds for i ∈ 1:n
            x1_save = x1[i]
            x1[i] += im * epsilon
            if inplace == Val{true}
                f(fx,x1)
                @. J[:,i] = imag(vfx) / epsilon
            else
                fx .= f(x1)
                J[:,i] = imag(vfx) / epsilon
            end
            x1[i] = x1_save
        end
    else
        fdtype_error(returntype)
    end
    J
end
