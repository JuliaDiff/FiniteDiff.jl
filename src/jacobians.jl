struct JacobianCache{CacheType1,CacheType2,CacheType3,fdtype,returntype,inplace}
    x1  :: CacheType1
    fx  :: CacheType2
    fx1 :: CacheType3
end

function JacobianCache(
    x,
    fdtype     :: Type{T1} = Val{:central},
    returntype :: Type{T2} = eltype(x),
    inplace    :: Type{Val{T3}} = Val{true}) where {T1,T2,T3}
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

    JacobianCache(x1,_fx,_fx1,fdtype,returntype,inplace)
end

function JacobianCache(
    x ,
    fx,
    fdtype     :: Type{T1} = Val{:central},
    returntype :: Type{T2} = eltype(x),
    inplace    :: Type{Val{T3}} = Val{true}) where {T1,T2,T3}

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

    JacobianCache(x1,_fx,_fx1,fdtype,returntype,inplace)
end

function JacobianCache(
    x1 ,
    fx ,
    fx1,
    fdtype     :: Type{T1} = Val{:central},
    returntype :: Type{T2} = eltype(fx),
    inplace    :: Type{Val{T3}} = Val{true}) where {T1,T2,T3}

    if fdtype==Val{:complex}
        !(returntype<:Real) && fdtype_error(returntype)

        if eltype(fx) <: Real
            _fx = fill(zero(Complex{eltype(x)}), size(fx))
        else
            _fx = fx
        end
        if eltype(x1) <: Real
            _x1 = fill(zero(Complex{eltype(x)}), size(x1))
        else
            _x1 = x1
        end
    else
        _x1 = x1
        @assert eltype(fx) == T2
        @assert eltype(fx1) == T2
        _fx = fx
    end
    JacobianCache{typeof(_x1),typeof(_fx),typeof(fx1),fdtype,returntype,inplace}(_x1,_fx,fx1)
end

function finite_difference_jacobian(f, x::AbstractArray{<:Number},
    fdtype     :: Type{T1}=Val{:central},
    returntype :: Type{T2}=eltype(x),
    inplace    :: Type{Val{T3}}=Val{true}) where {T1,T2,T3}

    cache = JacobianCache(x,fdtype,returntype,inplace)
    finite_difference_jacobian(f,x,cache)
end

function finite_difference_jacobian(f,x,cache::JacobianCache)
    J = fill(zero(eltype(x)), length(x), length(x))
    finite_difference_jacobian!(J,f,x,cache)
    J
end

function finite_difference_jacobian!(J::AbstractMatrix{<:Number},
    f,x::AbstractArray{<:Number},
    cache::JacobianCache{T1,T2,T3,fdtype,returntype,inplace}) where {T1,T2,T3,fdtype,returntype,inplace}

    m, n = size(J)
    x1, fx, fx1 = cache.x1, cache.fx, cache.fx1
    copyto!(x1, x)
    vfx = vec(fx)
    if fdtype == Val{:forward}
        vfx1 = vec(fx1)
        epsilon_factor = compute_epsilon_factor(Val{:forward}, eltype(x))
        @inbounds for i ∈ 1:n
            epsilon = compute_epsilon(Val{:forward}, x[i], epsilon_factor)
            x1_save = x1[i]
            x1[i] += epsilon
            if inplace == Val{true}
                f(fx1, x1)
                f(fx, x)
                J[:,i] = (vfx1 - vfx) / epsilon
            else
                fx1 .= f(x1)
                fx .= f(x)
                J[:,i] = (vfx1 - vfx) / epsilon
            end
            x1[i] = x1_save
        end
    elseif fdtype == Val{:central}
        vfx1 = vec(fx1)
        epsilon_factor = compute_epsilon_factor(Val{:central}, eltype(x))
        @inbounds for i ∈ 1:n
            epsilon = compute_epsilon(Val{:central}, x[i], epsilon_factor)
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
