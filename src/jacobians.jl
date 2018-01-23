struct JacobianCache{CacheType1,CacheType2,CacheType3,fdtype,returntype,inplace}
    x1  :: CacheType1
    fx  :: CacheType2
    fx1 :: CacheType3
end

function JacobianCache(
    x          :: AbstractArray{<:Number},
    x1         :: Union{Void,AbstractArray{<:Number}} = nothing,
    fx         :: Union{Void,AbstractArray{<:Number}} = nothing,
    fx1        :: Union{Void,AbstractArray{<:Number}} = nothing,
    fdtype     :: Type{T1} = Val{:central},
    returntype :: Type{T2} = eltype(x),
    inplace    :: Type{Val{T3}} = Val{true}) where {T1,T2,T3}

    if fdtype==Val{:complex}
        if !(returntype<:Real)
            fdtype_error(returntype)
        end
        if eltype(fx)!=Complex{eltype(x)}
            _fx = zeros(Complex{eltype(x)}, size(x))
        end
        if typeof(fx1)!=Void
            warn("fdtype==Val{:complex} doesn't benefit from caching fx1.")
        end
        _fx1 = nothing
        if eltype(x1) != Complex{eltype(x)}
            _x1 = zeros(Complex{eltype(x)}, size(x))
        else
            _x1 = x1
        end
    else
        if eltype(x1) != eltype(x)
            _x1 = similar(x)
        else
            _x1 = x1
        end
        if eltype(fx) != returntype
            _fx = zeros(returntype, size(x))
        else
            _fx = fx
        end
        if eltype(fx1) != returntype
            _fx1 = zeros(returntype, size(x))
        else
            _fx1 = fx1
        end
    end
    JacobianCache{typeof(_x1),typeof(_fx),typeof(_fx1),fdtype,returntype,inplace}(_x1,_fx,_fx1)
end

function finite_difference_jacobian(f, x::AbstractArray{<:Number},
    fdtype     :: Type{T1}=Val{:central},
    returntype :: Type{T2}=eltype(x),
    inplace    :: Type{Val{T3}}=Val{true}) where {T1,T2,T3}

    cache = JacobianCache(x,nothing,nothing,nothing,fdtype,returntype,inplace)
    finite_difference_jacobian(f,x,cache)
end

function finite_difference_jacobian(f,x,cache::JacobianCache)
    J = zeros(eltype(x), length(x), length(x))
    finite_difference_jacobian!(J,f,x,cache)
    J
end

function finite_difference_jacobian!(J::AbstractMatrix{<:Number}, f,x::AbstractArray{<:Number},
    cache::JacobianCache{T1,T2,T3,fdtype,returntype,inplace}) where {T1,T2,T3,fdtype,returntype,inplace}

    m, n = size(J)
    x1, fx, fx1 = cache.x1, cache.fx, cache.fx1
    copy!(x1, x)
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
            else
                fx1 .= f(x1)
                fx .= f(x)
            end
            @. J[:,i] = (vfx1 - vfx) / epsilon
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
            else
                fx1 .= f(x1)
                fx .= f(x)
            end
            @. J[:,i] = (vfx1 - vfx) / (2*epsilon)
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
            else
                fx .= f(x1)
            end
            @. J[:,i] = imag(vfx) / epsilon
            x1[i] = x1_save
        end
    else
        fdtype_error(returntype)
    end
    J
end
