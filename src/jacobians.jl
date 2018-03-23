struct JacobianCache{CacheType1,CacheType2,CacheType3,fdtype,returntype,inplace}
    x1  :: CacheType1
    fx  :: CacheType2
    fx1 :: CacheType3
end

function JacobianCache(
    x,
               :: Val{fdtype} = Val{:central}(),
    returntype :: Type{T1} = eltype(x),
               :: Val{inplace} = Val{true}()) where {T1,fdtype,inplace}
    if eltype(x) <: Real && fdtype==:complex
        x1 = zeros(Complex{eltype(x)}, size(x))
        _fx = zeros(Complex{eltype(x)}, size(x))
    else
        x1 = similar(x)
        _fx = similar(x)
    end

    if fdtype==:complex
        _fx1  = nothing
    else
        _fx1 = similar(x)
    end

    JacobianCache(x1,_fx,_fx1,Val{fdtype}(),returntype,Val{inplace}())
end

function JacobianCache(
    x ,
    fx,
               :: Val{fdtype} = Val{:central}(),
    returntype :: Type{T1} = eltype(x),
               :: Val{inplace} = Val{true}()) where {T1,fdtype,inplace}

    if eltype(x) <: Real && fdtype==:complex
        x1 = zeros(Complex{eltype(x)}, size(x))
    else
        x1 = similar(x)
    end

    if eltype(fx) <: Real && fdtype==:complex
        _fx = zeros(Complex{eltype(x)}, size(fx))
    else
        _fx = similar(fx)
    end

    if fdtype==:complex
        _fx1  = nothing
    else
        _fx1 = similar(fx)
    end

    JacobianCache(x1,_fx,_fx1,Val{fdtype}(),returntype,Val{inplace}())
end

function JacobianCache(
    x1 ,
    fx ,
    fx1,
                :: Val{fdtype} = Val{:central}(),
    returntype  :: Type{T1} = eltype(x),
                :: Val{inplace} = Val{true}()) where {T1,fdtype,inplace}

    if fdtype==:complex
        !(returntype<:Real) && fdtype_error(returntype)

        if eltype(fx) <: Real
            _fx = zeros(Complex{eltype(x)}, size(fx))
        else
            _fx = fx
        end
        if eltype(x1) <: Real
            _x1 = zeros(Complex{eltype(x)}, size(x1))
        else
            _x1 = x1
        end
    else
        _x1 = x1
        @assert eltype(fx) == T1
        @assert eltype(fx1) == T1
        _fx = fx
    end
    JacobianCache{typeof(_x1),typeof(_fx),typeof(fx1),fdtype,returntype,inplace}(_x1,_fx,fx1)
end

function finite_difference_jacobian(f, x::AbstractArray{<:Number},
                :: Val{fdtype}=Val{:central}(),
    returntype  :: Type{T1}=eltype(x),
                :: Val{inplace}=Val{true}()) where {T1,fdtype,inplace}

    cache = JacobianCache(x,Val{fdtype}(),returntype,Val{inplace}())
    finite_difference_jacobian(f,x,cache)
end

function finite_difference_jacobian(f,x,cache::JacobianCache)
    J = zeros(eltype(x), length(x), length(x))
    finite_difference_jacobian!(J,f,x,cache)
    J
end

function finite_difference_jacobian!(J::AbstractMatrix{<:Number},
    f,x::AbstractArray{<:Number},
    cache::JacobianCache{T1,T2,T3,fdtype,returntype,inplace}) where {T1,T2,T3,fdtype,returntype,inplace}

    m, n = size(J)
    x1, fx, fx1 = cache.x1, cache.fx, cache.fx1
    copy!(x1, x)
    vfx = vec(fx)
    if fdtype == :forward
        vfx1 = vec(fx1)
        epsilon_factor = compute_epsilon_factor(Val{:forward}(), eltype(x))
        @inbounds for i ∈ 1:n
            epsilon = compute_epsilon(Val{:forward}(), x[i], epsilon_factor)
            x1_save = x1[i]
            x1[i] += epsilon
            if inplace == true
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
    elseif fdtype == :central
        vfx1 = vec(fx1)
        epsilon_factor = compute_epsilon_factor(Val{:central}(), eltype(x))
        @inbounds for i ∈ 1:n
            epsilon = compute_epsilon(Val{:central}(), x[i], epsilon_factor)
            x1_save = x1[i]
            x_save = x[i]
            x1[i] += epsilon
            x[i]  -= epsilon
            if inplace == true
                f(fx1, x1)
                f(fx, x)
                @. J[:,i] = (vfx1 - vfx) / (2*epsilon)
            else
                fx1 = f(x1)
                fx = f(x)
                J[:,i] = (vfx1 - vfx) / (2*epsilon)
            end
            x1[i] = x1_save
            x[i]  = x_save
        end
    elseif fdtype==:complex && returntype<:Real
        epsilon = eps(eltype(x))
        @inbounds for i ∈ 1:n
            x1_save = x1[i]
            x1[i] += im * epsilon
            if inplace == true
                f(fx,x1)
                @. J[:,i] = imag(vfx) / epsilon
            else
                fx = f(x1)
                J[:,i] = imag(vfx) / epsilon
            end
            x1[i] = x1_save
        end
    else
        fdtype_error(returntype)
    end
    J
end
