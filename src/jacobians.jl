struct JacobianCache{CacheType,CacheType2,CacheType3,fdtype,RealOrComplex}
    x1::CacheType
    fx::CacheType2
    fx1::CacheType3
end

function JacobianCache(x1,fx,_fx1,fdtype::DataType=Val{:central},
                       RealOrComplex::DataType =
                       fdtype==Val{:complex} ? Val{:Real} : eltype(x1) <: Complex ?
                       Val{:Complex} : Val{:Real})
    if fdtype == Val{:complex} && _fx1 != nothing
        warn("fx1 cache is ignored when fdtype == Val{:complex}.")
        fx1 = nothing
    else
        fx1 = _fx1
    end
    JacobianCache{typeof(x1),typeof(fx),typeof(fx1),
                  fdtype,RealOrComplex}(x1,fx,fx1)
end

function finite_difference_jacobian(f,x,fdtype=Val{:central},
                    RealOrComplex::DataType =
                    fdtype==Val{:complex} ? Val{:Real} : eltype(x) <: Complex ?
                    Val{:Complex} : Val{:Real})
    x1 = similar(x)
    fx = similar(x)
    fx1 = similar(x)
    cache = JacobianCache(x1,fx,fx1,fdtype,RealOrComplex)
    finite_difference_jacobian(f,x,cache)
end

function finite_difference_jacobian(f,x,cache::JacobianCache)
    J = zeros(eltype(x), length(x), length(x))
    finite_difference_jacobian!(J,f,x,cache)
    J
end

function finite_difference_jacobian!(J::AbstractMatrix{<:Real}, f,
                     x::AbstractArray{<:Real},
                     cache::JacobianCache{CacheType,CacheType2,CacheType3,
                     fdtype,Val{:Real}}) where
                     {CacheType,CacheType2,CacheType3,fdtype}
    m, n = size(J)
    epsilon_elemtype = compute_epsilon_elemtype(nothing, x)
    x1, fx, fx1 = cache.x1, cache.fx, cache.fx1
    copy!(x1, x)
    vfx = vec(fx)
    if fdtype == Val{:forward}
        vfx1 = vec(fx1)
        epsilon_factor = compute_epsilon_factor(Val{:forward}, epsilon_elemtype)
        @inbounds for i ∈ 1:n
            epsilon = compute_epsilon(Val{:forward}, x[i], epsilon_factor)
            x1_save = x1[i]
            x1[i] += epsilon
            f(fx1, x1)
            f(fx, x)
            @. J[:,i] = (vfx1 - vfx) / epsilon
            x1[i] = x1_save
        end
    elseif fdtype == Val{:central}
        vfx1 = vec(fx1)
        epsilon_factor = compute_epsilon_factor(Val{:central}, epsilon_elemtype)
        @inbounds for i ∈ 1:n
            epsilon = compute_epsilon(Val{:central}, x[i], epsilon_factor)
            x1_save = x1[i]
            x_save = x[i]
            x1[i] += epsilon
            x[i] -= epsilon
            f(fx1, x1)
            f(fx, x)
            @. J[:,i] = (vfx1 - vfx) / (2*epsilon)
            x1[i] = x1_save
            x[i] = x_save
        end
    elseif fdtype == Val{:complex}
        epsilon = eps(eltype(real(x1))) # TODO: Remove in 1.0 when eps(x1) exists
        @inbounds for i ∈ 1:n
            x1_save = x1[i]
            x1[i] += im * epsilon
            f(fx,x1)
            @. J[:,i] = imag(vfx) / epsilon # Fix allocation
            x1[i] = x1_save
        end
    else
        fdtype_error(Val{:Real})
    end
    J
end

function finite_difference_jacobian!(J::AbstractMatrix{<:Number}, f,
                     x::AbstractArray{<:Number},
                     cache::JacobianCache{CacheType,CacheType2,CacheType3,
                     fdtype,Val{:Complex}}) where
                     {CacheType,CacheType2,CacheType3,fdtype}
    m, n = size(J)
    epsilon_elemtype = compute_epsilon_elemtype(nothing, x)
    x1, fx, fx1 = cache.x1, cache.fx, cache.fx1
    copy!(x1, x)
    vfx, vfx1 = vec(fx), vec(fx1)
    if fdtype == Val{:forward}
        epsilon_factor = compute_epsilon_factor(Val{:forward}, epsilon_elemtype)
        @inbounds for i ∈ 1:n
            epsilon = compute_epsilon(Val{:forward}, real(x[i]), epsilon_factor)
            x1_save = x1[i]
            x1[i] += epsilon
            f(fx1, x1)
            f(fx, x)
            @. J[:,i] = ( real( (vfx1 - vfx) ) + im*imag( (vfx1 - vfx) ) ) / epsilon
            x1[i] = x1_save
        end
    elseif fdtype == Val{:central}
        epsilon_factor = compute_epsilon_factor(Val{:central}, epsilon_elemtype)
        @inbounds for i ∈ 1:n
            epsilon = compute_epsilon(Val{:central}, real(x[i]), epsilon_factor)
            x1_save = x1[i]
            x_save = x[i]
            x1[i] += epsilon
            x[i] -= epsilon
            f(fx1, x1)
            f(fx, x)
            @. J[:,i] = ( real( (vfx1 - vfx) ) + im*imag(vfx1 - vfx) ) / (2*epsilon)
            x1[i] = x1_save
            x[i] = x_save
        end
    else
        fdtype_error(Val{:Complex})
    end
    J
end
