struct JacobianCache{CacheType,CacheType2,fdtype,RealOrComplex,return_type}
    x1::CacheType
    fx1::CacheType2
end

JacobianCache(fdtype::DataType, RealOrComplex::DataType,
              return_type::DataType,x1,fx1)
    JacobianCache{typeof(x1),typeof(fx1),fdtype,
                  RealOrComplex,return_type}(x1,fx1)
end

function finite_difference_jacobian!(J::AbstractMatrix{<:Real}, f,
                     x::AbstractArray{<:Real},
                     cache::JacobianCache{CacheType,
                     fdtype,:Real,return_type}) where
                     {CacheType,fdtype,RealOrComplex,return_type}
    m, n = size(J)
    epsilon_elemtype = compute_epsilon_elemtype(epsilon, x)
    x1, fx1 = cache.x1, cache.fx1
    copy!(x1, x)
    vfx, vfx1 = vec(fx1),vec(fx)
    if fdtype == Val{:forward}
        epsilon_factor = compute_epsilon_factor(Val{:forward}, epsilon_elemtype)
        @inbounds for i ∈ 1:n
            epsilon = compute_epsilon(Val{:forward}, x[i], epsilon_factor)
            x1_save = x1[i]
            x1[i] += epsilon
            f(fx1, x1)
            f(fx, x)
            @. J[:,i] = (vfx - vfx1) / epsilon
            x1[i] = x1_save
        end
    elseif fdtype == Val{:central}
        epsilon_factor = compute_epsilon_factor(Val{:central}, epsilon_elemtype)
        @inbounds for i ∈ 1:n
            epsilon = compute_epsilon(Val{:central}, x[i], epsilon_factor)
            x1_save = x1[i]
            x_save = x[i]
            x1[i] += epsilon
            x[i] -= epsilon
            f(fx1, x1)
            f(fx, x)
            @. J[:,i] = (vfx - vfx1) / (2*epsilon)
            x1[i] = x1_save
            x[i] = x_save
        end
    elseif fdtype == Val{:complex}
        vfx1 = vec(fx1)
        epsilon = eps(eltype(x1))
        @inbounds for i ∈ 1:n
            x1_save = x1[i]
            x1[i] += im * epsilon
            f(fx1,x1)
            @. J[:,i] = imag(vfx1) / epsilon # Fix allocation
            x1[i] = x1_save
        end
    else
        fdtype_error(Val{:Real})
    end
    J
end

function finite_difference_jacobian!(J::AbstractMatrix{<:Number}, f,
                     x::AbstractArray{<:Number},
                     cache::JacobianCache{CacheType,
                     fdtype,:Complex,return_type}) where
                     {CacheType,fdtype,RealOrComplex,return_type}
    m, n = size(J)
    epsilon_elemtype = compute_epsilon_elemtype(epsilon, x)
    x1, fx1 = f.x1, f.fx1
    copy!(x1, x)
    vfx, vfx1 = vec(fx1),vec(fx)
    if fdtype == Val{:forward}
        epsilon_factor = compute_epsilon_factor(Val{:forward}, epsilon_elemtype)
        @inbounds for i ∈ 1:n
            epsilon = compute_epsilon(Val{:forward}, real(x[i]), epsilon_factor)
            x1_save = x1[i]
            x1[i] += epsilon
            f(fx1, x1)
            f(fx, x)
            @. J[:,i] = ( real( (vfx - vfx1) ) + im*imag( (vfx - vfx1) ) ) / epsilon
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
            @. J[:,i] = ( real( (vfx - vfx1) ) + im*imag(vfx - vfx1) ) / (2*epsilon)
            x1[i] = x1_save
            x[i] = x_save
        end
    else
        fdtype_error(Val{:Complex})
    end
    J
end
