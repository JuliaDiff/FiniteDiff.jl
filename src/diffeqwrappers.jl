function finite_difference!(df::AbstractArray{<:Number}, f, x::Union{Number,AbstractArray{<:Number}},
    fdtype::DataType, funtype::DataType, ::Type{Val{:DiffEqDerivativeWrapper}},
    fx::Union{Void,AbstractArray{<:Number}}=nothing, epsilon::Union{Void,AbstractArray{<:Real}}=nothing, return_type::DataType=eltype(x))

    # TODO: optimized implementations for specific wrappers using the added DiffEq caching where appopriate

    finite_difference!(df, f, x, fdtype, funtype, Val{:Default}, fx, epsilon, return_type)
    df
end

function finite_difference_jacobian!(J::AbstractMatrix{<:Real}, f, x::AbstractArray{<:Real},
    fdtype::DataType, ::Type{Val{:Real}}, ::Type{Val{:JacobianWrapper}},
    fx::AbstractArray{<:Real}, epsilon::Union{Void,AbstractArray{<:Real}}=nothing, return_type::DataType=eltype(x))
    m, n = size(J)
    epsilon_elemtype = compute_epsilon_elemtype(epsilon, x)
    x1, fx1 = f.x1, f.fx1
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
        x0 = Complex{eltype(x)}.(x)
        cfx1 = Complex{eltype(x)}.(fx1)
        vcfx1 = vec(cfx1)
        epsilon = eps(eltype(x))
        @inbounds for i ∈ 1:n
            x0_save = x0[i]
            x0[i] += im * epsilon
            f(cfx1,x0)
            @. J[:,i] = imag(vcfx1) / epsilon # Fix allocation
            x0[i] = x0_save
        end
    else
        fdtype_error(Val{:Real})
    end
    J
end

function finite_difference_jacobian!(J::AbstractMatrix{<:Number}, f, x::AbstractArray{<:Number},
    fdtype::DataType, ::Type{Val{:Complex}}, ::Type{Val{:JacobianWrapper}},
    fx::AbstractArray{<:Number}, epsilon::Union{Void,AbstractArray{<:Real}}=nothing, return_type::DataType=eltype(x))

    # TODO: test this
    m, n = size(J)
    epsilon_elemtype = compute_epsilon_elemtype(epsilon, x)
    x1, fx1 = f.x1, f.fx1
    copy!(x1, x)
    vfx, vfx1 = vec(fx1),vec(fx)
    if fdtype == Val{:forward}
        epsilon_factor = compute_epsilon_factor(Val{:forward}, epsilon_elemtype)
        @inbounds for i ∈ 1:n
            epsilon = compute_epsilon(Val{:forward}, real(x[i]), epsilon_factor)
            x1[i] += epsilon
            f(fx1, x1)
            f(fx, x)
            @. J[:,i] = ( real( (vfx1 - vfx) ) + im*imag( (vfx1 - vfx) ) ) / epsilon
            x1[i] -= epsilon
        end
    elseif fdtype == Val{:central}
        epsilon_factor = compute_epsilon_factor(Val{:central}, epsilon_elemtype)
        @inbounds for i ∈ 1:n
            epsilon = compute_epsilon(Val{:central}, real(x[i]), epsilon_factor)
            x1[i] += epsilon
            x[i] -= epsilon
            f(fx1, x1)
            f(fx, x)
            @. J[:,i] = ( real( (vfx1 - vfx) ) + im*imag( vfx1 - vfx ) ) / (2*epsilon)
            x1[i] -= epsilon
            x[i] += epsilon
        end
    else
        fdtype_error(Val{:Complex})
    end
    J
end
