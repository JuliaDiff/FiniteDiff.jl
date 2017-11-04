function finite_difference!(df::AbstractArray{<:Real}, f, x::AbstractArray{<:Real},
    fdtype::DataType, ::Type{Val{:Real}}, ::Type{Val{:DiffEqDerivativeWrapper}},
    fx::Union{Void,AbstractArray{<:Real}}=nothing, epsilon::Union{Void,AbstractArray{<:Real}}=nothing, return_type::DataType=eltype(x))

    # TODO: test this one, and figure out what happens with epsilon
    fx1 = f.fx1
    if fdtype == Val{:forward}
        epsilon = compute_epsilon(Val{:forward}, x)
        f(fx, x)
        f(fx1, x+epsilon)
        @. df = (fx1 - fx) / epsilon
    elseif fdtype == Val{:central}
        epsilon = compute_epsilon(Val{:central}, x)
        f(fx, x-epsilon)
        f(fx1, x+epsilon)
        @. df = (fx1 - fx) / (2 * epsilon)
    elseif fdtype == Val{:complex}
        epsilon = eps(eltype(x))
        f(fx, f(x+im*epsilon))
        @. df = imag(fx) / epsilon
    end
    df
end

# AbstractArray{T} should be OK if JacobianWrapper is provided
function finite_difference_jacobian!(J::AbstractArray{T}, f, x::StridedArray{T}, ::Type{Val{:forward}}, fx::StridedArray{T}, ::Type{Val{:JacobianWrapper}}) where T<:Real
    m, n = size(J)
    epsilon_factor = compute_epsilon_factor(Val{:forward}, T)
    x1, fx1 = f.x1, f.fx1
    copy!(x1, x)
    copy!(fx1, fx)
    @inbounds for i in 1:n
        epsilon = compute_epsilon(Val{:forward}, x[i], epsilon_factor)
        epsilon_inv = one(T) / epsilon
        x1[i] += epsilon
        f(fx, x)
        f(fx1, x1)
        @. J[:,i] = (fx-fx1) * epsilon_inv
        x1[i] -= epsilon
    end
    J
end

function finite_difference_jacobian!(J::AbstractArray{T}, f, x::StridedArray{T}, ::Type{Val{:central}}, fx::StridedArray{T}, ::Type{Val{:JacobianWrapper}}) where T<:Real
    m, n = size(J)
    epsilon_factor = compute_epsilon_factor(Val{:central}, T)
    x1, fx1 = f.x1, f.fx1
    copy!(x1, x)
    copy!(fx1, fx)
    @inbounds for i in 1:n
        epsilon = compute_epsilon(Val{:central}, x[i], epsilon_factor)
        epsilon_double_inv = one(T) / (2 * epsilon)
        x[i] += epsilon
        x1[i] -= epsilon
        f(fx, x)
        f(fx1, x1)
        @. J[:,i] = (fx-fx1) * epsilon_double_inv
        x[i] -= epsilon
        x1[i] += epsilon
    end
    J
end
