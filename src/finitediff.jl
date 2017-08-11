#=
Very heavily inspired by Calculus.jl, but with an emphasis on performance and DiffEq API convenience.
=#

#=
Compute the finite difference interval epsilon.
Reference: Numerical Recipes, chapter 5.7.
=#
@inline function compute_epsilon{T<:Real}(::Type{Val{:forward}}, x::T, eps_sqrt::T=sqrt(eps(T)))
    eps_sqrt * max(one(T), abs(x))
end

@inline function compute_epsilon{T<:Real}(::Type{Val{:central}}, x::T, eps_cbrt::T=cbrt(eps(T)))
    eps_cbrt * max(one(T), abs(x))
end

@inline function compute_epsilon{T<:Complex}(::Type{Val{:complex}}, x::T)
    eps(real(x))
end


#=
Compute the derivative df of a real-valued callable f on a collection of points x.
Generic fallbacks for AbstractArrays that are not StridedArrays.
=#
function finite_difference{T<:Real}(f, x::AbstractArray{T}, ::Type{Val{:central}}, ::Union{Void,AbstractArray{T}}=nothing)
    df = zeros(T, size(x))
    finite_difference!(df, f, x, Val{:central})
end

function finite_difference!{T<:Real}(df::AbstractArray{T}, f, x::AbstractArray{T}, ::Type{Val{:central}}, ::Union{Void,AbstractArray{T}}=nothing)
    eps_sqrt = sqrt(eps(T))
    epsilon = compute_epsilon.(Val{:central}, x, eps_sqrt)
    @. df = (f(x+epsilon) - f(x-epsilon)) / (2 * epsilon)
end

function finite_difference{T<:Real}(f, x::AbstractArray{T}, ::Type{Val{:forward}}, f_x::AbstractArray{T}=f.(x))
    df = zeros(T, size(x))
    finite_difference!(df, f, x, Val{:forward}, f_x)
end

function finite_difference!{T<:Real}(df::AbstractArray{T}, f, x::AbstractArray{T}, ::Type{Val{:forward}}, f_x::AbstractArray{T}=f.(x))
    eps_cbrt = cbrt(eps(T))
    epsilon = compute_epsilon.(Val{:forward}, x, eps_cbrt)
    @. df = (f(x+epsilon) - f_x) / epsilon
end


#=
Compute the derivative df of a real-valued callable f on a collection of points x.
Optimized implementations for StridedArrays.
=#
function finite_difference{T<:Real}(f, x::StridedArray{T}, ::Type{Val{:central}}, ::Union{Void,StridedArray{T}}=nothing)
    df = zeros(T, size(x))
    finite_difference!(df, f, x, Val{:central})
end

function finite_difference!{T<:Real}(df::StridedArray{T}, f, x::StridedArray{T}, ::Type{Val{:central}}, ::Union{Void,StridedArray{T}}=nothing)
    eps_sqrt = sqrt(eps(T))
    @inbounds for i in 1 : length(x)
        epsilon = compute_epsilon(Val{:central}, x[i], eps_sqrt)
        epsilon_double_inv = one(T) / (2*epsilon)
        x_plus, x_minus = x[i]+epsilon, x[i]-epsilon
        df[i] = (f(x_plus) - f(x_minus)) * epsilon_double_inv
    end
    df
end

function finite_difference{T<:Real}(f, x::StridedArray{T}, ::Type{Val{:forward}}, fx::Union{Void,StridedArray{T}})
    df = zeros(T, size(x))
    if typeof(fx) == Void
        finite_difference!(df, f, x, Val{:forward})
    else
        finite_difference!(df, f, x, Val{:forward}, fx)
    end
    df
end

function finite_difference!{T<:Real}(df::StridedArray{T}, f, x::StridedArray{T}, ::Type{Val{:forward}})
    eps_cbrt = cbrt(eps(T))
    @inbounds for i in 1 : length(x)
        epsilon = compute_epsilon(Val{:forward}, x[i], eps_cbrt)
        epsilon_inv = one(T) / epsilon
        x_plus = x[i] + epsilon
        df[i] = (f(x_plus) - f(x[i])) * epsilon_inv
    end
    df
end

function finite_difference!{T<:Real}(df::StridedArray{T}, f, x::StridedArray{T}, ::Type{Val{:forward}}, fx::StridedArray{T})
    eps_cbrt = cbrt(eps(T))
    @inbounds for i in 1 : length(x)
        epsilon = compute_epsilon(Val{:forward}, x[i], eps_cbrt)
        epsilon_inv = one(T) / epsilon
        x_plus = x[i] + epsilon
        df[i] = (f(x_plus) - fx[i]) * epsilon_inv
    end
    df
end

#=
Compute the derivative df of a real-valued callable f on a collection of points x.
Single point implementations.
=#
function finite_difference{T<:Real}(f, x::T, t::DataType, f_x::Union{Void,T}=nothing)
    epsilon = compute_epsilon(t, x)
    finite_difference_kernel(f, x, t, epsilon, f_x)
end

@inline function finite_difference_kernel{T<:Real}(f, x::T, ::Type{Val{:forward}}, epsilon::T, f_x::Union{Void,T})
    if typeof(f_x) == Void
        return (f(x+epsilon) - f(x)) / epsilon
    else
        return (f(x+epsilon) - f_x) / epsilon
    end
end

@inline function finite_difference_kernel{T<:Real}(f, x::T, ::Type{Val{:central}}, epsilon::T, ::Union{Void,T}=nothing)
    (f(x+epsilon) - f(x-epsilon)) / (2 * epsilon)
end

# TODO: derivatives for complex-valued callables


#=
Compute the Jacobian matrix of a real-valued callable f: R^n -> R^m.
=#
function finite_difference_jacobian{T<:Real}(f, x::AbstractArray{T}, t::DataType)
    fx = f.(x)
    J = zeros(T, length(fx), length(x))
    finite_difference_jacobian!(J, f, x, t, fx)
end

function finite_difference_jacobian!{T<:Real}(J::AbstractArray{T}, f, x::AbstractArray{T}, t::DataType, fx::AbstractArray{T})
    m, n = size(J)
    if t == Val{:forward}
        shifted_x = copy(x)
        eps_sqrt = sqrt(eps(T))
        for i in 1 : n
            epsilon = compute_epsilon(t, x, eps_sqrt)
            shifted_x[i] += epsilon
            @. J[:, i] = (f(shifted_x) - f_x) / epsilon
            shifted_x[i] = x[i]
        end
    elseif t == Val{:central}
        shifted_x_plus = copy(x)
        shifted_x_minus = copy(x)
        eps_cbrt = cbrt(eps(T))
        for i in 1 : n
            epsilon = compute_epsilon(t, x, eps_cbrt)
            shifted_x_plus[i] += epsilon
            shifted_x_minus[i] -= epsilon
            @. J[:, i] = (f(shifted_x_plus) - f(shifted_x_minus)) / (epsilon + epsilon)
            shifted_x_plus[i] = x[i]
            shifted_x_minus[i] = x[i]
        end
    end
    J
end

function finite_difference_jacobian{T<:Real}(f, x::StridedArray{T}, t::DataType, fx::StridedArray{T})
    J = zeros(T, length(fx), length(x))
    finite_difference_jacobian!(J, f, x, t, fx)
end

function finite_difference_jacobian!{T<:Real}(J::StridedArray{T}, f, x::StridedArray{T}, ::Type{Val{:forward}}, fx::StridedArray{T})
    m, n = size(J)
    eps_sqrt = sqrt(eps(T))
    @inbounds for i = 1 : n
        epsilon = compute_epsilon(Val{:forward}, x[i], eps_sqrt)
        epsilon_inv = one(T) / epsilon
        for j in 1 : m
            if i == j
                J[j,i] = (f(x[j]+epsilon) - fx[j]) * epsilon_inv
            else
                J[j,i] = zero(T)
            end
        end
    end
    J
end

function finite_difference_jacobian!{T<:Real}(J::StridedArray{T}, f, x::StridedArray{T}, ::Type{Val{:central}}, ::Union{Void,StridedArray{T}}=nothing)
    m, n = size(J)
    eps_cbrt = cbrt(eps(T))
    @inbounds for i = 1 : n
        epsilon = compute_epsilon(Val{:central}, x[i], eps_cbrt)
        epsilon_double_inv = one(T) / (2 * epsilon)
        for j in 1 : m
            if i==j
                J[j,i] = (f(x[j]+epsilon) - f(x[j]-epsilon)) * epsilon_double_inv
            else
                J[j,i] = zero(T)
            end
        end
    end
    J
end

# TODO: Jacobians for complex-valued callables
