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

@inline function compute_epsilon{T<:Real}(::Type{Val{:complex}}, x::T, ::Union{Void,T}=nothing)
    eps(x)
end

@inline function compute_epsilon_factor{T<:Real}(fdtype::DataType, ::Type{T})
    if fdtype==Val{:forward}
        return sqrt(eps(T))
    elseif fdtype==Val{:central}
        return cbrt(eps(T))
    else
        error("Unrecognized fdtype: must be Val{:forward} or Val{:central}.")
    end
end


#=
Compute the derivative df of a real-valued callable f on a collection of points x.
Generic fallbacks for AbstractArrays that are not StridedArrays.
# TODO: test the fallbacks
=#
function finite_difference{T<:Real}(f, x::AbstractArray{T}, fdtype::DataType, fx::Union{Void,AbstractArray{T}}=nothing, funtype::DataType=Val{:Default})
    df = zeros(T, size(x))
    finite_difference!(df, f, x, fdtype, fx, funtype)
end

function finite_difference!{T<:Real}(df::AbstractArray{T}, f, x::AbstractArray{T}, fdtype::DataType, fx::Union{Void,AbstractArray{T}}, ::Type{Val{:Default}})
    epsilon_factor = compute_epsilon_factor(fdtype, T)
    @. epsilon = compute_epsilon(fdtype, x, epsilon_factor)
    if fdtype == Val{:forward}
        if typeof(fx) == Void
            @. df = (f(x+epsilon) - f(x)) / epsilon
        else
            @. df = (f(x+epsilon) - fx) / epsilon
        end
    elseif fdtype == Val{:central}
        @. df = (f(x+epsilon) - f(x-epsilon)) / (2 * epsilon)
    end
    df
end

function finite_difference!{T<:Real}(df::AbstractArray{T}, f, x::AbstractArray{T}, fdtype::DataType, fx::Union{Void,AbstractArray{T}}, ::Type{Val{:DiffEqDerivativeWrapper}})
    epsilon_factor = compute_epsilon_factor(fdtype, T)
    @. epsilon = compute_epsilon(fdtype, x, epsilon_factor)
    error("Not implemented yet.")

    if fdtype == Val{:forward}
        if typeof(fx) == Void

        else

        end
    elseif fdtype == Val{:central}

    end
    df
end

function finite_difference!{T<:Real}(df::AbstractArray{T}, f, x::T, fdtype::DataType, fx::AbstractArray{T}, ::Type{Val{:DiffEqDerivativeWrapper}})
    epsilon = compute_epsilon(fdtype, x)
    fx1 = f.fx1
    if fdtype == Val{:forward}
        f(fx, x)
        f(fx1, x+epsilon)
        @. df = (fx1 - fx) / epsilon
    elseif fdtype == Val{:central}
        f(fx, x-epsilon)
        f(fx1, x+epsilon)
        @. df = (fx1 - fx) / (2 * epsilon)
    end
    df
end

#=
Compute the derivative df of a real-valued callable f on a collection of points x.
Optimized implementations for StridedArrays.
=#
function finite_difference!{T<:Real}(df::StridedArray{T}, f, x::StridedArray{T}, ::Type{Val{:central}}, ::Union{Void,StridedArray{T}}, ::Type{Val{:Default}})
    epsilon_factor = compute_epsilon_factor(Val{:central}, T)
    @inbounds for i in 1 : length(x)
        epsilon = compute_epsilon(Val{:central}, x[i], epsilon_factor)
        epsilon_double_inv = one(T) / (2*epsilon)
        x_plus, x_minus = x[i]+epsilon, x[i]-epsilon
        df[i] = (f(x_plus) - f(x_minus)) * epsilon_double_inv
    end
    df
end

function finite_difference!{T<:Real}(df::StridedArray{T}, f, x::StridedArray{T}, ::Type{Val{:forward}}, ::Void, ::Type{Val{:Default}})
    epsilon_factor = compute_epsilon_factor(Val{:forward}, T)
    @inbounds for i in 1 : length(x)
        epsilon = compute_epsilon(Val{:forward}, x[i], epsilon_factor)
        epsilon_inv = one(T) / epsilon
        x_plus = x[i] + epsilon
        df[i] = (f(x_plus) - f(x[i])) * epsilon_inv
    end
    df
end

function finite_difference!{T<:Real}(df::StridedArray{T}, f, x::StridedArray{T}, ::Type{Val{:forward}}, fx::StridedArray{T}, ::Type{Val{:Default}})
    epsilon_factor = compute_epsilon_factor(Val{:forward}, T)
    @inbounds for i in 1 : length(x)
        epsilon = compute_epsilon(Val{:forward}, x[i], epsilon_factor)
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
function finite_difference{T<:Real}(f, x::T, fdtype::DataType, f_x::Union{Void,T}=nothing)
    epsilon = compute_epsilon(fdtype, x)
    finite_difference_kernel(f, x, fdtype, epsilon, f_x)
end

@inline function finite_difference_kernel{T<:Real}(f, x::T, ::Type{Val{:forward}}, epsilon::T, fx::Union{Void,T})
    if typeof(fx) == Void
        return (f(x+epsilon) - f(x)) / epsilon
    else
        return (f(x+epsilon) - fx) / epsilon
    end
end

@inline function finite_difference_kernel{T<:Real}(f, x::T, ::Type{Val{:central}}, epsilon::T, ::Union{Void,T}=nothing)
    (f(x+epsilon) - f(x-epsilon)) / (2 * epsilon)
end

# TODO: derivatives for complex-valued callables


#=
Compute the Jacobian matrix of a real-valued callable f: R^n -> R^m.
=#
function finite_difference_jacobian{T<:Real}(f, x::AbstractArray{T}, fdtype::DataType=Val{:central}, funtype::DataType=Val{:Default})
    if funtype==Val{:Default}
        fx = f.(x)
    elseif funtype==Val{:DiffEqJacobianWrapper}
        f(fx, x)
    else
        error("Unrecognized funtype: must be Val{:Default} or Val{:DiffEqJacobianWrapper}.")
    end
    J = zeros(T, length(fx), length(x))
    finite_difference_jacobian!(J, f, x, fdtype, fx, funtype)
end

function finite_difference_jacobian!{T<:Real}(J::AbstractArray{T}, f, x::AbstractArray{T}, fdtype::DataType, fx::AbstractArray{T}, ::DataType)
    # This is an inefficient fallback that only makes sense if setindex/getindex are unavailable, e.g. GPUArrays etc.
    m, n = size(J)
    epsilon_factor = compute_epsilon_factor(fdtype, T)
    if t == Val{:forward}
        shifted_x = copy(x)
        for i in 1:n
            epsilon = compute_epsilon(t, x[i], epsilon_factor)
            shifted_x[i] += epsilon
            J[:, i] .= (f(shifted_x) - f_x) / epsilon
            shifted_x[i] = x[i]
        end
    elseif t == Val{:central}
        shifted_x_plus  = copy(x)
        shifted_x_minus = copy(x)
        for i in 1:n
            epsilon = compute_epsilon(t, x[i], epsilon_factor)
            shifted_x_plus[i]  += epsilon
            shifted_x_minus[i] -= epsilon
            J[:, i] .= (f(shifted_x_plus) - f(shifted_x_minus)) / (epsilon + epsilon)
            shifted_x_plus[i]  = x[i]
            shifted_x_minus[i] = x[i]
        end
    else
        error("Unrecognized fdtype: must be Val{:forward} or Val{:central}.")
    end
    J
end

function finite_difference_jacobian!{T<:Real}(J::StridedArray{T}, f, x::StridedArray{T}, ::Type{Val{:central}}, fx::StridedArray{T}, ::Type{Val{:Default}})
    m, n = size(J)
    epsilon_factor = compute_epsilon_factor(Val{:central}, T)
    @inbounds for i in 1:n
        epsilon = compute_epsilon(Val{:central}, x[i], epsilon_factor)
        epsilon_double_inv = one(T) / (2 * epsilon)
        for j in 1:m
            if i==j
                J[j,i] = (f(x[j]+epsilon) - f(x[j]-epsilon)) * epsilon_double_inv
            else
                J[j,i] = zero(T)
            end
        end
    end
    J
end

function finite_difference_jacobian!{T<:Real}(J::StridedArray{T}, f, x::StridedArray{T}, ::Type{Val{:forward}}, fx::StridedArray{T}, ::Type{Val{:Default}})
    m, n = size(J)
    epsilon_factor = compute_epsilon_factor(Val{:forward}, T)
    @inbounds for i in 1:n
        epsilon = compute_epsilon(Val{:forward}, x[i], epsilon_factor)
        epsilon_inv = one(T) / epsilon
        for j in 1:m
            if i==j
                J[j,i] = (f(x[j]+epsilon) - fx[j]) * epsilon_inv
            else
                J[j,i] = zero(T)
            end
        end
    end
    J
end

# efficient implementations for OrdinaryDiffEq Jacobian wrappers, assuming the system function supplies StridedArrays
function finite_difference_jacobian!{T<:Real}(J::StridedArray{T}, f, x::StridedArray{T}, ::Type{Val{:forward}}, fx::StridedArray{T}, ::Type{Val{:JacobianWrapper}})
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

function finite_difference_jacobian!{T<:Real}(J::StridedArray{T}, f, x::StridedArray{T}, ::Type{Val{:central}}, fx::StridedArray{T}, ::Type{Val{:JacobianWrapper}})
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


# TODO: Jacobians for complex-valued callables
