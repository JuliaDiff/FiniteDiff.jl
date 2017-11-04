#=
Very heavily inspired by Calculus.jl, but with an emphasis on performance and DiffEq API convenience.
=#

#=
Compute the finite difference interval epsilon.
Reference: Numerical Recipes, chapter 5.7.
=#
@inline function compute_epsilon(::Type{Val{:forward}}, x::T, eps_sqrt::T=sqrt(eps(T))) where T<:Real
    eps_sqrt * max(one(T), abs(x))
end

@inline function compute_epsilon(::Type{Val{:central}}, x::T, eps_cbrt::T=cbrt(eps(T))) where T<:Real
    eps_cbrt * max(one(T), abs(x))
end

@inline function compute_epsilon_factor(fdtype::DataType, ::Type{T}) where T<:Number
    if fdtype==Val{:forward}
        return sqrt(eps(T))
    elseif fdtype==Val{:central}
        return cbrt(eps(T))
    else
        error("Unrecognized fdtype: must be Val{:forward} or Val{:central}.")
    end
end

function compute_epsilon_elemtype(epsilon, x)
    if typeof(epsilon) != Void
        return eltype(epsilon)
    elseif eltype(x) <: Real
        return eltype(x)
    elseif eltype(x) <: Complex
        return eltype(x).parameters[1]
    else
        error("Could not compute epsilon type.")
    end
end

#=
Compute the derivative df of a callable f on a collection of points x.
Generic fallbacks for AbstractArrays that are not StridedArrays.
=#
function finite_difference(f, x::AbstractArray{<:Number},
    fdtype::DataType=Val{:central}, funtype::DataType=Val{:Real}, wrappertype::DataType=Val{:Default},
    fx::Union{Void,AbstractArray{<:Number}}=nothing, epsilon::Union{Void,AbstractArray{<:Real}}=nothing, return_type::DataType=eltype(x))

    df = zeros(return_type, size(x))
    finite_difference!(df, f, x, fdtype, funtype, wrappertype, fx, epsilon, return_type)
end

function finite_difference!(df::AbstractArray{<:Number}, f, x::AbstractArray{<:Number},
    fdtype::DataType=Val{:central}, funtype::DataType=Val{:Real}, wrappertype::DataType=Val{:Default},
    fx::Union{Void,AbstractArray{<:Number}}=nothing, epsilon::Union{Void,AbstractArray{<:Real}}=nothing, return_type::DataType=eltype(x))

    finite_difference!(df, f, x, fdtype, funtype, wrappertype, fx, return_type)
end

# Fallbacks for real-valued callables start here.
function finite_difference!(df::AbstractArray{<:Real}, f, x::AbstractArray{<:Real},
    fdtype::DataType, ::Type{Val{:Real}}, ::Type{Val{:Default}},
    fx::Union{Void,AbstractArray{<:Real}}=nothing, epsilon::Union{Void,AbstractArray{<:Real}}=nothing, return_type::DataType=eltype(x))

    epsilon_elemtype = compute_epsilon_elemtype(epsilon, x)
    if typeof(epsilon) == Void
        epsilon = zeros(epsilon_elemtype, size(x))
    end
    if fdtype == Val{:forward}
        epsilon_factor = compute_epsilon_factor(Val{:forward}, epsilon_elemtype)
        @. epsilon = compute_epsilon(Val{:forward}, x, epsilon_factor)
        if typeof(fx) == Void
            @. df = (f(x+epsilon) - f(x)) / epsilon
        else
            @. df = (f(x+epsilon) - fx) / epsilon
        end
    elseif fdtype == Val{:central}
        epsilon_factor = compute_epsilon_factor(Val{:central}, eltype(x))
        @. epsilon = compute_epsilon(Val{:central}, x, epsilon_factor)
        @. df = (f(x+epsilon) - f(x-epsilon)) / (2 * epsilon)
    elseif fdtype == Val{:complex}
        epsilon_complex = eps(epsilon_elemtype)
        @. df = imag(f(x+im*epsilon_complex)) / epsilon_complex
    else
        error("Unrecognized fdtype: valid values are Val{:forward}, Val{:central} and Val{:complex}.")
    end
    df
end

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
# Fallbacks for real-valued callables end here.

# Fallbacks for complex-valued callables start here.
function finite_difference!(df::AbstractArray{<:Number}, f, x::AbstractArray{<:Number},
    fdtype::DataType, ::Type{Val{:Complex}}, ::Type{Val{:Default}},
    fx::Union{Void,AbstractArray{<:Number}}=nothing, epsilon::Union{Void,AbstractArray{<:Real}}=nothing, return_type::DataType=eltype(x))

    if (fdtype == Val{:forward} || fdtype == Val{:central}) && typeof(epsilon) == Void
        if eltype(x) <: Real
            epsilon = zeros(eltype(x), size(x))
        else
            epsilon = zeros(eltype(real(x)), size(x))
        end
    end
    if fdtype == Val{:forward}
        @show typeof(x)
        epsilon_factor = compute_epsilon_factor(Val{:forward}, eltype(epsilon))
        @. epsilon = compute_epsilon(Val{:forward}, real(x), epsilon_factor)
        if typeof(fx) == Void
            fx = f.(x)
        end
        @. df = real((f(x+epsilon) - fx)) / epsilon + im*imag((f(x+im*epsilon) - fx)) / epsilon
    elseif fdtype == Val{:central}
        epsilon_factor = compute_epsilon_factor(Val{:central}, eltype(epsilon))
        @. epsilon = compute_epsilon(Val{:central}, real(x), epsilon_factor)
        @. df = real(f(x+epsilon) - f(x-epsilon)) / (2 * epsilon) + im*imag(f(x+im*epsilon) - f(x-epsilon)) / (2 * epsilon)
    elseif fdtype == Val{:complex}
        error("Invalid fdtype value, Val{:complex} not implemented for complex-valued functions.")
    end
    df
end

# TODO: fallbacks for DiffEq wrappers over complex-valued callables

# Fallbacks for complex-valued callables end here.


#=
Optimized implementations for StridedArrays.
=#
function finite_difference!(df::StridedArray{<:Real}, f, x::StridedArray{<:Real},
    ::Type{Val{:central}}, ::Type{Val{:Real}}, ::Type{Val{:Default}},
    fx::Union{Void,StridedArray{<:Real}}=nothing, epsilon::Union{Void,StridedArray{<:Real}}=nothing, return_type::DataType=eltype(x))

    epsilon_elemtype = compute_epsilon_elemtype(epsilon, x)
    epsilon_factor = compute_epsilon_factor(Val{:central}, epsilon_elemtype)
    @inbounds for i in 1 : length(x)
        epsilon = compute_epsilon(Val{:central}, x[i], epsilon_factor)
        epsilon_double_inv = one(typeof(epsilon)) / (2*epsilon)
        x_plus, x_minus = x[i]+epsilon, x[i]-epsilon
        df[i] = (f(x_plus) - f(x_minus)) * epsilon_double_inv
    end
    df
end

function finite_difference!(df::StridedArray{<:Real}, f, x::StridedArray{<:Real},
    ::Type{Val{:forward}}, ::Type{Val{:Real}}, ::Type{Val{:Default}},
    fx::Union{Void,StridedArray{<:Real}}=nothing, epsilon::Union{Void,StridedArray{<:Real}}=nothing, return_type::DataType=eltype(x))

    epsilon_elemtype = compute_epsilon_elemtype(epsilon, x)
    epsilon_factor = compute_epsilon_factor(Val{:forward}, epsilon_elemtype)
    @inbounds for i in 1 : length(x)
        epsilon = compute_epsilon(Val{:forward}, x[i], epsilon_factor)
        x_plus = x[i] + epsilon
        if typeof(fx) == Void
            df[i] = (f(x_plus) - f(x[i])) / epsilon
        else
            df[i] = (f(x_plus) - fx[i]) / epsilon
        end
    end
    df
end

function finite_difference!(df::StridedArray{<:Real}, f, x::StridedArray{<:Real},
    ::Type{Val{:complex}}, ::Type{Val{:Real}}, ::Type{Val{:Default}},
    fx::Union{Void,StridedArray{<:Real}}=nothing, epsilon::Union{Void,StridedArray{<:Real}}=nothing, return_type::DataType=eltype(x))

    epsilon_complex = eps(eltype(x))
    @inbounds for i in 1 : length(x)
        df[i] = imag(f(x[i]+im*epsilon_complex)) / epsilon_complex
    end
    df
end

function finite_difference!(df::StridedArray{<:Number}, f, x::StridedArray{<:Number},
    fdtype::DataType, ::Type{Val{:Complex}}, ::Type{Val{:Default}},
    fx::Union{Void,StridedArray{<:Number}}=nothing, epsilon::Union{Void,StridedArray{<:Real}}=nothing, return_type::DataType=eltype(x))

    epsilon_elemtype = compute_epsilon_elemtype(epsilon, x)
    if fdtype == Val{:forward}
        epsilon_factor = compute_epsilon_factor(Val{:forward}, epsilon_elemtype)
        @inbounds for i in 1 : length(x)
            epsilon = compute_epsilon(Val{:forward}, real(x[i]), epsilon_factor)
            if typeof(fx) == Void
                df[i] = ( real( f(x[i]+epsilon) - f(x[i]) ) + im*imag( f(x[i]+im*epsilon) - f(x[i]) ) ) / epsilon
            else
                df[i] = ( real( f(x[i]+epsilon) - fx[i] ) + im*imag( f(x[i]+im*epsilon) - fx[i] )) / epsilon
            end
        end
    elseif fdtype == Val{:central}
        epsilon_factor = compute_epsilon_factor(Val{:central}, epsilon_elemtype)
        @inbounds for i in 1 : length(x)
            epsilon = compute_epsilon(Val{:central}, real(x[i]), epsilon_factor)
            df[i] = (real(f(x[i]+epsilon) - f(x[i]-epsilon)) + im*imag(f(x[i]+im*epsilon) - f(x[i]-im*epsilon))) / (2 * epsilon)
        end
    elseif fdtype == Val{:complex}
        error("Invalid fdtype value, Val{:complex} not implemented for complex-valued functions.")
    end
    df
end
# TODO: optimized implementations for DiffEq wrappers

#=
Compute the derivative df of a callable f on a collection of points x.
Single point implementations.
=#
function finite_difference(f, x::T, fdtype::DataType, funtype::DataType=Val{:Real}, f_x::Union{Void,T}=nothing) where T<:Number
    if funtype == Val{:Real}
        if fdtype == Val{:complex}
            epsilon = eps(T)
            return imag(f(x+im*epsilon)) / epsilon
        else
            epsilon = compute_epsilon(fdtype, x)
            return finite_difference_kernel(f, x, fdtype, funtype, epsilon, f_x)
        end
    elseif funtype == Val{:Complex}
        epsilon = compute_epsilon(fdtype, real(x))
        return finite_difference_kernel(f, x, fdtype, funtype, epsilon, f_x)
    end
end

@inline function finite_difference_kernel(f, x::T, ::Type{Val{:forward}}, ::Type{Val{:Real}}, epsilon::T, fx::Union{Void,T}=nothing) where T<:Real
    if typeof(fx) == Void
        return (f(x+epsilon) - f(x)) / epsilon
    else
        return (f(x+epsilon) - fx) / epsilon
    end
end

@inline function finite_difference_kernel(f, x::T, ::Type{Val{:central}}, ::Type{Val{:Real}}, epsilon::T, ::Union{Void,T}=nothing) where T<:Real
    (f(x+epsilon) - f(x-epsilon)) / (2 * epsilon)
end

@inline function finite_difference_kernel(f, x::Number, ::Type{Val{:forward}}, ::Type{Val{:Complex}}, epsilon::Real, fx::Union{Void,<:Number}=nothing)
    if typeof(fx) == Void
        return real((f(x[i]+epsilon) - f(x[i]))) / epsilon + im*imag((f(x[i]+im*epsilon) - fx[i])) / epsilon
    else
        return real((f(x[i]+epsilon) - fx[i])) / epsilon + im*imag((f(x[i]+im*epsilon) - fx[i])) / epsilon
    end
end

@inline function finite_difference_kernel(f, x::Number, ::Type{Val{:central}}, ::Type{Val{:Complex}}, epsilon::Real, fx::Union{Void,<:Number}=nothing)
    real(f(x+epsilon) - f(x-epsilon)) / (2 * epsilon) + im*imag(f(x+im*epsilon) - f(x-im*epsilon)) / (2 * epsilon)
end

# Compute the Jacobian matrix of a real-valued callable f.
function finite_difference_jacobian(f, x::AbstractArray{<:Number},
    fdtype::DataType=Val{:central}, funtype::DataType=Val{:Real}, wrappertype::DataType=Val{:Default},
    fx::Union{Void,AbstractArray{<:Number}}=nothing, epsilon::Union{Void,AbstractArray{<:Real}}=nothing, returntype=eltype(x))

    J = zeros(returntype, length(x), length(x))
    finite_difference_jacobian!(J, f, x, fdtype, funtype, wrappertype, fx, epsilon, returntype)
end

function finite_difference_jacobian!(J::AbstractMatrix{<:Number}, f, x::AbstractArray{<:Number},
    fdtype::DataType=Val{:central}, funtype::DataType=Val{:Real}, wrappertype::DataType=Val{:Default},
    fx::Union{Void,AbstractArray{<:Number}}=nothing, epsilon::Union{Void,AbstractArray{<:Number}}=nothing, returntype=eltype(x))

    finite_difference_jacobian!(J, f, x, fdtype, funtype, wrappertype, fx, epsilon, returntype)
end

function finite_difference_jacobian!(J::AbstractMatrix{<:Real}, f, x::AbstractArray{<:Real},
    fdtype::DataType, ::Type{Val{:Real}}, wrappertype::DataType=Val{:Default},
    fx::Union{Void,AbstractArray{<:Real}}=nothing, epsilon::Union{Void,AbstractArray{<:Real}}=nothing, returntype=eltype(x))

    # TODO: test and rework this
    m, n = size(J)
    epsilon_elemtype = compute_epsilon_elemtype(epsilon, x)
    if fdtype == Val{:forward}
        if typeof(fx) == Void
            if wrappertype==Val{:Default}
                fx = f.(x)
            elseif wrappertype==Val{:DiffEqJacobianWrapper}
                fx = f(x)
            else
                error("Unrecognized wrappertype: must be Val{:Default} or Val{:DiffEqJacobianWrapper}.")
            end
        end
        epsilon_factor = compute_epsilon_factor(Val{:forward}, epsilon_elemtype)
        shifted_x = copy(x)
        @inbounds for i in 1:n
            epsilon = compute_epsilon(t, x[i], epsilon_factor)
            shifted_x[i] += epsilon
            J[:, i] .= (f(shifted_x) - f_x) / epsilon
            shifted_x[i] = x[i]
        end
    elseif fdtype == Val{:central}
        epsilon_factor = compute_epsilon_factor(Val{:central}, epsilon_elemtype)
        shifted_x_plus  = copy(x)
        shifted_x_minus = copy(x)
        @inbounds for i in 1:n
            epsilon = compute_epsilon(fdtype, x[i], epsilon_factor)
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

function finite_difference_jacobian!(J::StridedMatrix{<:Real}, f, x::StridedArray{<:Real},
    fdtype::DataType, ::Type{Val{:Real}}, ::Type{Val{:Default}},
    fx::Union{Void,StridedArray{<:Real}}=nothing, epsilon::Union{Void,StridedArray{<:Real}}=nothing, returntype=eltype(x))

    m, n = size(J)
    epsilon_elemtype = compute_epsilon_elemtype(epsilon, x)
    if fdtype == Val{:forward}
        epsilon_factor = compute_epsilon_factor(Val{:forward}, eltype(x))
        @inbounds for i in 1:n
            epsilon = compute_epsilon(Val{:forward}, x[i], epsilon_factor)
            epsilon_inv = one(returntype) / epsilon
            for j in 1:m
                if i==j
                    if typeof(fx) == Void
                        J[j,i] = (f(x[j]+epsilon) - f(x[j])) * epsilon_inv
                    else
                        if typeof(fx) == Void
                            J[j,i] = (f(x[j]+epsilon) - f(x[j])) * epsilon_inv
                        else
                            J[j,i] = (f(x[j]+epsilon) - fx[j]) * epsilon_inv
                        end
                    end
                else
                    J[j,i] = zero(returntype)
                end
            end
        end
    elseif fdtype == Val{:central}
        epsilon_factor = compute_epsilon_factor(Val{:central}, eltype(x))
        @inbounds for i in 1:n
            epsilon = compute_epsilon(Val{:central}, x[i], epsilon_factor)
            epsilon_double_inv = one(returntype) / (2 * epsilon)
            for j in 1:m
                if i==j
                    J[j,i] = (f(x[j]+epsilon) - f(x[j]-epsilon)) * epsilon_double_inv
                else
                    J[j,i] = zero(returntype)
                end
            end
        end
    elseif fdtype == Val{:complex}
        epsilon = eps(epsilon_elemtype)
        epsilon_inv = one(epsilon_elemtype) / epsilon
        @inbounds for i in 1:n
            for j in 1:m
                if i==j
                    J[j,i] = imag(f(x[j]+im*epsilon)) * epsilon_inv
                else
                    J[j,i] = zero(returntype)
                end
            end
        end
    end
    J
end

# efficient implementations for OrdinaryDiffEq Jacobian wrappers

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

# TODO: Jacobians for complex-valued callables
function finite_difference_jacobian!(J::StridedMatrix{<:Number}, f, x::StridedArray{<:Number},
    fdtype::DataType, ::Type{Val{:Complex}}, ::Type{Val{:Default}},
    fx::Union{Void,StridedArray{<:Number}}=nothing, epsilon::Union{Void,StridedArray{<:Real}}=nothing, returntype=eltype(x))

    # TODO: finish this
    m, n = size(J)
    epsilon_elemtype = compute_epsilon_elemtype(epsilon, x)
    if fdtype == Val{:forward}
        epsilon_factor = compute_epsilon_factor(Val{:forward}, epsilon_elemtype)
        @inbounds for i in 1:n
            epsilon = compute_epsilon(Val{:forward}, real(x[i]), epsilon_factor)
            epsilon_inv = one(returntype) / epsilon
            for j in 1:m
                if i==j
                    if typeof(fx) == Void
                        J[j,i] = ( real( f(x[j]+epsilon) - f(x[j]) ) + im*imag( f(x[j]+im*epsilon) - f(x[j]) ) ) * epsilon_inv
                    else
                        J[j,i] = ( real( f(x[j]+epsilon) - fx[j] ) + im*imag( f(x[j]+im*epsilon) - fx[j] ) ) * epsilon_inv
                    end
                else
                    J[j,i] = zero(returntype)
                end
            end
        end
    elseif fdtype == Val{:central}
        epsilon_factor = compute_epsilon_factor(Val{:central}, epsilon_elemtype)
        @inbounds for i in 1:n
            epsilon = compute_epsilon(Val{:central}, real(x[i]), epsilon_factor)
            epsilon_double_inv = one(returntype) / (2 * epsilon)
            for j in 1:m
                if i==j
                    J[j,i] = ( real( f(x[j]+epsilon)-f(x[j]-epsilon) ) + im*imag( f(x[j]+im*epsilon) - f(x[j]-im*epsilon) ) ) * epsilon_double_inv
                else
                    J[j,i] = zero(returntype)
                end
            end
        end
    else
        error("Unrecognized fdtype: must be Val{:forward} or Val{:central}.")
    end
    J
end
