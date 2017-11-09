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
        fdtype_error(Val{:Real})
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
    else
        fdtype_error(Val{:Complex})
    end
    df
end
# Fallbacks for complex-valued callables end here.

#=
Optimized implementations for StridedArrays.
=#
# for R -> R^n
function finite_difference!(df::StridedArray{<:Real}, f, x::Real,
    fdtype::DataType, ::Type{Val{:Real}}, ::Type{Val{:Default}},
    fx::Union{Void,StridedArray{<:Real}}=nothing, epsilon::Union{Void,StridedArray{<:Real}}=nothing, return_type::DataType=eltype(x))

    epsilon_elemtype = compute_epsilon_elemtype(epsilon, x)
    if fdtype == Val{:forward}
        epsilon = compute_epsilon(Val{:forward}, x)
        if typeof(fx) == Void
            df .= (f(x+epsilon) - f(x)) / epsilon
        else
            df .= (f(x+epsilon) - fx) / epsilon
        end
    elseif fdtype == Val{:central}
        epsilon = compute_epsilon(Val{:central}, x)
        df .= (f(x+epsilon) - f(x-epsilon)) / (2*epsilon)
    elseif fdtype == Val{:complex}
        epsilon = eps(eltype(x))
        df .= imag(f(x+im*epsilon)) / epsilon
    else
        fdtype_error(Val{:Real})
    end
    df
end

# for R^n -> R^n
function finite_difference!(df::StridedArray{<:Real}, f, x::StridedArray{<:Real},
    fdtype::DataType, ::Type{Val{:Real}}, ::Type{Val{:Default}},
    fx::Union{Void,StridedArray{<:Real}}=nothing, epsilon::Union{Void,StridedArray{<:Real}}=nothing, return_type::DataType=eltype(x))

    epsilon_elemtype = compute_epsilon_elemtype(epsilon, x)
    if fdtype == Val{:forward}
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
    elseif fdtype == Val{:central}
        epsilon_factor = compute_epsilon_factor(Val{:central}, epsilon_elemtype)
        @inbounds for i in 1 : length(x)
            epsilon = compute_epsilon(Val{:central}, x[i], epsilon_factor)
            epsilon_double_inv = one(typeof(epsilon)) / (2*epsilon)
            x_plus, x_minus = x[i]+epsilon, x[i]-epsilon
            df[i] = (f(x_plus) - f(x_minus)) * epsilon_double_inv
        end
    elseif fdtype == Val{:complex}
        epsilon_complex = eps(eltype(x))
        @inbounds for i in 1 : length(x)
            df[i] = imag(f(x[i]+im*epsilon_complex)) / epsilon_complex
        end
    else
        fdtype_error(Val{:Real})
    end
    df
end

# C -> C^n
function finite_difference!(df::StridedArray{<:Number}, f, x::Number,
    fdtype::DataType, ::Type{Val{:Complex}}, ::Type{Val{:Default}},
    fx::Union{Void,StridedArray{<:Number}}=nothing, epsilon::Union{Void,StridedArray{<:Real}}=nothing, return_type::DataType=eltype(x))

    epsilon_elemtype = compute_epsilon_elemtype(epsilon, x)
    if fdtype == Val{:forward}
        epsilon = compute_epsilon(Val{:forward}, real(x[i]))
        if typeof(fx) == Void
            df .= ( real( f(x+epsilon) - f(x) ) + im*imag( f(x+im*epsilon) - f(x) ) ) / epsilon
        else
            df .= ( real( f(x+epsilon) - fx ) + im*imag( f(x+im*epsilon) - fx )) / epsilon
        end
    elseif fdtype == Val{:central}
        epsilon = compute_epsilon(Val{:central}, real(x[i]))
        df .= (real(f(x+epsilon) - f(x-epsilon)) + im*imag(f(x+im*epsilon) - f(x-im*epsilon))) / (2 * epsilon)
    else
        fdtype_error(Val{:Complex})
    end
    df
end

# C^n -> C^n
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
    else
        fdtype_error(Val{:Complex})
    end
    df
end

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
    else
        fdtype_error(funtype)
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
        return real((f(x[i]+epsilon) - f(x[i]))) / epsilon + im*imag((f(x[i]+im*epsilon) - f(x[i]))) / epsilon
    else
        return real((f(x[i]+epsilon) - fx[i])) / epsilon + im*imag((f(x[i]+im*epsilon) - fx[i])) / epsilon
    end
end

@inline function finite_difference_kernel(f, x::Number, ::Type{Val{:central}}, ::Type{Val{:Complex}}, epsilon::Real, fx::Union{Void,<:Number}=nothing)
    real(f(x+epsilon) - f(x-epsilon)) / (2 * epsilon) + im*imag(f(x+im*epsilon) - f(x-im*epsilon)) / (2 * epsilon)
end
