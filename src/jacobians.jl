# Compute the Jacobian matrix of a real-valued callable f.
function finite_difference_jacobian(f, x::AbstractArray{<:Number},
    fdtype::DataType=Val{:central}, funtype::DataType=Val{:Real}, wrappertype::DataType=Val{:Default},
    fx::Union{Void,AbstractArray{<:Number}}=nothing, epsilon::Union{Void,AbstractArray{<:Real}}=nothing, returntype=eltype(x),
    df::Union{Void,AbstractArray{<:Number}}=nothing)

    J = zeros(returntype, length(x), length(x))
    finite_difference_jacobian!(J, f, x, fdtype, funtype, wrappertype, fx, epsilon, returntype, df)
end

function finite_difference_jacobian!(J::AbstractMatrix{<:Number}, df::AbstractVector, f, x::AbstractArray{<:Number},
    fdtype::DataType=Val{:central}, funtype::DataType=Val{:Real}, wrappertype::DataType=Val{:Default},
    fx::Union{Void,AbstractArray{<:Number}}=nothing, epsilon::Union{Void,AbstractArray{<:Number}}=nothing, returntype=eltype(x))

    finite_difference_jacobian!(J, f, x, fdtype, funtype, wrappertype, fx, epsilon, returntype, df)
end

function finite_difference_jacobian!(J::AbstractMatrix{<:Number}, f, x::AbstractArray{<:Number},
    fdtype::DataType=Val{:central}, funtype::DataType=Val{:Real}, wrappertype::DataType=Val{:Default},
    fx::Union{Void,AbstractArray{<:Number}}=nothing, epsilon::Union{Void,AbstractArray{<:Number}}=nothing, returntype=eltype(x),
    df::Union{Void,AbstractArray{<:Number}}=nothing)

    finite_difference_jacobian!(J, f, x, fdtype, funtype, wrappertype, fx, epsilon, returntype, df)
end

function finite_difference_jacobian!(J::AbstractMatrix{<:Real}, f, x::AbstractArray{<:Real},
    fdtype::DataType, ::Type{Val{:Real}}, ::Type{Val{:Default}},
    fx::Union{Void,AbstractArray{<:Real}}=nothing, epsilon::Union{Void,AbstractArray{<:Real}}=nothing, returntype=eltype(x),
    df::Union{Void,AbstractArray{<:Real}}=nothing)

    # TODO: test and rework this to support GPUArrays and non-indexable types, if possible
    m, n = size(J)
    epsilon_elemtype = compute_epsilon_elemtype(epsilon, x)
    if fdtype == Val{:forward}
        if typeof(fx) == Void
            fx = f(x)
        end
        epsilon_factor = compute_epsilon_factor(Val{:forward}, epsilon_elemtype)
        shifted_x = copy(x)
        @inbounds for i in 1:n
            epsilon = compute_epsilon(Val{:forward}, x[i], epsilon_factor)
            shifted_x[i] += epsilon
            J[:, i] .= (f(shifted_x) - fx) / epsilon
            shifted_x[i] = x[i]
        end
    elseif fdtype == Val{:central}
        epsilon_factor = compute_epsilon_factor(Val{:central}, epsilon_elemtype)
        shifted_x_plus  = copy(x)
        shifted_x_minus = copy(x)
        @inbounds for i in 1:n
            epsilon = compute_epsilon(Val{:central}, x[i], epsilon_factor)
            shifted_x_plus[i]  += epsilon
            shifted_x_minus[i] -= epsilon
            J[:, i] .= (f(shifted_x_plus) - f(shifted_x_minus)) / (epsilon + epsilon)
            shifted_x_plus[i]  = x[i]
            shifted_x_minus[i] = x[i]
        end
    elseif fdtype == Val{:complex}
        x0 = Vector{Complex{eltype(x)}}(x)
        epsilon = eps(eltype(x))
        @inbounds for i in 1:n
            x0[i] += im * epsilon
            J[:,i] .= imag.(f(x0)) / epsilon
            x0[i] -= im * epsilon
        end
    else
        fdtype_error(Val{:Real})
    end
    if typeof(df) != Void
        df .= diag(J)
    end
    J
end

function finite_difference_jacobian!(J::AbstractMatrix{<:Number}, f, x::AbstractArray{<:Number},
    fdtype::DataType, ::Type{Val{:Complex}}, ::Type{Val{:Default}},
    fx::Union{Void,AbstractArray{<:Number}}=nothing, epsilon::Union{Void,AbstractArray{<:Real}}=nothing, returntype=eltype(x),
    df::Union{Void,AbstractArray{<:Number}}=nothing)

    # TODO: test and rework this to support GPUArrays and non-indexable types, if possible
    m, n = size(J)
    epsilon_elemtype = compute_epsilon_elemtype(epsilon, x)
    if fdtype == Val{:forward}
        if typeof(fx) == Void
            fx = f(x)
        end
        epsilon_factor = compute_epsilon_factor(Val{:forward}, epsilon_elemtype)
        shifted_x = copy(x)
        @inbounds for i in 1:n
            epsilon = compute_epsilon(Val{:forward}, real(x[i]), epsilon_factor)
            shifted_x[i] += epsilon
            J[:, i] .= ( real.( f(shifted_x) - fx ) + im*imag.( f(shifted_x) - fx ) ) / epsilon
            shifted_x[i] = x[i]
        end
    elseif fdtype == Val{:central}
        epsilon_factor = compute_epsilon_factor(Val{:central}, epsilon_elemtype)
        shifted_x_plus  = copy(x)
        shifted_x_minus = copy(x)
        @inbounds for i in 1:n
            epsilon = compute_epsilon(Val{:central}, real(x[i]), epsilon_factor)
            shifted_x_plus[i]  += epsilon
            shifted_x_minus[i] -= epsilon
            J[:, i] .= ( real(f(shifted_x_plus) - f(shifted_x_minus)) + im*imag(f(shifted_x_plus) - f(shifted_x_minus)) ) / (2 * epsilon)
            shifted_x_plus[i]  = x[i]
            shifted_x_minus[i] = x[i]
        end
    else
        fdtype_error(Val{:Complex})
    end
    if typeof(df) != Void
        df .= diag(J)
    end
    J
end

#=
function finite_difference_jacobian!(J::StridedMatrix{<:Real}, f, x::StridedArray{<:Real},
    fdtype::DataType, ::Type{Val{:Real}}, ::Type{Val{:Default}},
    fx::Union{Void,StridedArray{<:Real}}=nothing, epsilon::Union{Void,StridedArray{<:Real}}=nothing, returntype=eltype(x),
    df::Union{Void,StridedArray{<:Real}}=nothing)

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
                    if typeof(df) != Void
                        df[i] = J[j,i]
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
                    if typeof(df) != Void
                        df[i] = J[j,i]
                    end
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
                    if typeof(df) != Void
                        df[i] = J[j,i]
                    end
                else
                    J[j,i] = zero(returntype)
                end
            end
        end
    end
    J
end

function finite_difference_jacobian!(J::StridedMatrix{<:Number}, f, x::StridedArray{<:Number},
    fdtype::DataType, ::Type{Val{:Complex}}, ::Type{Val{:Default}},
    fx::Union{Void,StridedArray{<:Number}}=nothing, epsilon::Union{Void,StridedArray{<:Real}}=nothing, returntype=eltype(x),
    df::Union{Void,StridedArray{<:Number}}=nothing)

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
                    if typeof(df) != Void
                        df[i] = J[j,i]
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
                    if typeof(df) != Void
                        df[i] = J[j,i]
                    end
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
=#
