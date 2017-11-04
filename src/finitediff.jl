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

include("derivatives.jl")
include("jacobians.jl")
include("diffeqwrappers.jl")
