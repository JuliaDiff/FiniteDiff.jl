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

@inline function compute_epsilon(::Type{Val{:complex}}, x::T, ::Union{Void,T}=nothing) where T<:Real
    eps(T)
end

@inline function compute_epsilon_factor(fdtype::DataType, ::Type{T}) where T<:Number
    if fdtype==Val{:forward}
        return sqrt(eps(T))
    elseif fdtype==Val{:central}
        return cbrt(eps(T))
    else
        return one(T)
    end
end

function fdtype_error(funtype::DataType=Val{:Real})
    if funtype == Val{:Real}
        error("Unrecognized fdtype: valid values are Val{:forward}, Val{:central} and Val{:complex}.")
    elseif funtype == Val{:Complex}
        error("Unrecognized fdtype: valid values are Val{:forward} or Val{:central}.")
    else
        error("Unrecognized funtype: valid values are Val{:Real} or Val{:Complex}.")
    end
end
