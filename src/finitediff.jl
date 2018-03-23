#=
Very heavily inspired by Calculus.jl, but with an emphasis on performance and DiffEq API convenience.
=#

#=
Compute the finite difference interval epsilon.
Reference: Numerical Recipes, chapter 5.7.
=#
@inline function compute_epsilon(::Val{:forward}, x::T, eps_sqrt=sqrt(eps(real(T)))) where T<:Number
    eps_sqrt * max(one(real(T)), abs(x))
end

@inline function compute_epsilon(::Val{:central}, x::T, eps_cbrt=cbrt(eps(real(T)))) where T<:Number
    eps_cbrt * max(one(real(T)), abs(x))
end

@inline function compute_epsilon(::Val{:complex}, x::T, ::Union{Nothing,T}=nothing) where T<:Real
    eps(T)
end

@inline function compute_epsilon_factor(::Val{fdtype}, ::Type{T}) where {fdtype, T<:Number}
    if fdtype==:forward
        return sqrt(eps(real(T)))
    elseif fdtype==:central
        return cbrt(eps(real(T)))
    else
        return one(real(T))
    end
end

function fdtype_error(funtype::Type{T}=Float64) where T
    if funtype<:Real
        error("Unrecognized fdtype: valid values are Val(:forward), Val(:central) and Val(:complex).")
    elseif funtype<:Complex
        error("Unrecognized fdtype: valid values are Val(:forward) or Val(:central).")
    else
        error("Unrecognized returntype: should be a subtype of Real or Complex.")
    end
end
