#=
Very heavily inspired by Calculus.jl, but with an emphasis on performance and DiffEq API convenience.
=#

#=
Compute the finite difference interval epsilon.
Reference: Numerical Recipes, chapter 5.7.
=#
@inline function compute_epsilon(::Type{Val{:forward}}, x::T, relstep::Real, absstep::Real) where T<:Number
    return max(relstep*abs(x), absstep)
end

@inline function compute_epsilon(::Type{Val{:central}}, x::T, relstep::Real, absstep::Real) where T<:Number
    return max(relstep*abs(x), absstep)
end

@inline function compute_epsilon(::Type{Val{:hcentral}}, x::T, relstep::Real, absstep::Real) where T<:Number
    return max(relstep*abs(x), absstep)
end

@inline function compute_epsilon(::Type{Val{:complex}}, x::T, ::Union{Nothing,T}=nothing, ::Union{Nothing,T}=nothing) where T<:Real
    return eps(T)
end

@inline function default_relstep(fdtype::DataType, ::Type{T}) where T<:Number
    if fdtype==Val{:forward}
        return sqrt(eps(real(T)))
    elseif fdtype==Val{:central}
        return cbrt(eps(real(T)))
    elseif fdtype==Val{:hcentral}
        eps(T)^(1/4)
    else
        return one(real(T))
    end
end

function fdtype_error(funtype::Type{T}=Float64) where T
    if funtype<:Real
        error("Unrecognized fdtype: valid values are Val{:forward}, Val{:central} and Val{:complex}.")
    elseif funtype<:Complex
        error("Unrecognized fdtype: valid values are Val{:forward} or Val{:central}.")
    else
        error("Unrecognized returntype: should be a subtype of Real or Complex.")
    end
end
