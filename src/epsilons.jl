#=
Very heavily inspired by Calculus.jl, but with an emphasis on performance and DiffEq API convenience.
=#

#=
Compute the finite difference interval epsilon.
Reference: Numerical Recipes, chapter 5.7.
=#
@inline function compute_epsilon(::Val{:forward}, x::T, relstep::Real, absstep::Real, dir::Real) where T<:Number
    return max(relstep*abs(x), absstep)*dir
end

@inline function compute_epsilon(::Val{:central}, x::T, relstep::Real, absstep::Real, dir=nothing) where T<:Number
    return max(relstep*abs(x), absstep)
end

@inline function compute_epsilon(::Val{:hcentral}, x::T, relstep::Real, absstep::Real, dir=nothing) where T<:Number
    return max(relstep*abs(x), absstep)
end

@inline function compute_epsilon(::Val{:complex}, x::T, ::Union{Nothing,T}=nothing, ::Union{Nothing,T}=nothing, dir=nothing) where T<:Real
    return eps(T)
end

default_relstep(v::Type, T) = default_relstep(v(), T)
@inline function default_relstep(::Val{fdtype}, ::Type{T}) where {fdtype,T<:Number}
    if fdtype==:forward
        return sqrt(eps(real(T)))
    elseif fdtype==:central
        return cbrt(eps(real(T)))
    elseif fdtype==:hcentral
        eps(T)^(1/4)
    else
        return one(real(T))
    end
end

function fdtype_error(::Type{T}=Float64) where T
    if T<:Real
        error("Unrecognized fdtype: valid values are Val{:forward}, Val{:central} and Val{:complex}.")
    elseif T<:Complex
        error("Unrecognized fdtype: valid values are Val{:forward} or Val{:central}.")
    else
        error("Unrecognized returntype: should be a subtype of Real or Complex.")
    end
end
