#=
Very heavily inspired by Calculus.jl, but with an emphasis on performance and DiffEq API convenience.
=#

"""
    compute_epsilon(::Val{:forward}, x::T, relstep::Real, absstep::Real, dir::Real) where T<:Number

Compute the finite difference step size (epsilon) for forward finite differences.

The step size is computed as `max(relstep*abs(x), absstep)*dir`, which ensures 
numerical stability by using a relative step scaled by the magnitude of `x` 
when `x` is large, and an absolute step when `x` is small.

# Arguments
- `::Val{:forward}`: Finite difference type indicator for forward differences
- `x::T`: Point at which to compute the step size
- `relstep::Real`: Relative step size factor
- `absstep::Real`: Absolute step size fallback
- `dir::Real`: Direction multiplier (typically ±1)

# Returns
- Step size `ϵ` for forward finite difference: `f(x + ϵ)`

Reference: Numerical Recipes, chapter 5.7.
"""
@inline function compute_epsilon(
        ::Val{:forward}, x::T, relstep::Real, absstep::Real, dir::Real) where {T <: Number}
    return max(relstep*abs(x), absstep)*dir
end

"""
    compute_epsilon(::Val{:central}, x::T, relstep::Real, absstep::Real, dir=nothing) where T<:Number

Compute the finite difference step size (epsilon) for central finite differences.

The step size is computed as `max(relstep*abs(x), absstep)`, which ensures 
numerical stability by using a relative step scaled by the magnitude of `x` 
when `x` is large, and an absolute step when `x` is small.

# Arguments
- `::Val{:central}`: Finite difference type indicator for central differences
- `x::T`: Point at which to compute the step size
- `relstep::Real`: Relative step size factor
- `absstep::Real`: Absolute step size fallback
- `dir`: Direction parameter (unused for central differences)

# Returns
- Step size `ϵ` for central finite difference: `(f(x + ϵ) - f(x - ϵ)) / (2ϵ)`
"""
@inline function compute_epsilon(::Val{:central}, x::T, relstep::Real,
        absstep::Real, dir = nothing) where {T <: Number}
    return max(relstep*abs(x), absstep)
end

"""
    compute_epsilon(::Val{:hcentral}, x::T, relstep::Real, absstep::Real, dir=nothing) where T<:Number

Compute the finite difference step size (epsilon) for central finite differences in Hessian computations.

The step size is computed as `max(relstep*abs(x), absstep)`, which ensures 
numerical stability by using a relative step scaled by the magnitude of `x` 
when `x` is large, and an absolute step when `x` is small.

# Arguments
- `::Val{:hcentral}`: Finite difference type indicator for Hessian central differences
- `x::T`: Point at which to compute the step size
- `relstep::Real`: Relative step size factor
- `absstep::Real`: Absolute step size fallback
- `dir`: Direction parameter (unused for central differences)

# Returns
- Step size `ϵ` for Hessian central finite differences
"""
@inline function compute_epsilon(::Val{:hcentral}, x::T, relstep::Real,
        absstep::Real, dir = nothing) where {T <: Number}
    return max(relstep*abs(x), absstep)
end

"""
    compute_epsilon(::Val{:complex}, x::T, ::Union{Nothing,T}=nothing, ::Union{Nothing,T}=nothing, dir=nothing) where T<:Real

Compute the finite difference step size (epsilon) for complex step differentiation.

For complex step differentiation, the step size is simply the machine epsilon `eps(T)`,
which provides optimal accuracy since complex step differentiation doesn't suffer from
subtractive cancellation errors.

# Arguments
- `::Val{:complex}`: Finite difference type indicator for complex step differentiation
- `x::T`: Point at which to compute the step size (unused, type determines epsilon)
- Additional arguments are unused for complex step differentiation

# Returns
- Machine epsilon `eps(T)` for complex step differentiation: `imag(f(x + iϵ)) / ϵ`

# Notes
Complex step differentiation computes derivatives as `imag(f(x + iϵ)) / ϵ` where `ϵ = eps(T)`.
This method provides machine precision accuracy without subtractive cancellation.
"""
@inline function compute_epsilon(::Val{:complex}, x::T, ::Union{Nothing, T} = nothing,
        ::Union{Nothing, T} = nothing, dir = nothing) where {T <: Real}
    return eps(T)
end

"""
    default_relstep(fdtype, ::Type{T}) where T<:Number

Compute the default relative step size for finite difference approximations.

Returns optimal default step sizes based on the finite difference method and 
numerical type, balancing truncation error and round-off error.

# Arguments
- `fdtype`: Finite difference type (`Val(:forward)`, `Val(:central)`, `Val(:hcentral)`, etc.)
- `::Type{T}`: Numerical type for which to compute the step size

# Returns
- `sqrt(eps(real(T)))` for forward differences
- `cbrt(eps(real(T)))` for central differences  
- `eps(T)^(1/4)` for Hessian central differences
- `one(real(T))` for other types

# Notes
These step sizes minimize the total error (truncation + round-off) for each method:
- Forward differences have O(h) truncation error, optimal h ~ sqrt(eps)
- Central differences have O(h²) truncation error, optimal h ~ eps^(1/3)
- Hessian methods have O(h²) truncation error but involve more operations
"""
default_relstep(::Type{V}, T) where {V} = default_relstep(V(), T)
@inline function default_relstep(::Val{fdtype}, ::Type{T}) where {fdtype, T <: Number}
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

"""
    fdtype_error(::Type{T}=Float64) where T

Throw an informative error for unsupported finite difference type combinations.

# Arguments
- `::Type{T}`: Return type of the function being differentiated

# Errors
- For `Real` return types: suggests `Val{:forward}`, `Val{:central}`, `Val{:complex}`
- For `Complex` return types: suggests `Val{:forward}`, `Val{:central}` (no complex step)
- For other types: suggests the return type should be Real or Complex subtype
"""
function fdtype_error(::Type{T} = Float64) where {T}
    if T<:Real
        error("Unrecognized fdtype: valid values are Val{:forward}, Val{:central} and Val{:complex}.")
    elseif T<:Complex
        error("Unrecognized fdtype: valid values are Val{:forward} or Val{:central}.")
    else
        error("Unrecognized returntype: should be a subtype of Real or Complex.")
    end
end
