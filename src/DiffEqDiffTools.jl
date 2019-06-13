__precompile__()

module DiffEqDiffTools

using LinearAlgebra, SparseArrays, StaticArrays

include("function_wrappers.jl")
include("finitediff.jl")
include("derivatives.jl")
include("gradients.jl")
include("jacobians.jl")

# Piracy
function Base.setindex(x::Array,v,i::Int)
  _x = copy(x)
  _x[i] = v
  _x
end

end # module
