__precompile__()

module DiffEqDiffTools

using LinearAlgebra, SparseArrays, StaticArrays

import Base: resize!

include("function_wrappers.jl")
include("finitediff.jl")
include("derivatives.jl")
include("gradients.jl")
include("jacobians.jl")
include("hessians.jl")

# Piracy
function Base.setindex(x::Array,v,i::Int)
  _x = copy(x)
  _x[i] = v
  _x
end

end # module
