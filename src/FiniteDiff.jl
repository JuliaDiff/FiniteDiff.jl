module FiniteDiff


using LinearAlgebra, SparseArrays, StaticArrays, ArrayInterface, Requires

import Base: resize!

_vec(x) = vec(x)
_vec(x::Number) = x

_mat(x::AbstractMatrix) = x
_mat(x::StaticVector)   = reshape(x, (axes(x, 1),     SOneTo(1)))
_mat(x::AbstractVector) = reshape(x, (axes(x, 1), Base.OneTo(1)))

include("iteration_utils.jl")
include("epsilons.jl")
include("derivatives.jl")
include("gradients.jl")
include("jacobians.jl")
include("hessians.jl")



end # module
