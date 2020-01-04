module FiniteDiff


using LinearAlgebra, SparseArrays, StaticArrays, ArrayInterface, Requires

import Base: resize!

include("iteration_utils.jl")
include("epsilons.jl")
include("derivatives.jl")
include("gradients.jl")
include("jacobians.jl")
include("hessians.jl")

end # module
