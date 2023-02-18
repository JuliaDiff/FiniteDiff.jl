module FiniteDiff

using LinearAlgebra, SparseArrays, StaticArrays, ArrayInterface, Requires

import Base: resize!

_vec(x) = vec(x)
_vec(x::Number) = x

_mat(x::AbstractMatrix) = x
_mat(x::StaticVector)   = reshape(x, (axes(x, 1),     SOneTo(1)))
_mat(x::AbstractVector) = reshape(x, (axes(x, 1), Base.OneTo(1)))

# Setindex overloads without piracy
setindex(x...) = Base.setindex(x...)
setindex(x::StaticArray, v, i::Int...) = StaticArrays.setindex(x, v, i...)

function setindex(x::AbstractArray, v, i...)
    _x = Base.copymutable(x)
    _x[i...] = v
    return _x
end

function setindex(x::AbstractVector, v, i::Int)
    n = length(x)
    x .* (i .!== 1:n) .+ v .* (i .== 1:n)
end

function setindex(x::AbstractMatrix, v, i::Int, j::Int)
    n, m = Base.size(x)
    x .* (i .!== 1:n) .* (j .!== i:m)' .+ v .* (i .== 1:n) .* (j .== i:m)'
end

include("iteration_utils.jl")
include("epsilons.jl")
include("derivatives.jl")
include("gradients.jl")
include("jacobians.jl")
include("hessians.jl")



end # module
