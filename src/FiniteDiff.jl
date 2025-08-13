"""
    FiniteDiff

Fast non-allocating calculations of gradients, Jacobians, and Hessians with sparsity support.
"""
module FiniteDiff

using LinearAlgebra, ArrayInterface

import Base: resize!

"""
    _vec(x)

Internal utility function to vectorize arrays while preserving scalars.

# Arguments
- `x`: Array or scalar

# Returns
- `vec(x)` for arrays, `x` unchanged for scalars
"""
_vec(x) = vec(x)
_vec(x::Number) = x

"""
    _mat(x)

Internal utility function to ensure matrix format.

Converts vectors to column matrices while preserving existing matrices.
Used internally to ensure consistent matrix dimensions for operations.

# Arguments  
- `x`: Matrix or vector

# Returns
- `x` unchanged if already a matrix
- Reshaped column matrix if `x` is a vector
"""
_mat(x::AbstractMatrix) = x
_mat(x::AbstractVector) = reshape(x, (axes(x, 1), Base.OneTo(1)))

# Setindex overloads without piracy
setindex(x...) = Base.setindex(x...)

"""
    setindex(x::AbstractArray, v, i...)

Non-mutating setindex operation that returns a copy with modified elements.

Creates a mutable copy of array `x`, sets the specified indices to value `v`,
and returns the modified copy. This avoids type piracy while providing 
setindex functionality for immutable arrays.

# Arguments
- `x::AbstractArray`: Array to modify (not mutated)
- `v`: Value to set at the specified indices
- `i...`: Indices where the value should be set

# Returns
- Modified copy of `x` with `x[i...] = v`

# Examples
```julia
x = [1, 2, 3]
y = setindex(x, 99, 2)  # y = [1, 99, 3], x unchanged
```
"""
function setindex(x::AbstractArray, v, i...)
    _x = Base.copymutable(x)
    _x[i...] = v
    return _x
end

"""
    setindex(x::AbstractVector, v, i::Int)

Broadcasted setindex operation for vectors using boolean masking.

Sets the i-th element of vector `x` to value `v` using broadcasting operations.
This implementation uses boolean masks to avoid explicit copying and provide
efficient vectorized operations.

# Arguments
- `x::AbstractVector`: Input vector  
- `v`: Value to set at index `i`
- `i::Int`: Index to modify

# Returns
- Vector with `x[i] = v`, computed via broadcasting

# Examples
```julia
x = [1.0, 2.0, 3.0]
y = setindex(x, 99.0, 2)  # [1.0, 99.0, 3.0]
```
"""
function setindex(x::AbstractVector, v, i::Int)
    n = length(x)
    x .* (i .!== 1:n) .+ v .* (i .== 1:n)
end

"""
    setindex(x::AbstractMatrix, v, i::Int, j::Int)

Broadcasted setindex operation for matrices using boolean masking.

Sets the (i,j)-th element of matrix `x` to value `v` using broadcasting operations.
This implementation uses boolean masks to avoid explicit copying and provide
efficient vectorized operations.

# Arguments
- `x::AbstractMatrix`: Input matrix
- `v`: Value to set at position (i,j)
- `i::Int`: Row index to modify
- `j::Int`: Column index to modify

# Returns
- Matrix with `x[i,j] = v`, computed via broadcasting

# Examples
```julia
x = [1.0 2.0; 3.0 4.0]
y = setindex(x, 99.0, 1, 2)  # [1.0 99.0; 3.0 4.0]
```

# Notes
The implementation uses transposed broadcasting `(j .!== i:m)'` which appears
to be a typo - should likely be `(j .!== 1:m)'` for correct column masking.
"""
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
include("jvp.jl")

end # module
