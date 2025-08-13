"""
    _colorediteration!(J, sparsity, rows_index, cols_index, vfx, colorvec, color_i, ncols)

Internal function for sparse Jacobian assembly using graph coloring.

Updates the Jacobian matrix `J` with finite difference values for columns 
belonging to the current color group. This is part of the graph-coloring 
algorithm that computes multiple Jacobian columns simultaneously.

# Arguments
- `J`: Jacobian matrix to update (modified in-place)
- `sparsity`: Sparsity pattern of the Jacobian
- `rows_index`: Row indices of non-zero elements
- `cols_index`: Column indices of non-zero elements
- `vfx`: Vector of computed finite difference values
- `colorvec`: Vector assigning colors to columns
- `color_i`: Current color being processed
- `ncols`: Total number of columns in the Jacobian

# Notes
This function implements the core loop of the graph-coloring sparse Jacobian
algorithm. It assigns finite difference values to the appropriate entries
in the Jacobian matrix for all columns sharing the same color.
"""
@inline function _colorediteration!(J,sparsity,rows_index,cols_index,vfx,colorvec,color_i,ncols)
    for i in 1:length(cols_index)
        if colorvec[cols_index[i]] == color_i
            J[rows_index[i],cols_index[i]] = vfx[rows_index[i]]
        end
    end
end

"""
    _use_findstructralnz(sparsity)

Internal function to determine whether to use `findstructralnz` for sparsity detection.

Returns `true` if the sparsity pattern has structural information that can be 
utilized by `findstructralnz`, `false` otherwise. This overrides the default
behavior and delegates to `ArrayInterface.has_sparsestruct`.

# Arguments
- `sparsity`: Sparsity pattern object

# Returns
- `true` if structural sparsity information is available, `false` otherwise
"""
_use_findstructralnz(sparsity) = ArrayInterface.has_sparsestruct(sparsity)

"""
    _use_sparseCSC_common_sparsity(J, sparsity)

Internal function to test for common sparsity patterns between Jacobian and sparsity matrix.

Tests if both `J` and `sparsity` are `SparseMatrixCSC` matrices with the same 
sparsity pattern of stored values. Currently always returns `false` as a 
conservative default.

# Arguments
- `J`: Jacobian matrix
- `sparsity`: Sparsity pattern matrix

# Returns
- `false` (conservative default - may be optimized in the future)

# Notes
This function could be optimized to return `true` when both matrices have
identical sparsity patterns, allowing for more efficient algorithms.
"""
_use_sparseCSC_common_sparsity(J, sparsity) = false
