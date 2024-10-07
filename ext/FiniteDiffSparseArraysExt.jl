module FiniteDiffSparseArraysExt

using SparseArrays
using FiniteDiff

# jacobians.jl
function FiniteDiff._make_Ji(::SparseMatrixCSC, rows_index, cols_index, dx, colorvec, color_i, nrows, ncols)
    pick_inds = [i for i in 1:length(rows_index) if colorvec[cols_index[i]] == color_i]
    rows_index_c = rows_index[pick_inds]
    cols_index_c = cols_index[pick_inds]
    Ji = sparse(rows_index_c, cols_index_c, dx[rows_index_c], nrows, ncols)
    Ji
end

function FiniteDiff._make_Ji(::SparseMatrixCSC, xtype, dx, color_i, nrows, ncols)
    Ji = sparse(1:nrows, fill(color_i, nrows), dx, nrows, ncols)
    Ji
end

@inline function FiniteDiff._colorediteration!(J, sparsity::SparseMatrixCSC, rows_index, cols_index, vfx, colorvec, color_i, ncols)
    for col_index in 1:ncols
        if colorvec[col_index] == color_i
            for row_index in view(sparsity.rowval, sparsity.colptr[col_index]:sparsity.colptr[col_index+1]-1)
                J[row_index, col_index] = vfx[row_index]
            end
        end
    end
end

@inline FiniteDiff.fill_matrix!(J::AbstractSparseMatrix, v) = fill!(nonzeros(J), v)

@inline function FiniteDiff.fast_jacobian_setindex!(J::AbstractSparseMatrix, rows_index, cols_index, _color, color_i, vfx)
    @. FiniteDiff.void_setindex!((J.nzval,), getindex((J.nzval,), rows_index) + (getindex((_color,), cols_index) == color_i) * getindex((vfx,), rows_index), rows_index)
end

# iteration_utils.jl
## fast version for the case where J and sparsity have the same sparsity pattern
@inline function FiniteDiff._colorediteration!(Jsparsity::SparseMatrixCSC, vfx, colorvec, color_i, ncols)
    for col_index in 1:ncols
        if colorvec[col_index] == color_i
            for spidx in nzrange(Jsparsity, col_index)
                row_index = Jsparsity.rowval[spidx]
                Jsparsity.nzval[spidx] = vfx[row_index]
            end
        end
    end
end

FiniteDiff._use_findstructralnz(::SparseMatrixCSC) = false

FiniteDiff._use_sparseCSC_common_sparsity(J::SparseMatrixCSC, sparsity::SparseMatrixCSC) =
    ((J.colptr == sparsity.colptr) && (J.rowval == sparsity.rowval))


end
