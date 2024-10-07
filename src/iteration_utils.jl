@inline function _colorediteration!(J,sparsity,rows_index,cols_index,vfx,colorvec,color_i,ncols)
    for i in 1:length(cols_index)
        if colorvec[cols_index[i]] == color_i
            J[rows_index[i],cols_index[i]] = vfx[rows_index[i]]
        end
    end
end

#override default setting of using findstructralnz
_use_findstructralnz(sparsity) = ArrayInterface.has_sparsestruct(sparsity)

# test if J, sparsity are both SparseMatrixCSC and have the same sparsity pattern of stored values
_use_sparseCSC_common_sparsity(J, sparsity) = false
