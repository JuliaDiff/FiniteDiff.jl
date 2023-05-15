@inline function _colorediteration!(J,sparsity,rows_index,cols_index,vfx,colorvec,color_i,ncols)
    @inbounds for i in 1:length(cols_index)
        if colorvec[cols_index[i]] == color_i
            J[rows_index[i],cols_index[i]] = vfx[rows_index[i]]
        end
    end
end

@inline function _colorediteration!(J,sparsity::SparseMatrixCSC,rows_index,cols_index,vfx,colorvec,color_i,ncols)
    @inbounds for col_index in 1:ncols
        if colorvec[col_index] == color_i
            @inbounds for row_index in view(sparsity.rowval,sparsity.colptr[col_index]:sparsity.colptr[col_index+1]-1)
                J[row_index,col_index]=vfx[row_index]
            end
        end
    end
end

# fast version for the case where J and sparsity have the same sparsity pattern
@inline function _colorediteration!(Jsparsity::SparseMatrixCSC,vfx,colorvec,color_i,ncols)
    @inbounds for col_index in 1:ncols
        if colorvec[col_index] == color_i
            @inbounds for spidx in nzrange(Jsparsity, col_index)
                row_index = Jsparsity.rowval[spidx]
                Jsparsity.nzval[spidx]=vfx[row_index]
            end
        end
    end
end

#override default setting of using findstructralnz
_use_findstructralnz(sparsity) = ArrayInterface.has_sparsestruct(sparsity)
_use_findstructralnz(::SparseMatrixCSC) = false

# test if J, sparsity are both SparseMatrixCSC and have the same sparsity pattern of stored values
_use_sparseCSC_common_sparsity(J, sparsity) = false
_use_sparseCSC_common_sparsity(J::SparseMatrixCSC, sparsity::SparseMatrixCSC) =
    ((J.colptr == sparsity.colptr) && (J.rowval == sparsity.rowval))

