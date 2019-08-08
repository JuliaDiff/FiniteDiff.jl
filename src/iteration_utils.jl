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

#override default setting of using findstructralnz
_use_findstructralnz(sparsity) = ArrayInterface.has_sparsestruct(sparsity)
_use_findstructralnz(::SparseMatrixCSC) = false

function __init__()
    @require BlockBandedMatrices="ffab5731-97b5-5995-9138-79e8c1846df0" begin
        _use_findstructralnz(::BlockBandedMatrices.BandedBlockBandedMatrix) = false
        _use_findstructralnz(::BlockBandedMatrices.BlockBandedMatrix) = false

        @inline function _colorediteration!(Jac::BlockBandedMatrices.BandedBlockBandedMatrix,
                                            sparsity::BlockBandedMatrices.BandedBlockBandedMatrix,
                                            rows_index,cols_index,vfx,colorvec,color_i,ncols)
            λ,μ = BlockBandedMatrices.subblockbandwidths(Jac)
            rs = BlockBandedMatrices.BlockSizes((BlockBandedMatrices.cumulsizes(Jac,1),)) # column block sizes
            cs = BlockBandedMatrices.BlockSizes((BlockBandedMatrices.cumulsizes(Jac,2),))
            b = BlockBandedMatrices.BlockArray(vfx,rs)
            c = BlockBandedMatrices.BlockArray(colorvec,cs)
            @inbounds for J=BlockBandedMatrices.Block.(1:BlockBandedMatrices.nblocks(Jac,2))
                c_v = c.blocks[J.n[1]]
                @inbounds for K=BlockBandedMatrices.blockcolrange(Jac,J)
                    V = view(Jac,K,J)
                    b_v = b.blocks[K.n[1]]
                    data = BlockBandedMatrices.bandeddata(V)
                    p = pointer(data)
                    st = stride(data,2)
                    m,n = size(V)
                    @inbounds for j=1:n
                        if c_v[j] == color_i
                            @inbounds for k=max(1,j-μ):min(m,j+λ)
                                unsafe_store!(p, b_v[k], (j-1)*st + μ + k - j + 1)
                            end
                        end
                    end
                end
            end
        end

        @inline function _colorediteration!(Jac::BlockBandedMatrices.BlockBandedMatrix,
                                            sparsity::BlockBandedMatrices.BlockBandedMatrix,
                                            rows_index,cols_index,vfx,colorvec,color_i,ncols)
            rs = BlockBandedMatrices.BlockSizes((BlockBandedMatrices.cumulsizes(Jac,1),)) # column block sizes
            cs = BlockBandedMatrices.BlockSizes((BlockBandedMatrices.cumulsizes(Jac,2),))
            b = BlockBandedMatrices.BlockArray(vfx,rs)
            c = BlockBandedMatrices.BlockArray(colorvec,cs)
            @inbounds for J=BlockBandedMatrices.Block.(1:BlockBandedMatrices.nblocks(Jac,2))
                c_v = c.blocks[J.n[1]]
                blockcolrange = BlockBandedMatrices.blockcolrange(Jac,J)
                _,n = BlockBandedMatrices.blocksize(Jac,(blockcolrange[1].n[1],J.n[1]))
                @inbounds for j = 1:n
                    if c_v[j] == color_i
                        @inbounds for K = blockcolrange
                            V = view(Jac,K,J)
                            b_v = b.blocks[K.n[1]]
                            m = size(V,1)
                            @inbounds for k = 1:m
                                V[k,j] = b_v[k]
                            end
                        end
                    end
                end
            end
        end

    end

    @require BandedMatrices = "aae01518-5342-5314-be14-df237901396f" begin

        _use_findstructralnz(::BandedMatrices.BandedMatrix) = false

        @inline function _colorediteration!(Jac::BandedMatrices.BandedMatrix,
                                            sparsity::BandedMatrices.BandedMatrix,
                                            rows_index,cols_index,vfx,colorvec,color_i,ncols)
            nrows = size(Jac,1)
            l,u = BandedMatrices.bandwidths(Jac)
            #data = BandedMatrices.bandeddata(Jac)
            @inbounds for col_index in max(1,1-l):min(ncols,ncols+u)
                if colorvec[col_index] == color_i
                    @inbounds for row_index in max(1,col_index-u):min(nrows,col_index+l)
                        #data[u+row_index-col_index+1,col_index] = vfx[row_index]
                        Jac[row_index,col_index]=vfx[row_index]
                    end
                end
            end
        end
    end

end
