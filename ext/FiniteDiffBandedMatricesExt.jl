module FiniteDiffBandedMatricesExt

if isdefined(Base, :get_extension)
    using FiniteDiff: FiniteDiff, ArrayInterface
    using BandedMatrices: BandedMatrices
else
    using ..FiniteDiff: FiniteDiff, ArrayInterface
    using ..BandedMatrices: BlockBandedMatrices
end

FiniteDiff._use_findstructralnz(::BandedMatrices.BandedMatrix) = false

@inline function FiniteDiff._colorediteration!(Jac::BandedMatrices.BandedMatrix,
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

end #module
