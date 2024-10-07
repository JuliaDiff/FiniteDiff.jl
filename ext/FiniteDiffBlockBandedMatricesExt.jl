module FiniteDiffBlockBandedMatricesExt

if isdefined(Base, :get_extension)
    using FiniteDiff: FiniteDiff, ArrayInterface
    using BlockBandedMatrices: BlockBandedMatrices
    using BlockBandedMatrices.BlockArrays
else
    using ..FiniteDiff: FiniteDiff, ArrayInterface
    using ..BlockBandedMatrices: BlockBandedMatrices
    using ..BlockBandedMatrices.BlockArrays
end

FiniteDiff._use_findstructralnz(::BlockBandedMatrices.BandedBlockBandedMatrix) = false
FiniteDiff._use_findstructralnz(::BlockBandedMatrices.BlockBandedMatrix) = false

@inline function FiniteDiff._colorediteration!(Jac::BlockBandedMatrices.BandedBlockBandedMatrix,
                                    sparsity::BlockBandedMatrices.BandedBlockBandedMatrix,
                                    rows_index,cols_index,vfx,colorvec,color_i,ncols)
    λ,μ = BlockBandedMatrices.subblockbandwidths(Jac)
    rs = BlockArrays.blocklengths(BlockArrays.axes(Jac,1)) # column block sizes
    cs = BlockArrays.blocklengths(BlockArrays.axes(Jac,1))
    b = BlockBandedMatrices.BlockArray(vfx,rs)
    c = BlockBandedMatrices.BlockArray(colorvec,cs)
    for J=BlockArrays.blockaxes(Jac,2)
        c_v = c.blocks[J.n[1]]
        for K=BlockBandedMatrices.blockcolrange(Jac,J)
            V = view(Jac,K,J)
            b_v = b.blocks[K.n[1]]
            data = BlockBandedMatrices.bandeddata(V)
            p = pointer(data)
            st = stride(data,2)
            m,n = size(V)
            for j=1:n
                if c_v[j] == color_i
                    for k=max(1,j-μ):min(m,j+λ)
                        unsafe_store!(p, b_v[k], (j-1)*st + μ + k - j + 1)
                    end
                end
            end
        end
    end
end

@inline function FiniteDiff._colorediteration!(Jac::BlockBandedMatrices.BlockBandedMatrix,
                                    sparsity::BlockBandedMatrices.BlockBandedMatrix,
                                    rows_index,cols_index,vfx,colorvec,color_i,ncols)
    rs = BlockArrays.blocklengths(BlockArrays.axes(Jac,1)) # column block sizes
    cs = BlockArrays.blocklengths(BlockArrays.axes(Jac,1))
    b = BlockBandedMatrices.BlockArray(vfx,rs)
    c = BlockBandedMatrices.BlockArray(colorvec,cs)
    for J=BlockArrays.blockaxes(Jac,2)
        c_v = c.blocks[J.n[1]]
        blockcolrange = BlockBandedMatrices.blockcolrange(Jac,J)
        _,n = length.(getindex.(axes(Jac), (blockcolrange[1], J)))
        for j = 1:n
            if c_v[j] == color_i
                for K = blockcolrange
                    V = view(Jac,K,J)
                    b_v = b.blocks[K.n[1]]
                    m = size(V,1)
                    for k = 1:m
                        V[k,j] = b_v[k]
                    end
                end
            end
        end
    end
end


end #module
