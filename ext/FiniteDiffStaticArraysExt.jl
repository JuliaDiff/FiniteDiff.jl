module FiniteDiffStaticArraysExt

if isdefined(Base, :get_extension)
    using FiniteDiff: FiniteDiff, ArrayInterface
    using StaticArrays
else
    using ..FiniteDiff: FiniteDiff, ArrayInterface
    using ..StaticArrays
end
FiniteDiff._mat(x::StaticVector)   = reshape(x, (axes(x, 1),     SOneTo(1)))
FiniteDiff.setindex(x::StaticArray, v, i::Int...) = StaticArrays.setindex(x, v, i...)
FiniteDiff.__Symmetric(x::SMatrix) = Symmetric(SArray(H))

end #module
