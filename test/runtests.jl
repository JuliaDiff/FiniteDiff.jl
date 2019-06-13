using DiffEqDiffTools
using Test, LinearAlgebra

@time begin
  include("finitedifftests.jl")
  include("coloring_tests.jl")
end
