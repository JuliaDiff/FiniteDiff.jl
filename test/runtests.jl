using DiffEqDiffTools
using Compat, Compat.Test

@time include("finitedifftests.jl")
@time include("old_interface_tests.jl")
