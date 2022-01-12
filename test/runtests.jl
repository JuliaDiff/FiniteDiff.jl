using Pkg
using SafeTestsets
const LONGER_TESTS = false

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV,"APPVEYOR")

function activate_downstream_env()
    Pkg.activate("downstream")
    Pkg.develop(PackageSpec(path=dirname(@__DIR__)))
    Pkg.instantiate()
end

@time begin

if GROUP == "All" || GROUP == "Core"
  @time @safetestset "FiniteDiff Standard Tests" begin include("finitedifftests.jl") end
  @time @safetestset "Color Differentiation Tests" begin include("coloring_tests.jl") end
  @time @safetestset "Out of Place Tests" begin include("out_of_place_tests.jl") end
end

if GROUP == "All" || GROUP == "Downstream"
  activate_downstream_env()
  @time @safetestset "ODEs" begin
    import OrdinaryDiffEq
    @time @safetestset "OrdinaryDiffEq Tridiagonal" begin include("downstream/ordinarydiffeq_tridiagonal_solve.jl") end
    include(joinpath(dirname(pathof(OrdinaryDiffEq)), "..", "test/interface/sparsediff_tests.jl"))
  end
end

end
