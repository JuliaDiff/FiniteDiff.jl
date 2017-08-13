x = collect(linspace(-2π, 2π, 100))
y = sin.(x)
df = zeros(100)
df_ref = cos.(x)

# TODO: add tests for non-StridedArrays and with more complicated functions

# derivative tests
@test maximum(abs.(DiffEqDiffTools.finite_difference(sin, x, Val{:forward})    - df_ref)) < 1e-4
@test maximum(abs.(DiffEqDiffTools.finite_difference(sin, x, Val{:forward}, y) - df_ref)) < 1e-4
@test maximum(abs.(DiffEqDiffTools.finite_difference(sin, x, Val{:central})    - df_ref)) < 1e-8
@test maximum(abs.(DiffEqDiffTools.finite_difference!(df, sin, x, Val{:forward}, nothing, Val{:Default}) - df_ref)) < 1e-4
@test maximum(abs.(DiffEqDiffTools.finite_difference!(df, sin, x, Val{:forward}, y,       Val{:Default}) - df_ref)) < 1e-4
@test maximum(abs.(DiffEqDiffTools.finite_difference!(df, sin, x, Val{:central}, nothing, Val{:Default}) - df_ref)) < 1e-8

# Jacobian tests
using Calculus
@test DiffEqDiffTools.finite_difference_jacobian(sin, x, Val{:forward}) ≈ Calculus.finite_difference_jacobian(sin, x, :forward)
@test DiffEqDiffTools.finite_difference_jacobian(sin, x, Val{:central}) ≈ Calculus.finite_difference_jacobian(sin, x, :central)
