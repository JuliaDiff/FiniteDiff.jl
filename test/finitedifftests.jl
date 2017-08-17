x = collect(linspace(-2π, 2π, 100))
y = sin.(x)
df = zeros(100)
df_ref = cos.(x)
J_ref = diagm(cos.(x))

err_func(a,b) = maximum(abs.(a-b))

# TODO: add tests for non-StridedArrays and with more complicated functions

# derivative tests
@test err_func(DiffEqDiffTools.finite_difference(sin, x, Val{:forward}), df_ref)    < 1e-4
@test err_func(DiffEqDiffTools.finite_difference(sin, x, Val{:forward}, y), df_ref) < 1e-4
@test err_func(DiffEqDiffTools.finite_difference(sin, x, Val{:central}), df_ref)    < 1e-8
@test err_func(DiffEqDiffTools.finite_difference(sin, x, Val{:complex}), df_ref)    < 1e-15
@test err_func(DiffEqDiffTools.finite_difference!(df, sin, x, Val{:forward}, nothing, Val{:Default}), df_ref) < 1e-4
@test err_func(DiffEqDiffTools.finite_difference!(df, sin, x, Val{:forward}, y,       Val{:Default}), df_ref) < 1e-4
@test err_func(DiffEqDiffTools.finite_difference!(df, sin, x, Val{:central}, nothing, Val{:Default}), df_ref) < 1e-8
@test err_func(DiffEqDiffTools.finite_difference!(df, sin, x, Val{:complex}, nothing, Val{:Default}), df_ref) < 1e-15

# Jacobian tests
@test err_func(DiffEqDiffTools.finite_difference_jacobian(sin, x, Val{:forward}), J_ref) < 1e-4
@test err_func(DiffEqDiffTools.finite_difference_jacobian(sin, x, Val{:central}), J_ref) < 1e-8
@test err_func(DiffEqDiffTools.finite_difference_jacobian(sin, x, Val{:complex}), J_ref) < 1e-15
