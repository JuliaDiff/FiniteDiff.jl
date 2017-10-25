x = collect(linspace(-2π, 2π, 100))
y = sin.(x)
df = zeros(100)
epsilon = zeros(100)
df_ref = cos.(x)
J_ref = diagm(cos.(x))
J = zeros(J_ref)

err_func(a,b) = maximum(abs.(a-b))

# TODO: add tests for GPUArrays, those should work now
# TODO: add tests for DEDataArrays

# StridedArray tests start here
# derivative tests for real-valued callables
@test err_func(DiffEqDiffTools.finite_difference(sin, x, Val{:forward}), df_ref) < 1e-4
@test err_func(DiffEqDiffTools.finite_difference(sin, x, Val{:central}), df_ref) < 1e-8
@test err_func(DiffEqDiffTools.finite_difference(sin, x, Val{:complex}), df_ref) < 1e-15

@test err_func(DiffEqDiffTools.finite_difference(sin, x, Val{:forward}, Val{:Real}, Val{:Default}, y), df_ref) < 1e-4
@test err_func(DiffEqDiffTools.finite_difference(sin, x, Val{:central}, Val{:Real}, Val{:Default}, y), df_ref) < 1e-8
@test err_func(DiffEqDiffTools.finite_difference(sin, x, Val{:complex}, Val{:Real}, Val{:Default}, y), df_ref) < 1e-15

@test err_func(DiffEqDiffTools.finite_difference(sin, x, Val{:forward}, Val{:Real}, Val{:Default}, y, epsilon), df_ref) < 1e-4
@test err_func(DiffEqDiffTools.finite_difference(sin, x, Val{:central}, Val{:Real}, Val{:Default}, y, epsilon), df_ref) < 1e-8
@test err_func(DiffEqDiffTools.finite_difference(sin, x, Val{:complex}, Val{:Real}, Val{:Default}, y, epsilon), df_ref) < 1e-15

@test err_func(DiffEqDiffTools.finite_difference!(df, sin, x, Val{:forward}), df_ref) < 1e-4
@test err_func(DiffEqDiffTools.finite_difference!(df, sin, x, Val{:central}), df_ref) < 1e-8
@test err_func(DiffEqDiffTools.finite_difference!(df, sin, x, Val{:complex}), df_ref) < 1e-15

@test err_func(DiffEqDiffTools.finite_difference!(df, sin, x, Val{:forward}, Val{:Real}, Val{:Default}, y), df_ref) < 1e-4
@test err_func(DiffEqDiffTools.finite_difference!(df, sin, x, Val{:central}, Val{:Real}, Val{:Default}, y), df_ref) < 1e-8
@test err_func(DiffEqDiffTools.finite_difference!(df, sin, x, Val{:complex}, Val{:Real}, Val{:Default}, y), df_ref) < 1e-15

@test err_func(DiffEqDiffTools.finite_difference!(df, sin, x, Val{:forward}, Val{:Real}, Val{:Default}, y, epsilon), df_ref) < 1e-4
@test err_func(DiffEqDiffTools.finite_difference!(df, sin, x, Val{:central}, Val{:Real}, Val{:Default}, y, epsilon), df_ref) < 1e-8
@test err_func(DiffEqDiffTools.finite_difference!(df, sin, x, Val{:complex}, Val{:Real}, Val{:Default}, y, epsilon), df_ref) < 1e-15

# Jacobian tests for real-valued callables
@test err_func(DiffEqDiffTools.finite_difference_jacobian(sin, x, Val{:forward}), J_ref) < 1e-4
@test err_func(DiffEqDiffTools.finite_difference_jacobian(sin, x, Val{:central}), J_ref) < 1e-8
@test err_func(DiffEqDiffTools.finite_difference_jacobian(sin, x, Val{:complex}), J_ref) < 1e-15

@test err_func(DiffEqDiffTools.finite_difference_jacobian(sin, x, Val{:forward}, Val{:Real}, Val{:Default}, y), J_ref) < 1e-4
@test err_func(DiffEqDiffTools.finite_difference_jacobian(sin, x, Val{:central}, Val{:Real}, Val{:Default}, y), J_ref) < 1e-8
@test err_func(DiffEqDiffTools.finite_difference_jacobian(sin, x, Val{:complex}, Val{:Real}, Val{:Default}, y), J_ref) < 1e-15

@test err_func(DiffEqDiffTools.finite_difference_jacobian(sin, x, Val{:forward}, Val{:Real}, Val{:Default}, y, epsilon), J_ref) < 1e-4
@test err_func(DiffEqDiffTools.finite_difference_jacobian(sin, x, Val{:central}, Val{:Real}, Val{:Default}, y, epsilon), J_ref) < 1e-8
@test err_func(DiffEqDiffTools.finite_difference_jacobian(sin, x, Val{:complex}, Val{:Real}, Val{:Default}, y, epsilon), J_ref) < 1e-15

@test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, sin, x, Val{:forward}), J_ref) < 1e-4
@test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, sin, x, Val{:central}), J_ref) < 1e-8
@test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, sin, x, Val{:complex}), J_ref) < 1e-15

@test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, sin, x, Val{:forward}, Val{:Real}, Val{:Default}, y), J_ref) < 1e-4
@test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, sin, x, Val{:central}, Val{:Real}, Val{:Default}, y), J_ref) < 1e-8
@test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, sin, x, Val{:complex}, Val{:Real}, Val{:Default}, y), J_ref) < 1e-15

@test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, sin, x, Val{:forward}, Val{:Real}, Val{:Default}, y, epsilon), J_ref) < 1e-4
@test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, sin, x, Val{:central}, Val{:Real}, Val{:Default}, y, epsilon), J_ref) < 1e-8
@test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, sin, x, Val{:complex}, Val{:Real}, Val{:Default}, y, epsilon), J_ref) < 1e-15

# derivative tests for complex-valued callables
x = x + im*x
f(x) = cos(real(x)) + im*sin(imag(x))
y = f.(x)
df = zeros(x)
epsilon = zeros(length(x))
df_ref = -sin.(real(x)) + im*cos.(imag(x))
J_ref = diagm(df_ref)

@test err_func(DiffEqDiffTools.finite_difference(f, x, Val{:forward}, Val{:Complex}), df_ref) < 1e-4
@test err_func(DiffEqDiffTools.finite_difference(f, x, Val{:central}, Val{:Complex}), df_ref) < 1e-8

@test err_func(DiffEqDiffTools.finite_difference(f, x, Val{:forward}, Val{:Complex}, Val{:Default}, y), df_ref) < 1e-4
@test err_func(DiffEqDiffTools.finite_difference(f, x, Val{:central}, Val{:Complex}, Val{:Default}, y), df_ref) < 1e-8

@test err_func(DiffEqDiffTools.finite_difference(f, x, Val{:forward}, Val{:Complex}, Val{:Default}, y, epsilon), df_ref) < 1e-4
@test err_func(DiffEqDiffTools.finite_difference(f, x, Val{:central}, Val{:Complex}, Val{:Default}, y, epsilon), df_ref) < 1e-8

# Jacobian tests for complex-valued callables
#@test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, Val{:forward}, Val{:Complex}, Val{:Default}), J_ref) < 1e-4
#@test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, Val{:central}, Val{:Complex}, Val{:Default}), J_ref) < 1e-8
# StridedArray tests end here
