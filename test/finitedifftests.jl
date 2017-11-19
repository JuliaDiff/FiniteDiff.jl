x = collect(linspace(-2π, 2π, 100))
y = sin.(x)
df = zeros(100)
epsilon = zeros(100)
df_ref = cos.(x)
J_ref = diagm(cos.(x))
J = zeros(J_ref)

err_func(a,b) = maximum(abs.(a-b))

# TODO: add tests for GPUArrays
# TODO: add tests for DEDataArrays

# StridedArray tests start here
# derivative tests for real-valued callables
@time @testset "Derivative StridedArray real-valued tests" begin
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
end

# derivative tests for complex-valued callables
x = x + im*x
f(x) = cos(real(x)) + im*sin(imag(x))
y = f.(x)
df = zeros(x)
epsilon = zeros(length(x))
df_ref = -sin.(real(x)) + im*cos.(imag(x))

@time @testset "Derivative StridedArray complex-valued tests" begin
    @test err_func(DiffEqDiffTools.finite_difference(f, x, Val{:forward}, Val{:Complex}), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference(f, x, Val{:central}, Val{:Complex}), df_ref) < 1e-8

    @test err_func(DiffEqDiffTools.finite_difference(f, x, Val{:forward}, Val{:Complex}, Val{:Default}, y), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference(f, x, Val{:central}, Val{:Complex}, Val{:Default}, y), df_ref) < 1e-8

    @test err_func(DiffEqDiffTools.finite_difference(f, x, Val{:forward}, Val{:Complex}, Val{:Default}, y, epsilon), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference(f, x, Val{:central}, Val{:Complex}, Val{:Default}, y, epsilon), df_ref) < 1e-8

    @test err_func(DiffEqDiffTools.finite_difference!(df, f, x, Val{:forward}, Val{:Complex}), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference!(df, f, x, Val{:central}, Val{:Complex}), df_ref) < 1e-8

    @test err_func(DiffEqDiffTools.finite_difference!(df, f, x, Val{:forward}, Val{:Complex}, Val{:Default}, y), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference!(df, f, x, Val{:central}, Val{:Complex}, Val{:Default}, y), df_ref) < 1e-8

    @test err_func(DiffEqDiffTools.finite_difference!(df, f, x, Val{:forward}, Val{:Complex}, Val{:Default}, y, epsilon), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference!(df, f, x, Val{:central}, Val{:Complex}, Val{:Default}, y, epsilon), df_ref) < 1e-8
end

function f(x)
    fvec = zeros(x)
    fvec[1] = (x[1]+3)*(x[2]^3-7)+18
    fvec[2] = sin(x[2]*exp(x[1])-1)
    fvec
end
x = rand(2)
y = f(x)
J_ref = [[-7+x[2]^3 3*(3+x[1])*x[2]^2]; [exp(x[1])*x[2]*cos(1-exp(x[1])*x[2]) exp(x[1])*cos(1-exp(x[1])*x[2])]]
J = zeros(J_ref)
df = zeros(x)
df_ref = diag(J_ref)
epsilon = zeros(x)

# Jacobian tests for real-valued callables
@time @testset "Jacobian StridedArray real-valued tests" begin
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, Val{:forward}), J_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, Val{:central}), J_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, Val{:complex}), J_ref) < 1e-14

    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, Val{:forward}, Val{:Real}, Val{:Default}, y), J_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, Val{:central}, Val{:Real}, Val{:Default}, y), J_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, Val{:complex}, Val{:Real}, Val{:Default}, y), J_ref) < 1e-14

    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, Val{:forward}, Val{:Real}, Val{:Default}, y, epsilon), J_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, Val{:central}, Val{:Real}, Val{:Default}, y, epsilon), J_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, Val{:complex}, Val{:Real}, Val{:Default}, y, epsilon), J_ref) < 1e-14

    @test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, f, x, Val{:forward}), J_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, f, x, Val{:central}), J_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, f, x, Val{:complex}), J_ref) < 1e-14

    @test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, f, x, Val{:forward}, Val{:Real}, Val{:Default}, y), J_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, f, x, Val{:central}, Val{:Real}, Val{:Default}, y), J_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, f, x, Val{:complex}, Val{:Real}, Val{:Default}, y), J_ref) < 1e-14

    @test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, f, x, Val{:forward}, Val{:Real}, Val{:Default}, y, epsilon), J_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, f, x, Val{:central}, Val{:Real}, Val{:Default}, y, epsilon), J_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, f, x, Val{:complex}, Val{:Real}, Val{:Default}, y, epsilon), J_ref) < 1e-14
end

function f(x)
    fvec = zeros(x)
    fvec[1] = (im*x[1]+3)*(x[2]^3-7)+18
    fvec[2] = sin(x[2]*exp(x[1])-1)
    fvec
end
x = rand(2) + im*rand(2)
y = f(x)
J_ref = [[im*(-7+x[2]^3) 3*(3+im*x[1])*x[2]^2]; [exp(x[1])*x[2]*cos(1-exp(x[1])*x[2]) exp(x[1])*cos(1-exp(x[1])*x[2])]]
J = zeros(J_ref)
df = zeros(x)
df_ref = diag(J_ref)
epsilon = zeros(real.(x))

# Jacobian tests for complex-valued callables
@time @testset "Jacobian StridedArray complex-valued tests" begin
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, Val{:forward}, Val{:Complex}, Val{:Default}), J_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, Val{:central}, Val{:Complex}, Val{:Default}), J_ref) < 1e-8

    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, Val{:forward}, Val{:Complex}, Val{:Default}, y), J_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, Val{:central}, Val{:Complex}, Val{:Default}, y), J_ref) < 1e-8

    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, Val{:forward}, Val{:Complex}, Val{:Default}, y, epsilon), J_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, Val{:central}, Val{:Complex}, Val{:Default}, y, epsilon), J_ref) < 1e-8

    @test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, f, x, Val{:forward}, Val{:Complex}, Val{:Default}), J_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, f, x, Val{:central}, Val{:Complex}, Val{:Default}), J_ref) < 1e-8

    @test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, f, x, Val{:forward}, Val{:Complex}, Val{:Default}, y), J_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, f, x, Val{:central}, Val{:Complex}, Val{:Default}, y), J_ref) < 1e-8

    @test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, f, x, Val{:forward}, Val{:Complex}, Val{:Default}, y, epsilon), J_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_jacobian!(J, f, x, Val{:central}, Val{:Complex}, Val{:Default}, y, epsilon), J_ref) < 1e-8
end

# StridedArray tests end here
