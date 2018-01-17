
# TODO: add tests for GPUArrays
# TODO: add tests for DEDataArrays


# Derivative tests
x = collect(linspace(-2π, 2π, 100))
y = sin.(x)
df = zeros(100)
epsilon = zeros(100)
df_ref = cos.(x)
forward_cache = DiffEqDiffTools.DerivativeCache(x, y, epsilon, Val{:forward})
central_cache = DiffEqDiffTools.DerivativeCache(x, nothing, epsilon, Val{:central})
complex_cache = DiffEqDiffTools.DerivativeCache(x, nothing, nothing, Val{:complex})

err_func(a,b) = maximum(abs.(a-b))

@time @testset "Derivative single point real-valued tests" begin
    @test err_func(DiffEqDiffTools.finite_difference_derivative(sin, π/4, Val{:forward}), √2/2) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_derivative(sin, π/4, Val{:central}), √2/2) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_derivative(sin, π/4, Val{:complex}), √2/2) < 1e-15
end

@time @testset "Derivative StridedArray real-valued tests" begin
    @test err_func(DiffEqDiffTools.finite_difference_derivative(sin, x, Val{:forward}), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_derivative(sin, x, Val{:central}), df_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_derivative(sin, x, Val{:complex}), df_ref) < 1e-15

    @test err_func(DiffEqDiffTools.finite_difference_derivative(sin, x, Val{:forward}, Val{:Real}, y), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_derivative(sin, x, Val{:central}, Val{:Real}, y), df_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_derivative(sin, x, Val{:complex}, Val{:Real}, y), df_ref) < 1e-15

    @test err_func(DiffEqDiffTools.finite_difference_derivative(sin, x, Val{:forward}, Val{:Real}, y, epsilon), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_derivative(sin, x, Val{:central}, Val{:Real}, y, epsilon), df_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_derivative(sin, x, Val{:complex}, Val{:Real}, y, epsilon), df_ref) < 1e-15

    @test err_func(DiffEqDiffTools.finite_difference_derivative!(df, sin, x, Val{:forward}), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_derivative!(df, sin, x, Val{:central}), df_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_derivative!(df, sin, x, Val{:complex}), df_ref) < 1e-15

    @test err_func(DiffEqDiffTools.finite_difference_derivative!(df, sin, x, Val{:forward}, Val{:Real}, y), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_derivative!(df, sin, x, Val{:central}, Val{:Real}, y), df_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_derivative!(df, sin, x, Val{:complex}, Val{:Real}, y), df_ref) < 1e-15

    @test err_func(DiffEqDiffTools.finite_difference_derivative!(df, sin, x, Val{:forward}, Val{:Real}, y, epsilon), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_derivative!(df, sin, x, Val{:central}, Val{:Real}, y, epsilon), df_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_derivative!(df, sin, x, Val{:complex}, Val{:Real}, y, epsilon), df_ref) < 1e-15

    @test err_func(DiffEqDiffTools.finite_difference_derivative!(df, sin, x, forward_cache), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_derivative!(df, sin, x, central_cache), df_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_derivative!(df, sin, x, complex_cache), df_ref) < 1e-15
end

x = x + im*x
#f(x) = cos(real(x)) + im*sin(imag(x))
f(x) = sin(x) + cos(x)
y = f.(x)
df = zeros(x)
epsilon = zeros(length(x))
#df_ref = -sin.(real(x)) + im*cos.(imag(x))
df_ref = cos.(x) - sin.(x)
forward_cache = DiffEqDiffTools.DerivativeCache(x, y, epsilon, Val{:forward})
central_cache = DiffEqDiffTools.DerivativeCache(x, nothing, epsilon, Val{:central})

@time @testset "Derivative single point complex-valued tests" begin
    @test err_func(DiffEqDiffTools.finite_difference_derivative(f, π/4+im*π/4, Val{:forward}, Val{:Complex}), cos(π/4+im*π/4)-sin(π/4+im*π/4)) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_derivative(f, π/4+im*π/4, Val{:central}, Val{:Complex}), cos(π/4+im*π/4)-sin(π/4+im*π/4)) < 1e-8
end

@time @testset "Derivative StridedArray complex-valued tests" begin
    @test err_func(DiffEqDiffTools.finite_difference_derivative(f, x, Val{:forward}, Val{:Complex}), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_derivative(f, x, Val{:central}, Val{:Complex}), df_ref) < 1e-8

    @test err_func(DiffEqDiffTools.finite_difference_derivative(f, x, Val{:forward}, Val{:Complex}, y), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_derivative(f, x, Val{:central}, Val{:Complex}, y), df_ref) < 1e-8

    @test err_func(DiffEqDiffTools.finite_difference_derivative(f, x, Val{:forward}, Val{:Complex}, y, epsilon), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_derivative(f, x, Val{:central}, Val{:Complex}, y, epsilon), df_ref) < 1e-8

    @test err_func(DiffEqDiffTools.finite_difference_derivative!(df, f, x, Val{:forward}, Val{:Complex}), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_derivative!(df, f, x, Val{:central}, Val{:Complex}), df_ref) < 1e-8

    @test err_func(DiffEqDiffTools.finite_difference_derivative!(df, f, x, Val{:forward}, Val{:Complex}, y), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_derivative!(df, f, x, Val{:central}, Val{:Complex}, y), df_ref) < 1e-8

    @test err_func(DiffEqDiffTools.finite_difference_derivative!(df, f, x, Val{:forward}, Val{:Complex}, y, epsilon), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_derivative!(df, f, x, Val{:central}, Val{:Complex}, y, epsilon), df_ref) < 1e-8

    @test err_func(DiffEqDiffTools.finite_difference_derivative!(df, f, x, forward_cache), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_derivative!(df, f, x, central_cache), df_ref) < 1e-8
end

# Gradient tests
f(x) = 2x[1] + x[2]^2
x = rand(2)
fx = f(x)
df = zeros(2)
df_ref = [2., 2*x[2]]
forward_cache = DiffEqDiffTools.GradientCache(df,x,fx,nothing,nothing,Val{:forward},Val{:Real})
central_cache = DiffEqDiffTools.GradientCache(df,x,fx,nothing,nothing,Val{:central},Val{:Real})
complex_cache = DiffEqDiffTools.GradientCache(df,x,fx,nothing,nothing,Val{:complex},Val{:Real})

@time @testset "Gradient of f:R^n->R real-valued tests" begin
    @test err_func(DiffEqDiffTools.finite_difference_gradient(f, x, Val{:forward}), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_gradient(f, x, Val{:central}), df_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_gradient(f, x, Val{:complex}), df_ref) < 1e-15

    @test err_func(DiffEqDiffTools.finite_difference_gradient!(df, f, x, Val{:forward}), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_gradient!(df, f, x, Val{:central}), df_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_gradient!(df, f, x, Val{:complex}), df_ref) < 1e-15

    @test err_func(DiffEqDiffTools.finite_difference_gradient!(df, f, x, forward_cache), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_gradient!(df, f, x, central_cache), df_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_gradient!(df, f, x, complex_cache), df_ref) < 1e-15
end

f(x) = 2x[1] + im*2x[1] + x[2]^2
x = x + im*x
fx = f(x)
df = zeros(x)
df_ref = [2.0+2.0*im, 2.0*x[2]]
DiffEqDiffTools.finite_difference_gradient(f, x, Val{:forward})
DiffEqDiffTools.finite_difference_gradient(f, x, Val{:central})
forward_cache = DiffEqDiffTools.GradientCache(df,x,fx,nothing,nothing,Val{:forward},Val{:Real})
central_cache = DiffEqDiffTools.GradientCache(df,x,fx,nothing,nothing,Val{:central},Val{:Real})
complex_cache = DiffEqDiffTools.GradientCache(df,x,fx,nothing,nothing,Val{:complex},Val{:Real})

@time @testset "Gradient of f:R^n->R complex-valued tests" begin
    @test err_func(DiffEqDiffTools.finite_difference_gradient(f, x, Val{:forward}), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_gradient(f, x, Val{:central}), df_ref) < 1e-8

    @test err_func(DiffEqDiffTools.finite_difference_gradient!(df, f, x, Val{:forward}), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_gradient!(df, f, x, Val{:central}), df_ref) < 1e-8

    @test err_func(DiffEqDiffTools.finite_difference_gradient!(df, f, x, forward_cache), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_gradient!(df, f, x, central_cache), df_ref) < 1e-8
end

f(x) = [sin(x), cos(x)]
x = 2π * rand()
fx = f(x)
df = zeros(2)
df_ref = [cos(x), -sin(x)]
forward_cache = DiffEqDiffTools.GradientCache(df,x,fx,nothing,nothing,Val{:forward},Val{:Real})
central_cache = DiffEqDiffTools.GradientCache(df,x,fx,nothing,nothing,Val{:central},Val{:Real})
complex_cache = DiffEqDiffTools.GradientCache(df,x,fx,nothing,nothing,Val{:complex},Val{:Real})

@time @testset "Gradient of f:R->R^n real-valued tests" begin
    @test err_func(DiffEqDiffTools.finite_difference_gradient(f, x, Val{:forward}), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_gradient(f, x, Val{:central}), df_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_gradient(f, x, Val{:complex}), df_ref) < 1e-15

    @test err_func(DiffEqDiffTools.finite_difference_gradient!(df, f, x, Val{:forward}), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_gradient!(df, f, x, Val{:central}), df_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_gradient!(df, f, x, Val{:complex}), df_ref) < 1e-15

    @test err_func(DiffEqDiffTools.finite_difference_gradient!(df, f, x, forward_cache), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_gradient!(df, f, x, central_cache), df_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_gradient!(df, f, x, complex_cache), df_ref) < 1e-15
end

f(x) = [sin(x), cos(x)]
@show x = (2π * rand()) * (1 + im)
@show fx = f(x)
df = zeros(fx)
@show df_ref = [cos(x), -sin(x)]
forward_cache = DiffEqDiffTools.GradientCache(df,x,fx,nothing,nothing,Val{:forward},Val{:Real})
central_cache = DiffEqDiffTools.GradientCache(df,x,fx,nothing,nothing,Val{:central},Val{:Real})
complex_cache = DiffEqDiffTools.GradientCache(df,x,fx,nothing,nothing,Val{:complex},Val{:Real})
@show DiffEqDiffTools.finite_difference_gradient(f, x, Val{:forward})
@show DiffEqDiffTools.finite_difference_gradient(f, x, Val{:central})

@time @testset "Gradient of f:R->R^n complex-valued tests" begin
    @test err_func(DiffEqDiffTools.finite_difference_gradient(f, x, Val{:forward}), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_gradient(f, x, Val{:central}), df_ref) < 1e-7

    @test err_func(DiffEqDiffTools.finite_difference_gradient!(df, f, x, Val{:forward}), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_gradient!(df, f, x, Val{:central}), df_ref) < 1e-7

    @test err_func(DiffEqDiffTools.finite_difference_gradient!(df, f, x, forward_cache), df_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_gradient!(df, f, x, central_cache), df_ref) < 1e-7
end

# Jacobian tests
function f(fvec,x)
    fvec[1] = (x[1]+3)*(x[2]^3-7)+18
    fvec[2] = sin(x[2]*exp(x[1])-1)
end
x = rand(2); y = rand(2)
f(y,x)
J_ref = [[-7+x[2]^3 3*(3+x[1])*x[2]^2]; [exp(x[1])*x[2]*cos(1-exp(x[1])*x[2]) exp(x[1])*cos(1-exp(x[1])*x[2])]]
J = zeros(J_ref)
df = zeros(x)
df_ref = diag(J_ref)
epsilon = zeros(x)
forward_cache = DiffEqDiffTools.JacobianCache(similar(x),similar(x),similar(x),Val{:forward})
central_cache = DiffEqDiffTools.JacobianCache(similar(x),similar(x),similar(x))
complex_cache = DiffEqDiffTools.JacobianCache(
                Complex{eltype(x)}.(similar(x)),Complex{eltype(x)}.(similar(x)),
                Complex{eltype(x)}.(similar(x)),Val{:complex})

@time @testset "Jacobian StridedArray real-valued tests" begin
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, forward_cache), J_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, central_cache), J_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x), J_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, complex_cache), J_ref) < 1e-14
end

function f(fvec,x)
    fvec[1] = (im*x[1]+3)*(x[2]^3-7)+18
    fvec[2] = sin(x[2]*exp(x[1])-1)
end
x = rand(2) + im*rand(2)
y = similar(x)
f(y,x)
J_ref = [[im*(-7+x[2]^3) 3*(3+im*x[1])*x[2]^2]; [exp(x[1])*x[2]*cos(1-exp(x[1])*x[2]) exp(x[1])*cos(1-exp(x[1])*x[2])]]
J = zeros(J_ref)
df = zeros(x)
df_ref = diag(J_ref)
epsilon = zeros(real.(x))
forward_cache = DiffEqDiffTools.JacobianCache(similar(x),similar(x),similar(x),Val{:forward})
central_cache = DiffEqDiffTools.JacobianCache(similar(x),similar(x),similar(x))

@time @testset "Jacobian StridedArray complex-valued tests" begin
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, forward_cache), J_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, central_cache), J_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x), J_ref) < 1e-8
end

# These seem redundant?
#=
# Tests for C -> C and C -> C^n
x = 2.0 + im*3.0
f(x) = cos(real(x)) + im*log(imag(x))
y = f(x)
df_ref = -sin(real(x)) + im*1/imag(x)

@time @testset "Derivative Number complex-valued tests" begin
    # Finite difference kernel.
    @test err_func(DiffEqDiffTools.finite_difference_derivative(f, x, Val{:forward}, Val{:Complex}, y), df_ref) < 1e-7
    @test err_func(DiffEqDiffTools.finite_difference_derivative(f, x, Val{:central}, Val{:Complex}, y), df_ref) < 1e-10
    # C -> C^n
    @test err_func(DiffEqDiffTools.finite_difference_derivative(f, x, Val{:forward}, Val{:Complex}, [y]), df_ref) < 1e-7
    @test err_func(DiffEqDiffTools.finite_difference_derivative(f, x, Val{:central}, Val{:Complex}, [y]), df_ref) < 1e-10
end
=#
