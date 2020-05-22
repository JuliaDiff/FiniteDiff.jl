using FiniteDiff, Test, LinearAlgebra

# code quality guards
@test isempty(detect_unbound_args(FiniteDiff))
@test isempty(detect_ambiguities(FiniteDiff) )

# TODO: add tests for GPUArrays
# TODO: add tests for DEDataArrays

# Derivative tests
x = collect(range(-2π, stop=2π, length=100))
y = sin.(x)
df = fill(0., 100)
epsilon = fill(0., 100)
df_ref = cos.(x)
forward_cache = FiniteDiff.DerivativeCache(x, y, epsilon, Val{:forward})
central_cache = FiniteDiff.DerivativeCache(x, nothing, epsilon, Val{:central})
complex_cache = FiniteDiff.DerivativeCache(x, nothing, nothing, Val{:complex})

err_func(a,b) = maximum(abs.(a-b))

@time @testset "Derivative single point f : R -> R tests" begin
    @test err_func(FiniteDiff.finite_difference_derivative(sin, π/4, Val{:forward}), √2/2) < 1e-4
    @test err_func(FiniteDiff.finite_difference_derivative(x->x > π/4 ? error() : sin(x), π/4, Val{:forward}, dir=-1), √2/2) < 1e-4
    @test_throws Any err_func(FiniteDiff.finite_difference_derivative(x->x >= π/4 ? error() : sin(x), π/4, Val{:forward}), √2/2) < 1e-4
    @test err_func(FiniteDiff.finite_difference_derivative(sin, π/4, Val{:central}), √2/2) < 1e-8
    @test err_func(FiniteDiff.finite_difference_derivative(sin, π/4, Val{:complex}), √2/2) < 1e-15
end

@time @testset "Derivative StridedArray f : R -> R tests" begin
    @test err_func(FiniteDiff.finite_difference_derivative(sin, x, Val{:forward}), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_derivative(sin, x, Val{:central}), df_ref) < 1e-8
    @test err_func(FiniteDiff.finite_difference_derivative(sin, x, Val{:complex}), df_ref) < 1e-15

    @test err_func(FiniteDiff.finite_difference_derivative(sin, x, Val{:forward}, eltype(x), y, epsilon), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_derivative(sin, x, Val{:central}, eltype(x), y, epsilon), df_ref) < 1e-8
    @test err_func(FiniteDiff.finite_difference_derivative(sin, x, Val{:complex}, eltype(x), y, epsilon), df_ref) < 1e-15

    @test err_func(FiniteDiff.finite_difference_derivative!(df, sin, x, Val{:forward}), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_derivative!(df, sin, x, Val{:central}), df_ref) < 1e-8
    @test err_func(FiniteDiff.finite_difference_derivative!(df, sin, x, Val{:complex}), df_ref) < 1e-15

    @test err_func(FiniteDiff.finite_difference_derivative!(df, sin, x, Val{:forward}, eltype(x), y, epsilon), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_derivative!(df, sin, x, Val{:central}, eltype(x), y, epsilon), df_ref) < 1e-8
    @test err_func(FiniteDiff.finite_difference_derivative!(df, sin, x, Val{:complex}, eltype(x), y, epsilon), df_ref) < 1e-15

    @test err_func(FiniteDiff.finite_difference_derivative!(df, sin, x, forward_cache), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_derivative!(df, sin, x, central_cache), df_ref) < 1e-8
    @test err_func(FiniteDiff.finite_difference_derivative!(df, sin, x, complex_cache), df_ref) < 1e-15
end

x = x + im*x
f(x) = sin(x) + cos(x)
y = f.(x)
df = zero(x)
epsilon = similar(real(x))
df_ref = cos.(x) - sin.(x)
forward_cache = FiniteDiff.DerivativeCache(x, y, epsilon, Val{:forward})
central_cache = FiniteDiff.DerivativeCache(x, y, epsilon, Val{:central})

@time @testset "Derivative single point f : C -> C tests" begin
    @test err_func(FiniteDiff.finite_difference_derivative(f, π/4+im*π/4, Val{:forward}, Val{:Complex}), cos(π/4+im*π/4)-sin(π/4+im*π/4)) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative(f, π/4+im*π/4, Val{:central}, Val{:Complex}), cos(π/4+im*π/4)-sin(π/4+im*π/4)) < 1e-7
end

@time @testset "Derivative StridedArray f : C -> C tests" begin
    @test err_func(FiniteDiff.finite_difference_derivative(f, x, Val{:forward}), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative(f, x, Val{:central}), df_ref) < 1e-6

    @test err_func(FiniteDiff.finite_difference_derivative(f, x, Val{:forward}, eltype(x), y), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative(f, x, Val{:central}, eltype(x), y), df_ref) < 1e-6

    @test err_func(FiniteDiff.finite_difference_derivative(f, x, Val{:forward}, eltype(x), y, epsilon), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative(f, x, Val{:central}, eltype(x), y, epsilon), df_ref) < 1e-6

    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, Val{:forward}, eltype(x)), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, Val{:central}, eltype(x)), df_ref) < 1e-6

    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, Val{:forward}, eltype(x), y), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, Val{:central}, eltype(x), y), df_ref) < 1e-6

    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, Val{:forward}, eltype(x), y, epsilon), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, Val{:central}, eltype(x), y, epsilon), df_ref) < 1e-6

    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, forward_cache), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, central_cache), df_ref) < 1e-6
end

x = collect(range(-2π, stop=2π, length=100))
f(x) = sin(x) + im*cos(x)
y = f.(x)
df = fill(zero(Complex{eltype(x)}), size(x))
epsilon = similar(real(x))
df_ref = cos.(x) - im*sin.(x)
forward_cache = FiniteDiff.DerivativeCache(x, y, epsilon, Val{:forward}, eltype(df))
central_cache = FiniteDiff.DerivativeCache(x, y, epsilon, Val{:central}, eltype(df))

@time @testset "Derivative single point f : R -> C tests" begin
    @test err_func(FiniteDiff.finite_difference_derivative(f, π/4, Val{:forward}, Val{:Complex}), cos(π/4)-im*sin(π/4)) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative(f, π/4, Val{:central}, Val{:Complex}), cos(π/4)-im*sin(π/4)) < 1e-7
end

@time @testset "Derivative StridedArray f : R -> C tests" begin
    @test err_func(FiniteDiff.finite_difference_derivative(f, x, Val{:forward}, Complex{eltype(x)}), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative(f, x, Val{:forward}, Complex{eltype(x)}, relstep=sqrt(eps())), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative(f, x, Val{:central}, Complex{eltype(x)}), df_ref) < 1e-6

    @test err_func(FiniteDiff.finite_difference_derivative(f, x, Val{:forward}, Complex{eltype(x)}, y), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative(f, x, Val{:forward}, Complex{eltype(x)}, y, relstep=sqrt(eps())), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative(f, x, Val{:central}, Complex{eltype(x)}, y), df_ref) < 1e-6

    @test err_func(FiniteDiff.finite_difference_derivative(f, x, Val{:forward}, Complex{eltype(x)}, y, epsilon), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative(f, x, Val{:forward}, Complex{eltype(x)}, y, epsilon, relstep=sqrt(eps())), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative(f, x, Val{:central}, Complex{eltype(x)}, y, epsilon), df_ref) < 1e-6

    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, Val{:forward}, Complex{eltype(x)}), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, Val{:forward}, Complex{eltype(x)}, relstep=sqrt(eps())), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, Val{:central}, Complex{eltype(x)}), df_ref) < 1e-6

    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, Val{:forward}, Complex{eltype(x)}, y), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, Val{:forward}, Complex{eltype(x)}, y, relstep=sqrt(eps())), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, Val{:central}, Complex{eltype(x)}, y), df_ref) < 1e-6

    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, Val{:forward}, Complex{eltype(x)}, y, epsilon), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, Val{:forward}, Complex{eltype(x)}, y, epsilon, relstep=sqrt(eps())), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, Val{:central}, Complex{eltype(x)}, y, epsilon), df_ref) < 1e-6

    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, forward_cache), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, forward_cache, relstep=sqrt(eps())), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, central_cache), df_ref) < 1e-6
end

#=
x = x + im*x
f(x) = abs2(x)
y = f.(x)
df = fill(zero(eltype(x)), size(x))
epsilon = similar(real(x))
df_ref = 2*conj.(x)
forward_cache = FiniteDiff.DerivativeCache(x, y, epsilon, Val{:forward}, eltype(df))
central_cache = FiniteDiff.DerivativeCache(x, y, epsilon, Val{:central}, eltype(df))
@show typeof(FiniteDiff.finite_difference_derivative(f, 1.+im*1., Val{:forward}, real(eltype(x))))

@time @testset "Derivative single point f : C -> R tests" begin
    @test err_func(FiniteDiff.finite_difference_derivative(f, 1.+im*1., Val{:forward}, real(eltype(x))), 2.-2.*im) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative(f, 1.+im*1., Val{:central}, real(eltype(x))), 2.-2.*im) < 1e-7
end

@time @testset "Derivative StridedArray f : C -> R tests" begin
    @test err_func(FiniteDiff.finite_difference_derivative(f, x, Val{:forward}, real(eltype(x))), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative(f, x, Val{:central}, real(eltype(x))), df_ref) < 1e-6

    @test err_func(FiniteDiff.finite_difference_derivative(f, x, Val{:forward}, real(eltype(x)), y), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative(f, x, Val{:central}, real(eltype(x)), y), df_ref) < 1e-6

    @test err_func(FiniteDiff.finite_difference_derivative(f, x, Val{:forward}, real(eltype(x)), y, epsilon), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative(f, x, Val{:central}, real(eltype(x)), y, epsilon), df_ref) < 1e-6

    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, Val{:forward}, real(eltype(x))), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, Val{:central}, real(eltype(x))), df_ref) < 1e-6

    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, Val{:forward}, real(eltype(x)), y), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, Val{:central}, real(eltype(x)), y), df_ref) < 1e-6

    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, Val{:forward}, real(eltype(x)), y, epsilon), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, Val{:central}, real(eltype(x)), y, epsilon), df_ref) < 1e-6

    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, forward_cache), df_ref) < 1e-3
    @test err_func(FiniteDiff.finite_difference_derivative!(df, f, x, central_cache), df_ref) < 1e-6
end
=#

# Gradient tests
f(x) = 2x[1] + x[2]^2
x = rand(2)
z = copy(x)
ff(k) = !all(k .<= z) ? error() : f(k)
fx = f(x)
df = fill(0.,2)
df_ref = [2., 2*x[2]]
forward_cache = FiniteDiff.GradientCache(df,x,Val{:forward})
central_cache = FiniteDiff.GradientCache(df,x,Val{:central})
complex_cache = FiniteDiff.GradientCache(df,x,Val{:complex})

@time @testset "Gradient of f:vector->scalar real-valued tests" begin
    @test err_func(FiniteDiff.finite_difference_gradient(f, x, Val{:forward}), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient(ff, x, Val{:forward}, dir=-1), df_ref) < 1e-4
    @test_throws Any err_func(FiniteDiff.finite_difference_gradient(ff, x, Val{:forward}), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient(f, x, Val{:forward}, relstep=sqrt(eps())), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient(f, x, Val{:central}), df_ref) < 1e-8
    @test err_func(FiniteDiff.finite_difference_gradient(f, x, Val{:complex}), df_ref) < 1e-15

    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, Val{:forward}), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, Val{:forward}, relstep=sqrt(eps())), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, Val{:central}), df_ref) < 1e-8
    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, Val{:complex}), df_ref) < 1e-15

    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, forward_cache), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, forward_cache, relstep=sqrt(eps())), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, central_cache), df_ref) < 1e-8
    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, complex_cache), df_ref) < 1e-15
end

f(x) = 2x[1] + im*2x[1] + x[2]^2
x = x + im*x
fx = f(x)
df = zero(x)
df_ref = conj([2.0+2.0*im, 2.0*x[2]])
forward_cache = FiniteDiff.GradientCache(df,x,Val{:forward})
central_cache = FiniteDiff.GradientCache(df,x,Val{:central})

@time @testset "Gradient of f : C^N -> C tests" begin
    @test err_func(FiniteDiff.finite_difference_gradient(f, x, Val{:forward}), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient(f, x, Val{:forward}, relstep=sqrt(eps())), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient(f, x, Val{:central}), df_ref) < 1e-8

    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, Val{:forward}), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, Val{:forward}, relstep=sqrt(eps())), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, Val{:central}), df_ref) < 1e-8

    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, forward_cache), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, forward_cache, relstep=sqrt(eps())), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, central_cache), df_ref) < 1e-8
end

f(x) = sum(abs2, x)
x = ones(2) * (1 + im)
fx = f(x)
df = zero(x)
df_ref = 2*x
forward_cache = FiniteDiff.GradientCache(df,x,Val{:forward})
central_cache = FiniteDiff.GradientCache(df,x,Val{:central})

@time @testset "Gradient of f : C^N -> R tests" begin
    @test err_func(FiniteDiff.finite_difference_gradient(f, x, Val{:forward}), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient(f, x, Val{:central}), df_ref) < 1e-8

    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, Val{:forward}), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, Val{:central}), df_ref) < 1e-8

    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, forward_cache), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, central_cache), df_ref) < 1e-8
end

f(x) = 2*x[1] + im*x[2]^2
x = ones(2)
fx = f(x)
df = fill(zero(eltype(fx)), size(x))
df_ref = [2.0, -im*2*x[2]]
forward_cache = FiniteDiff.GradientCache(df,x,Val{:forward},eltype(df))
central_cache = FiniteDiff.GradientCache(df,x,Val{:central},eltype(df))

@time @testset "Gradient of f : R^N -> C tests" begin
    @test err_func(FiniteDiff.finite_difference_gradient(f, x, Val{:forward}, eltype(df)), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient(f, x, Val{:central}, eltype(df)), df_ref) < 1e-8

    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, Val{:forward}, eltype(df)), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, Val{:central}, eltype(df)), df_ref) < 1e-8

    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, forward_cache), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, central_cache), df_ref) < 1e-8
end

f(df,x) = (df[1]=sin(x); df[2]=cos(x); df)
z = x = 2π * rand()
fx = fill(0.,2)
f(fx,x)
ff(df, x) = !all(x .<= z) ? error() : f(df, x)
df = fill(0.,2)
df_ref = [cos(x), -sin(x)]
forward_cache = FiniteDiff.GradientCache(df,x,Val{:forward})
central_cache = FiniteDiff.GradientCache(df,x,Val{:central})
complex_cache = FiniteDiff.GradientCache(df,x,Val{:complex})


@time @testset "Gradient of f:scalar->vector real-valued tests" begin
    @test_broken err_func(FiniteDiff.finite_difference_gradient(f, x, Val{:forward}), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient(f, x, Val{:forward}, eltype(x), Val{true}, fx), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient(ff, x, Val{:forward}, eltype(x), Val{true}, fx, dir=-1), df_ref) < 1e-4
    @test_throws Any err_func(FiniteDiff.finite_difference_gradient(ff, x, Val{:forward}), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient(f, x, Val{:central}, eltype(x), Val{true}, fx), df_ref) < 1e-8
    @test err_func(FiniteDiff.finite_difference_gradient(f, x, Val{:complex}, eltype(x), Val{true}, fx), df_ref) < 1e-15

    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, Val{:forward}), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, Val{:central}), df_ref) < 1e-8
    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, Val{:complex}), df_ref) < 1e-15

    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, forward_cache), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, central_cache), df_ref) < 1e-8
    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, complex_cache), df_ref) < 1e-15
end

f(df,x) = (df[1]=sin(x); df[2]=cos(x); df)
x = (2π * rand()) * (1 + im)
fx = fill(zero(typeof(x)), 2)
f(fx,x)
df = zero(fx)
df_ref = [cos(x), -sin(x)]
forward_cache = FiniteDiff.GradientCache(df,x,Val{:forward})
central_cache = FiniteDiff.GradientCache(df,x,Val{:central})

@time @testset "Gradient of f:vector->scalar complex-valued tests" begin
    @test err_func(FiniteDiff.finite_difference_gradient(f, x, Val{:forward}, eltype(x), Val{true}, fx), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient(f, x, Val{:central}, eltype(x), Val{true}, fx), df_ref) < 3e-7

    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, Val{:forward}), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, Val{:central}), df_ref) < 3e-7

    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, forward_cache), df_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_gradient!(df, f, x, central_cache), df_ref) < 3e-7
end

# Jacobian tests
function iipf(fvec,x)
    fvec[1] = (x[1]+3)*(x[2]^3-7)+18
    fvec[2] = sin(x[2]*exp(x[1])-1)
end
function oopf(x)
    [(x[1]+3)*(x[2]^3-7)+18,
     sin(x[2]*exp(x[1])-1)]
end
x = rand(2); y = rand(2)
z = copy(x)
iipff(df, x) = !all(x .<= z) ? error() : iipf(df, x)
iipf(y,x)
oopff(x) = !all(x .<= z) ? error() : oopf(x)
J_ref = [[-7+x[2]^3 3*(3+x[1])*x[2]^2]; [exp(x[1])*x[2]*cos(1-exp(x[1])*x[2]) exp(x[1])*cos(1-exp(x[1])*x[2])]]
J = zero(J_ref)
df = zero(x)
df_ref = diag(J_ref)
epsilon = zero(x)
forward_cache = FiniteDiff.JacobianCache(x,Val{:forward},eltype(x))
central_cache = FiniteDiff.JacobianCache(x,Val{:central},eltype(x))
complex_cache = FiniteDiff.JacobianCache(x,Val{:complex},eltype(x))
f_in = copy(y)

@time @testset "Out-of-Place Jacobian StridedArray real-valued tests" begin
    @test err_func(FiniteDiff.finite_difference_jacobian(oopf, x, forward_cache), J_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_jacobian(oopff, x, forward_cache, dir=-1), J_ref) < 1e-4
    @test_throws Any err_func(FiniteDiff.finite_difference_jacobian(oopff, x, forward_cache), J_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_jacobian(oopf, x, forward_cache, relstep=sqrt(eps())), J_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_jacobian(oopf, x, forward_cache, f_in), J_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_jacobian(oopf, x, central_cache), J_ref) < 1e-8
    @test err_func(FiniteDiff.finite_difference_jacobian(oopf, x, Val{:central}), J_ref) < 1e-8
    @test err_func(FiniteDiff.finite_difference_jacobian(oopf, x, complex_cache), J_ref) < 1e-14
end

function test_iipJac(J_ref,args...;kwargs...)
    _J = zero(J_ref)
    FiniteDiff.finite_difference_jacobian!(_J,args...;kwargs...)
    _J
end
@time @testset "inPlace Jacobian StridedArray real-valued tests" begin
    @test err_func(test_iipJac(J_ref,iipf,x,forward_cache), J_ref) < 1e-4
    @test err_func(test_iipJac(J_ref,iipff,x,forward_cache,dir=-1), J_ref) < 1e-4
    @test_throws Any err_func(test_iipJac(J_ref,iipff, x, forward_cache), J_ref) < 1e-4
    @test err_func(test_iipJac(J_ref,iipf, x, forward_cache, relstep=sqrt(eps())), J_ref) < 1e-4
    @test err_func(test_iipJac(J_ref,iipf, x, forward_cache, f_in), J_ref) < 1e-4
    @test err_func(test_iipJac(J_ref,iipf, x, central_cache), J_ref) < 1e-8
    @test err_func(test_iipJac(J_ref,iipf, x, Val{:central}), J_ref) < 1e-8
    @test err_func(test_iipJac(J_ref,iipf, x, complex_cache), J_ref) < 1e-14
end

function iipf(fvec,x)
    fvec[1] = (im*x[1]+3)*(x[2]^3-7)+18
    fvec[2] = sin(x[2]*exp(x[1])-1)
end
function oopf(x)
    [(im*x[1]+3)*(x[2]^3-7)+18,
     sin(x[2]*exp(x[1])-1)]
end
x = rand(2) + im*rand(2)
y = similar(x)
iipf(y,x)
J_ref = [[im*(-7+x[2]^3) 3*(3+im*x[1])*x[2]^2]; [exp(x[1])*x[2]*cos(1-exp(x[1])*x[2]) exp(x[1])*cos(1-exp(x[1])*x[2])]]
J = zero(J_ref)
df = zero(x)
df_ref = diag(J_ref)
epsilon = zero(real.(x))
forward_cache = FiniteDiff.JacobianCache(x,Val{:forward},eltype(x))
central_cache = FiniteDiff.JacobianCache(x,Val{:central},eltype(x))
f_in = copy(y)

@time @testset "Out-of-Place Jacobian StridedArray f : C^N -> C^N tests" begin
    @test err_func(FiniteDiff.finite_difference_jacobian(oopf, x, forward_cache), J_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_jacobian(oopf, x, forward_cache, relstep=sqrt(eps())), J_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_jacobian(oopf, x, forward_cache, f_in), J_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_jacobian(oopf, x, central_cache), J_ref) < 1e-8
    @test err_func(FiniteDiff.finite_difference_jacobian(oopf, x, Val{:central}), J_ref) < 1e-8
end
@time @testset "inPlace Jacobian StridedArray f : C^N -> C^N tests" begin
    @test err_func(test_iipJac(J_ref, iipf, x, forward_cache), J_ref) < 1e-4
    @test err_func(test_iipJac(J_ref, iipf, x, forward_cache, relstep=sqrt(eps())), J_ref) < 1e-4
    @test err_func(test_iipJac(J_ref, iipf, x, forward_cache, f_in), J_ref) < 1e-4
    @test err_func(test_iipJac(J_ref, iipf, x, central_cache), J_ref) < 1e-8
    @test err_func(test_iipJac(J_ref, iipf, x, Val{:central}), J_ref) < 1e-8
end

# Non vector input
x = rand(2,2)
oopf(x) = x
iipf(fx,x) = (fx.=x)
J_ref = Matrix{Float64}(I,4,4)
@time @testset "Jacobian for non-vector inputs" begin
    @test err_func(FiniteDiff.finite_difference_jacobian(oopf, x, Val{:forward}), J_ref) < 1e-8
    @test err_func(FiniteDiff.finite_difference_jacobian(oopf, x, Val{:central}), J_ref) < 1e-8
    @test err_func(FiniteDiff.finite_difference_jacobian(oopf, x, Val{:complex}), J_ref) < 1e-8
    @test err_func(test_iipJac(J_ref, iipf, x, Val{:forward}, eltype(x), iipf(similar(x),x)), J_ref) < 1e-8
    @test err_func(test_iipJac(J_ref, iipf, x, Val{:central}, eltype(x), iipf(similar(x),x)), J_ref) < 1e-8
    @test err_func(test_iipJac(J_ref, iipf, x, Val{:complex}, eltype(x), iipf(similar(x),x)), J_ref) < 1e-8
end
# Hessian tests

f(x) = sin(x[1]) + cos(x[2])
x = rand(2)
H_ref = [-sin(x[1]) 0.0; 0.0 -cos(x[2])]
hcache = FiniteDiff.HessianCache(x)
hcache_oop = FiniteDiff.HessianCache(x,Val{:hcentral},Val{false})
H = similar(H_ref)

@time @testset "Hessian StridedArray f : R^N -> R tests" begin
    @test err_func(FiniteDiff.finite_difference_hessian(f, x), H_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_hessian(f, x, hcache), H_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_hessian(f, x, hcache_oop), H_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_hessian!(H, f, x), H_ref) < 1e-4
    @test err_func(FiniteDiff.finite_difference_hessian!(H, f, x, hcache), H_ref) < 1e-4
end
