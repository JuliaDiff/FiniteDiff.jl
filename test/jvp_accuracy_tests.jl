using FiniteDiff, LinearAlgebra, Test

# f_i(x) = x_i * x_{i+1} (cyclic): smooth, varies on the scale of x, and has an
# exactly known Jacobian, so the reference J*v carries no discretization error.
const N_JVP = 200
nxt(i) = i == N_JVP ? 1 : i + 1
fcyc(x) = [x[i] * x[nxt(i)] for i in eachindex(x)]
fcyc!(y, x) = (for i in eachindex(x)
        y[i] = x[i] * x[nxt(i)]
    end; y)
Jv_exact(x, v) = [x[nxt(i)] * v[i] + x[i] * v[nxt(i)] for i in eachindex(x)]

relerr(a, b) = norm(a - b) / norm(b)

x_big = [100.0 * (1 + i / N_JVP) for i in 1:N_JVP]
v_unit = [sin(float(i)) for i in 1:N_JVP]
v_unit ./= norm(v_unit)
ref_big = Jv_exact(x_big, v_unit)

@testset "JVP step size and accuracy" begin
    @testset "JVP step size scales with norm(x)" begin
        @test relerr(FiniteDiff.finite_difference_jvp(fcyc, copy(x_big), v_unit), ref_big) < 1e-7
        jvp = similar(x_big)
        FiniteDiff.finite_difference_jvp!(jvp, fcyc!, copy(x_big), v_unit)
        @test relerr(jvp, ref_big) < 1e-7

        @test relerr(FiniteDiff.finite_difference_jvp(fcyc, copy(x_big), v_unit, Val{:central}),
            ref_big) < 1e-10
        FiniteDiff.finite_difference_jvp!(jvp, fcyc!, copy(x_big), v_unit, Val{:central})
        @test relerr(jvp, ref_big) < 1e-10
    end

    @testset "JVP is invariant to the scaling of v" begin
        base = FiniteDiff.finite_difference_jvp(fcyc, copy(x_big), v_unit)
        for s in (1.0e-4, 1.0e2, 1.0e4)
            scaled = FiniteDiff.finite_difference_jvp(fcyc, copy(x_big), s .* v_unit) ./ s
            @test relerr(scaled, base) < 1e-8
            @test relerr(scaled, ref_big) < 1e-7
        end
    end

    @testset "JVP degenerate directions and states" begin
        zerojvp = FiniteDiff.finite_difference_jvp(fcyc, copy(x_big), zero(v_unit))
        @test all(isfinite, zerojvp)
        @test iszero(zerojvp)

        # norm(x) == 0 must fall back to the absstep floor rather than a zero step
        zerox = zeros(N_JVP)
        atzero = FiniteDiff.finite_difference_jvp(fcyc, zerox, v_unit)
        @test all(isfinite, atzero)
        @test norm(atzero) < 1e-6

        # a non-finite entry of x must not contaminate the step, and hence every output
        nanx = copy(x_big)
        nanx[1] = NaN
        nanjvp = FiniteDiff.finite_difference_jvp(fcyc, nanx, v_unit)
        @test count(isnan, nanjvp) == 2
    end

    @testset "JVP complex-valued state" begin
        xc = [(1 + im) * 100.0 * (1 + i / N_JVP) for i in 1:N_JVP]
        vc = v_unit .+ im .* reverse(v_unit)
        vc ./= norm(vc)
        refc = Jv_exact(xc, vc)
        @test relerr(FiniteDiff.finite_difference_jvp(fcyc, copy(xc), vc), refc) < 1e-7
        jvpc = similar(xc)
        FiniteDiff.finite_difference_jvp!(jvpc, fcyc!, copy(xc), vc)
        @test relerr(jvpc, refc) < 1e-7
    end
end
