using FiniteDiff, LinearAlgebra, SparseArrays, StaticArrays, Test

# Tests for issue #213: caches must be safe to reuse at a new x, regardless of
# how their internal scratch fields (e.g. JacobianCache.x1) were initialized.
# The original symptom was DI building a JacobianCache from `similar(x)`
# (uninitialized) and getting junk Jacobians in :central mode.

const J_REF = [2.0 0.0; 0.0 3.0; 4.0 0.0]
foo_oop(x) = [2x[1], 3x[2], 4x[1]]
foo_iip!(y, x) = (y[1] = 2x[1]; y[2] = 3x[2]; y[3] = 4x[1]; y)

# A non-zero point where the bug becomes obvious — at zeros(2) the junk in x1
# cancels by symmetry on this affine function and hides the issue.
const X_TEST = [1.0, 2.0]

"""Build a JacobianCache whose scratch fields are explicitly poisoned with a
huge value, mimicking what happens when a caller hands FiniteDiff a cache
allocated via `similar(x)` (which gives uninitialized memory)."""
function poisoned_jcache(fdtype; x_template = X_TEST, y_template = foo_oop(X_TEST), poison = 1.0e10)
    x1 = fill(poison, length(x_template))
    fx = fill(poison, length(y_template))
    if fdtype === Val(:complex)
        FiniteDiff.JacobianCache(x1, fx, nothing, fdtype)
    else
        fx1 = fill(poison, length(y_template))
        FiniteDiff.JacobianCache(x1, fx, fx1, fdtype)
    end
end

@testset "Cache reuse safety (issue #213)" begin

@testset "JacobianCache out-of-place reuse" begin
    @testset "fresh cache reused at new x" for fdtype in (Val(:forward), Val(:central), Val(:complex))
        cache = FiniteDiff.JacobianCache(zeros(2), zeros(3), fdtype)
        # Exercise the cache once at x_old, then reuse at X_TEST.
        FiniteDiff.finite_difference_jacobian(foo_oop, zeros(2), cache)
        J = FiniteDiff.finite_difference_jacobian(foo_oop, X_TEST, cache)
        @test J ≈ J_REF atol=1e-6
    end

    @testset "cache built with garbage x1/fx ($(fdtype))" for fdtype in (Val(:forward), Val(:central), Val(:complex))
        cache = poisoned_jcache(fdtype)
        J = FiniteDiff.finite_difference_jacobian(foo_oop, X_TEST, cache)
        @test J ≈ J_REF atol=1e-6
    end

    @testset "cache built with garbage x1/fx + sparsity ($(fdtype))" for fdtype in (Val(:forward), Val(:central))
        spJ = sparse(J_REF)
        cache = poisoned_jcache(fdtype)
        J = FiniteDiff.finite_difference_jacobian(foo_oop, X_TEST, cache;
                                                   sparsity = spJ, jac_prototype = spJ)
        @test Matrix(J) ≈ J_REF atol=1e-6
    end
end

@testset "JacobianCache in-place reuse" begin
    @testset "fresh cache reused at new x ($(fdtype))" for fdtype in (Val(:forward), Val(:central), Val(:complex))
        cache = FiniteDiff.JacobianCache(zeros(2), zeros(3), fdtype)
        J = zeros(3, 2)
        FiniteDiff.finite_difference_jacobian!(J, foo_iip!, zeros(2), cache)
        fill!(J, 0)
        FiniteDiff.finite_difference_jacobian!(J, foo_iip!, X_TEST, cache)
        @test J ≈ J_REF atol=1e-6
    end

    @testset "cache built with garbage x1/fx ($(fdtype))" for fdtype in (Val(:forward), Val(:central), Val(:complex))
        cache = poisoned_jcache(fdtype)
        J = zeros(3, 2)
        FiniteDiff.finite_difference_jacobian!(J, foo_iip!, X_TEST, cache)
        @test J ≈ J_REF atol=1e-6
    end

    @testset "in-place :central must not mutate x (sparse path)" begin
        spJ = sparse(J_REF)
        cache = poisoned_jcache(Val(:central))
        J = zeros(3, 2)
        x = copy(X_TEST)
        x_orig = copy(x)
        FiniteDiff.finite_difference_jacobian!(J, foo_iip!, x, cache;
                                                sparsity = spJ, colorvec = 1:2)
        @test Matrix(J) ≈ J_REF atol=1e-6
        @test x == x_orig  # x should be restored / unmutated
    end
end

@testset "GradientCache reuse" begin
    # `:complex` requires an analytic function, so don't use abs2 here.
    g(x) = x[1]^2 + x[2]^2 + x[3]^2
    grad_ref = [2.0, 4.0, 6.0]
    x = [1.0, 2.0, 3.0]

    # Use the allocating constructor so buffer types are correct, then poison
    # any non-`nothing` buffer to simulate a stale cache.
    @testset "vector → scalar with poisoned cache ($(fdtype))" for fdtype in (Val(:forward), Val(:central), Val(:complex))
        df = zeros(3)
        cache = FiniteDiff.GradientCache(df, x, fdtype, Float64, Val(false))
        for fld in (:c1, :c2, :c3)
            buf = getfield(cache, fld)
            buf isa AbstractArray && fill!(buf, 1e10)
        end
        grad = zeros(3)
        FiniteDiff.finite_difference_gradient!(grad, g, x, cache)
        @test grad ≈ grad_ref atol=1e-5
    end

    @testset "fresh cache reused at new x ($(fdtype))" for fdtype in (Val(:forward), Val(:central))
        cache = FiniteDiff.GradientCache(zeros(3), zeros(3), fdtype)
        grad = zeros(3)
        FiniteDiff.finite_difference_gradient!(grad, g, zeros(3), cache)
        FiniteDiff.finite_difference_gradient!(grad, g, x, cache)
        @test grad ≈ grad_ref atol=1e-5
    end
end

@testset "JVPCache reuse" begin
    foo_iip!_3 = (y, x) -> (y[1] = 2x[1]; y[2] = 3x[2]; y[3] = 4x[1]; y)
    v = [1.0, 0.0]
    jvp_ref = J_REF * v

    @testset "garbage cache ($(fdtype))" for fdtype in (Val(:forward), Val(:central))
        x1 = fill(1e10, 2)
        fx1 = fill(1e10, 3)
        cache = FiniteDiff.JVPCache(x1, fx1, fdtype)
        jvp = zeros(3)
        FiniteDiff.finite_difference_jvp!(jvp, foo_iip!_3, X_TEST, v, cache)
        @test jvp ≈ jvp_ref atol=1e-6
    end
end

@testset "HessianCache reuse" begin
    h(x) = x[1]^2 + 2 * x[2]^2
    H_ref = [2.0 0.0; 0.0 4.0]

    xpp = fill(1e10, 2); xpm = fill(1e10, 2); xmp = fill(1e10, 2); xmm = fill(1e10, 2)
    cache = FiniteDiff.HessianCache(xpp, xpm, xmp, xmm, Val(:hcentral), Val(true))
    H = zeros(2, 2)
    FiniteDiff.finite_difference_hessian!(H, h, X_TEST, cache)
    @test H ≈ H_ref atol=1e-3
end

# Mirrors the failure mode from JuliaDiff/DifferentiationInterface.jl#983: a
# caller building a cache with `similar(x)` fields and then asking for a
# Jacobian via the non-allocating entry point.
@testset "DI-style similar() cache (issue #983 reproduction)" begin
    foo(x) = [2x[1], 3x[2], 4x[1]]
    y = foo(X_TEST)

    @testset "$(fdtype)" for fdtype in (Val(:forward), Val(:central), Val(:complex))
        x1 = similar(X_TEST)
        fx = similar(y)
        if fdtype === Val(:complex)
            cache = FiniteDiff.JacobianCache(x1, fx, nothing, fdtype)
        else
            fx1 = similar(y)
            cache = FiniteDiff.JacobianCache(x1, fx, fx1, fdtype)
        end
        J = FiniteDiff.finite_difference_jacobian(foo, X_TEST, cache)
        @test J ≈ J_REF atol=1e-6
    end
end

end  # outer testset
