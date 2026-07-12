using FiniteDiff, Test

# The documented user-facing API: every name rendered in a `@docs` block in the
# manual. These must be reachable and, on Julia 1.11+, marked `public` so that
# downstream packages' ExplicitImports/Aqua qualified-access checks pass.
const PUBLIC_API = (
    :finite_difference_derivative, :finite_difference_derivative!,
    :finite_difference_gradient, :finite_difference_gradient!,
    :finite_difference_jacobian, :finite_difference_jacobian!,
    :finite_difference_hessian, :finite_difference_hessian!,
    :finite_difference_jvp, :finite_difference_jvp!,
    :DerivativeCache, :GradientCache, :JacobianCache, :HessianCache, :JVPCache,
)

@testset "Public API is defined" begin
    for name in PUBLIC_API
        @test isdefined(FiniteDiff, name)
    end
end

@static if VERSION >= v"1.11.0-DEV.469"
    @testset "Public API is marked public" begin
        for name in PUBLIC_API
            @test Base.ispublic(FiniteDiff, name)
        end
    end
end
