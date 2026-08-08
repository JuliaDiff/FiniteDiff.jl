# Jacobian-Vector Products (JVP)

Functions for computing Jacobian-vector products efficiently without forming the full Jacobian matrix.

## Mathematical Background

The JVP computes `J(x) * v` where `J(x)` is the Jacobian of function `f` at point `x` and `v` is a direction vector. This is computed using finite difference approximations:

- **Forward**: `J(x) * v ≈ (f(x + h*v) - f(x)) / h`  
- **Central**: `J(x) * v ≈ (f(x + h*v) - f(x - h*v)) / (2h)`

where `h` is the step size and `v` is the direction vector.

## Step Size

Because `f` is evaluated at `x + h*v`, the step `h` has units of `[x]/[v]`: it is not a
step in `x` and it is not dimensionless. The step used is

```julia
h = max(relstep * norm(x), absstep) * dir / norm(v)
```

so that the perturbation `h*v` is a `relstep` relative change of `x`, floored at
`absstep`, and the computed JVP is invariant (up to that floor) to rescaling of `v`.
This matters for matrix-free Krylov solvers, where `v` is a unit-norm basis vector and
`norm(x)` may be large: a step that ignores `norm(x)` is stuck at the `absstep` floor
and the JVP loses accuracy proportionally to `norm(x)`.

## Performance Benefits

JVP functions are particularly efficient when you only need directional derivatives:

- **Function evaluations**: Only 2 function evaluations (vs `O(n)` for full Jacobian)
- **Forward differences**: 2 function evaluations, `O(h)` accuracy
- **Central differences**: 2 function evaluations, `O(h²)` accuracy  
- **Memory efficient**: No need to store the full Jacobian matrix

## Use Cases

JVP is particularly useful for:

- **Optimization**: Computing directional derivatives along search directions
- **Sparse directions**: When `v` has few non-zero entries
- **Memory constraints**: Avoiding storage of large Jacobian matrices
- **Newton methods**: Computing Newton steps `J⁻¹ * v` iteratively

## Limitations

- **Complex step**: JVP does not currently support complex step differentiation (`Val(:complex)`)
- **In-place functions**: For in-place function evaluation, ensure proper cache sizing

## Functions

```@docs
FiniteDiff.finite_difference_jvp
FiniteDiff.finite_difference_jvp!
```

## Cache

```@docs
FiniteDiff.JVPCache
```