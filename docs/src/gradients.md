# Gradients

Functions for computing gradients of scalar-valued functions with respect to vector inputs.

## Function Types

Gradients support two types of function mappings:

- **Vector→scalar**: `f(x)` where `x` is a vector and `f` returns a scalar
- **Scalar→vector**: `f(fx, x)` for in-place evaluation or `fx = f(x)` for out-of-place

## Performance Notes

- **Forward differences**: `O(n)` function evaluations, `O(h)` accuracy
- **Central differences**: `O(2n)` function evaluations, `O(h²)` accuracy
- **Complex step**: `O(n)` function evaluations, machine precision accuracy

## Cache Management

When using `GradientCache` with pre-computed function values:

- If you provide `fx`, then `fx` will be used in forward differencing to skip a function call
- You must update `cache.fx` before each call to `finite_difference_gradient!`
- For immutable types (scalars, `StaticArray`), use `@set` from [Setfield.jl](https://github.com/jw3126/Setfield.jl)
- Consider aliasing existing arrays into the cache for memory efficiency

## Functions

```@docs
FiniteDiff.finite_difference_gradient
FiniteDiff.finite_difference_gradient!
```

## Cache

```@docs
FiniteDiff.GradientCache
```