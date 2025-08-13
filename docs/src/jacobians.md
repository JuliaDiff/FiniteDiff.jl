# Jacobians

Functions for computing Jacobian matrices of vector-valued functions.

## Functions

```@docs
FiniteDiff.finite_difference_jacobian
FiniteDiff.finite_difference_jacobian!
```

## Cache

```@docs
FiniteDiff.JacobianCache
```

## Function Types

Jacobians support the following function signatures:

- **Out-of-place**: `fx = f(x)` where both `x` and `fx` are vectors
- **In-place**: `f!(fx, x)` where `f!` modifies `fx` in-place

## Sparse Jacobians

FiniteDiff.jl provides efficient sparse Jacobian computation using graph coloring:

- Pass a `colorvec` of matrix colors to enable column compression
- Provide `sparsity` as a sparse or structured matrix (`Tridiagonal`, `Banded`, etc.)
- Supports automatic sparsity pattern detection via ArrayInterfaceCore.jl
- Results are automatically decompressed unless `sparsity=nothing`

## Performance Notes

- **Forward differences**: `O(n)` function evaluations, `O(h)` accuracy  
- **Central differences**: `O(2n)` function evaluations, `O(h²)` accuracy
- **Complex step**: `O(n)` function evaluations, machine precision accuracy
- **Sparse Jacobians**: Use graph coloring to reduce function evaluations significantly

For non-square Jacobians, specify the output vector `fx` when creating the cache to ensure proper sizing.