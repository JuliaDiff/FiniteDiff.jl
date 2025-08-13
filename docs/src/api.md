# API Reference

```@docs
FiniteDiff
```

FiniteDiff.jl provides fast, non-allocating finite difference calculations with support for sparsity patterns and various array types. The API is organized into several categories:

## Function Categories

### [Derivatives](@ref derivatives)
Single and multi-point derivatives of scalar functions.

### [Gradients](@ref gradients)  
Gradients of scalar-valued functions with respect to vector inputs.

### [Jacobians](@ref jacobians)
Jacobian matrices of vector-valued functions, including sparse Jacobian support.

### [Hessians](@ref hessians)
Hessian matrices of scalar-valued functions.

### [Jacobian-Vector Products](@ref jvp)
Efficient computation of directional derivatives without forming full Jacobians.

### [Utilities](@ref utilities)
Internal utilities and helper functions.

## Quick Start

All functions follow a consistent API pattern:

- **Cache-less versions**: `finite_difference_*` - convenient but allocate temporary arrays
- **In-place versions**: `finite_difference_*!` - efficient, non-allocating when used with caches  
- **Cache constructors**: `*Cache` - pre-allocate work arrays for repeated computations

## Method Selection

Choose your finite difference method based on accuracy and performance needs:

- **Forward differences**: Fast, `O(h)` accuracy, requires `O(n)` function evaluations
- **Central differences**: More accurate `O(h²)`, requires `O(2n)` function evaluations
- **Complex step**: Machine precision accuracy, `O(n)` evaluations, requires complex-analytic functions

## Performance Tips

1. **Use caches** for repeated computations to avoid allocations
2. **Consider sparsity** for large Jacobians with known sparsity patterns
3. **Choose appropriate methods** based on your accuracy requirements
4. **Leverage JVPs** when you only need directional derivatives

