# Step Size Selection (Epsilons)

Functions and theory for computing optimal step sizes in finite difference approximations.

## Theory

The choice of step size (epsilon) in finite difference methods is critical for accuracy. Too large a step leads to truncation error, while too small a step leads to round-off error. The optimal step size balances these two sources of error.

### Error Analysis

For a function `f` with bounded derivatives, the total error in finite difference approximations consists of:

1. **Truncation Error**: Comes from the finite difference approximation itself
   - Forward differences: `O(h)` where `h` is the step size
   - Central differences: `O(h²)`
   - Hessian central differences: `O(h²)` for second derivatives

2. **Round-off Error**: Comes from floating-point arithmetic
   - Forward differences: `O(eps/h)` where `eps` is machine epsilon
   - Central differences: `O(eps/h)`

### Optimal Step Sizes

Minimizing the total error `truncation + round-off` gives optimal step sizes:

- **Forward differences**: `h* = sqrt(eps)` - balances `O(h)` truncation with `O(eps/h)` round-off
- **Central differences**: `h* = eps^(1/3)` - balances `O(h²)` truncation with `O(eps/h)` round-off
- **Hessian central**: `h* = eps^(1/4)` - balances `O(h²)` truncation for mixed derivatives
- **Complex step**: `h* = eps` - no subtractive cancellation, only limited by machine precision

## Adaptive Step Sizing

The step size computation uses both relative and absolute components:

```julia
epsilon = max(relstep * abs(x), absstep) * dir
```

This ensures:
- **Large values**: Use relative step `relstep * |x|` for scale-invariant accuracy
- **Small values**: Use absolute step `absstep` to avoid underflow
- **Direction**: Multiply by `dir` (±1) for forward differences

## Implementation

The step size computation is handled by internal functions:

- **`compute_epsilon(fdtype, x, relstep, absstep, dir)`**: Computes the actual step size for a given finite difference method and input value
- **`default_relstep(fdtype, T)`**: Returns the optimal relative step size for a given method and numeric type

These functions are called automatically by all finite difference routines, but understanding their behavior can help with custom implementations or debugging numerical issues.

## Special Cases

### Complex Step Differentiation

For complex step differentiation, the step size is simply machine epsilon since this method avoids subtractive cancellation entirely:

⚠️ **Important**: The function `f` must be complex analytic when the input is complex!

### Sparse Jacobians

When computing sparse Jacobians with graph coloring, the step size is computed based on the norm of the perturbation vector to ensure balanced accuracy across all columns in the same color group.

## Practical Considerations

- **Default step sizes** are optimal for most smooth functions
- **Custom step sizes** may be needed for functions with unusual scaling or near-discontinuities
- **Relative steps** should scale with the magnitude of the input
- **Absolute steps** provide a fallback for inputs near zero
- **Direction parameter** allows for one-sided differences when needed (e.g., at domain boundaries)