# Hessians

Functions for computing Hessian matrices of scalar-valued functions.

## Function Requirements

Hessian functions are designed for scalar-valued functions `f(x)` where:

- `x` is a vector of parameters
- `f(x)` returns a scalar value
- The Hessian `H[i,j] = ∂²f/(∂x[i]∂x[j])` is automatically symmetrized

## Mathematical Background

For a scalar function `f: ℝⁿ → ℝ`, the Hessian central difference approximation is:

```
H[i,j] ≈ (f(x + eᵢhᵢ + eⱼhⱼ) - f(x + eᵢhᵢ - eⱼhⱼ) - f(x - eᵢhᵢ + eⱼhⱼ) + f(x - eᵢhᵢ - eⱼhⱼ)) / (4hᵢhⱼ)
```

where `eᵢ` is the i-th unit vector and `hᵢ` is the step size in dimension i.

## Performance Considerations

- **Complexity**: Requires `O(n²)` function evaluations for an n-dimensional input
- **Accuracy**: Central differences provide `O(h²)` accuracy for second derivatives  
- **Memory**: The result is returned as a `Symmetric` matrix view
- **Alternative**: For large problems, consider computing the gradient twice instead

## StaticArrays Support

The cache constructor automatically detects `StaticArray` types and adjusts the `inplace` parameter accordingly for optimal performance.

## Functions

```@docs
FiniteDiff.finite_difference_hessian
FiniteDiff.finite_difference_hessian!
```

## Cache

```@docs
FiniteDiff.HessianCache
```