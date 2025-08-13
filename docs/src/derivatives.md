# Derivatives

Functions for computing derivatives of scalar-valued functions.

## Overview

Derivatives are computed for scalar→scalar maps `f(x)` where `x` can be a single point or a collection of points. The derivative functions support:

- **Forward differences**: `O(1)` function evaluation per point, `O(h)` accuracy
- **Central differences**: `O(2)` function evaluations per point, `O(h²)` accuracy  
- **Complex step**: `O(1)` function evaluation per point, machine precision accuracy

For optimal performance with repeated computations, use the cached versions with `DerivativeCache`.

## Functions

```@docs
FiniteDiff.finite_difference_derivative
FiniteDiff.finite_difference_derivative!
```

## Cache

```@docs
FiniteDiff.DerivativeCache
```