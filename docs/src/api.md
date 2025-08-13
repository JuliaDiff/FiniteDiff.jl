# API

```@docs
FiniteDiff
```

## Derivatives

```@docs
FiniteDiff.finite_difference_derivative
FiniteDiff.finite_difference_derivative!
FiniteDiff.DerivativeCache
```

## Gradients

```@docs
FiniteDiff.finite_difference_gradient
FiniteDiff.finite_difference_gradient!
FiniteDiff.GradientCache
```

Gradients are either a vector->scalar map `f(x)`, or a scalar->vector map `f(fx,x)` if `inplace=Val{true}` and `fx=f(x)` if `inplace=Val{false}`.

Note that here `fx` is a cached function call of `f`. If you provide `fx`, then
`fx` will be used in the forward differencing method to skip a function call.
It is on you to make sure that you update `cache.fx` every time before
calling `FiniteDiff.finite_difference_gradient!`. If `fx` is an immutable, e.g. a scalar or 
a `StaticArray`, `cache.fx` should be updated using `@set` from [Setfield.jl](https://github.com/jw3126/Setfield.jl).
A good use of this is if you have a cache array for the output of `fx` already being used, you can make it alias
into the differencing algorithm here.

## Jacobians

```@docs
FiniteDiff.finite_difference_jacobian
FiniteDiff.finite_difference_jacobian!
FiniteDiff.JacobianCache
```

Jacobians are for functions `f!(fx,x)` when using in-place `finite_difference_jacobian!`,
and `fx = f(x)` when using out-of-place `finite_difference_jacobian`. The out-of-place
jacobian will return a similar type as `jac_prototype` if it is not a `nothing`. For non-square
Jacobians, a cache which specifies the vector `fx` is required.

For sparse differentiation, pass a `colorvec` of matrix colors. `sparsity` should be a sparse
or structured matrix (`Tridiagonal`, `Banded`, etc. according to the ArrayInterfaceCore.jl specs)
to allow for decompression, otherwise the result will be the colorvec compressed Jacobian.

## Hessians

```@docs
FiniteDiff.finite_difference_hessian
FiniteDiff.finite_difference_hessian!
FiniteDiff.HessianCache
```

Hessians are for functions `f(x)` which return a scalar.

## Jacobian-Vector Products (JVP)

```@docs
FiniteDiff.finite_difference_jvp
FiniteDiff.finite_difference_jvp!
FiniteDiff.JVPCache
```

JVP functions compute the Jacobian-vector product `J(x) * v` efficiently without computing the full Jacobian matrix. This is particularly useful when you only need directional derivatives.

