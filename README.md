# FiniteDiff

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/FiniteDiff/stable/)

[![codecov](https://codecov.io/gh/JuliaDiff/FiniteDiff.jl/branch/master/graph/badge.svg)](https://app.codecov.io/gh/JuliaDiff/FiniteDiff.jl)
[![Build Status](https://github.com/JuliaDiff/FiniteDiff.jl/workflows/CI/badge.svg)](https://github.com/JuliaDiff/FiniteDiff.jl/actions?query=workflow%3ACI)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

## Overview

This package is for calculating derivatives, gradients, Jacobians, Hessians,
etc. numerically. This library is for maximizing speed while giving a usable
interface to end users in a way that specializes on array types and sparsity.
Included is:

- Fully non-allocating mutable forms for fast array support
- Fully non-mutating forms for static array support
- Coloring vectors for efficient calculation of sparse Jacobians
- GPU-compatible, to the extent that you can be with finite differencing.

If you want the fastest versions, create a cache and repeatedly call the
differencing functions at different `x` values (or with different `f` functions),
while if you want a quick and dirty numerical answer, directly call a differencing
function.

**For analogous sparse differentiation with automatic differentiation, see [SparseDiffTools.jl](https://github.com/JuliaDiff/SparseDiffTools.jl).**

### FiniteDiff.jl vs FiniteDifferences.jl

[FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl) and [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl)
are similar libraries: both calculate approximate derivatives numerically.
You should definitely use one or the other, rather than the legacy [Calculus.jl](https://github.com/JuliaMath/Calculus.jl) finite differencing, or reimplementing it yourself.
At some point in the future they might merge, or one might depend on the other.
Right now here are the differences:

 - FiniteDifferences.jl supports basically any type, where as FiniteDiff.jl supports only array-ish types
 - FiniteDifferences.jl supports higher order approximation
 - FiniteDiff.jl is carefully optimized to minimize allocations
 - FiniteDiff.jl supports coloring vectors for efficient calculation of sparse Jacobians

## General structure

The general structure of the library is as follows. You can call the differencing
functions directly and this will allocate a temporary cache to solve the problem
with. To make this non-allocating for repeat calls, you can call the cache
construction functions. Each cache construction function has two possibilities:
one version where you give it prototype arrays and it generates the cache
variables, and one fully non-allocating version where you give it the cache
variables. This is summarized as:

- Just want a quick derivative? Calculating once? Call the differencing function.
- Going to calculate the derivative multiple times but don't have cache arrays
  around? Use the allocating cache and then pass this into the differencing
  function (this will allocate only in the one cache construction).
- Have cache variables around from your own algorithm and want to re-use them
  in the differencing functions? Use the non-allocating cache construction
  and pass the cache to the differencing function.

See the [documentation](https://docs.sciml.ai/FiniteDiff/stable/) for details on the API.

### Function definitions

In all functions, the inplace form is `f!(dx,x)` while the out of place form is `dx = f(x)`.

### Coloring vectors

Coloring vectors are allowed to be supplied to the Jacobian routines, and these are
the directional derivatives for constructing the Jacobian. For example, an accurate
NxN tridiagonal Jacobian can be computed in just 4 `f` calls by using
`colorvec=repeat(1:3,N÷3)`. For information on automatically generating coloring
vectors of sparse matrices, see [SparseDiffTools.jl](https://github.com/JuliaDiff/SparseDiffTools.jl).

Hessian coloring support is coming soon!

## Contributing

- Please refer to the
  [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
  for guidance on PRs, issues, and other matters relating to contributing to SciML.
- See the [SciML Style Guide](https://github.com/SciML/SciMLStyle) for common coding practices and other style decisions.
- There are a few community forums:
  - The #diffeq-bridged and #sciml-bridged channels in the
    [Julia Slack](https://julialang.org/slack/)
  - The #diffeq-bridged and #sciml-bridged channels in the
    [Julia Zulip](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
  - On the [Julia Discourse forums](https://discourse.julialang.org)
  - See also [SciML Community page](https://sciml.ai/community/)