# FiniteDiff

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/FiniteDiff/dev/)

[![codecov](https://codecov.io/gh/JuliaDiff/FiniteDiff.jl/branch/master/graph/badge.svg)](https://app.codecov.io/gh/JuliaDiff/FiniteDiff.jl)
[![Build Status](https://github.com/JuliaDiff/FiniteDiff.jl/workflows/CI/badge.svg)](https://github.com/JuliaDiff/FiniteDiff.jl/actions?query=workflow%3ACI)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

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

#### FiniteDiff.jl vs FiniteDifferences.jl
[FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl) and [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl)
are similar libraries: both calculate approximate derivatives numerically.
You should definately use one or the other, rather than the legacy [Calculus.jl](https://github.com/JuliaMath/Calculus.jl) finite differencing, or reimplementing it yourself.
At some point in the future they might merge, or one might depend on the other.
Right now here are the differences:

 - FiniteDifferences.jl supports basically any type, where as FiniteDiff.jl supports only array-ish types
 - FiniteDifferences.jl supports higher order approximation
 - FiniteDiff.jl is carefully optimized to minimize allocations
 - FiniteDiff.jl supports coloring vectors for efficient calculation of sparse Jacobians

## Tutorials

### Tutorial 1: Fast Dense Jacobians

It's always fun to start out with a tutorial before jumping into the details!
Suppose we had the functions:

```julia
using FiniteDiff, StaticArrays

fcalls = 0
function f(dx,x) # in-place
  global fcalls += 1
  for i in 2:length(x)-1
    dx[i] = x[i-1] - 2x[i] + x[i+1]
  end
  dx[1] = -2x[1] + x[2]
  dx[end] = x[end-1] - 2x[end]
  nothing
end

const N = 10
handleleft(x,i) = i==1 ? zero(eltype(x)) : x[i-1]
handleright(x,i) = i==length(x) ? zero(eltype(x)) : x[i+1]
function g(x) # out-of-place
  global fcalls += 1
  @SVector [handleleft(x,i) - 2x[i] + handleright(x,i) for i in 1:N]
end
```

and we wanted to calculate the derivatives of them. The simplest thing we can
do is ask for the Jacobian. If we want to allocate the result, we'd use the
allocating function `finite_difference_jacobian` on a 1-argument function `g`:

```julia
x = @SVector rand(N)
FiniteDiff.finite_difference_jacobian(g,x)

#=
10×10 SArray{Tuple{10,10},Float64,2,100} with indices SOneTo(10)×SOneTo(10):
 -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
  1.0  -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
  0.0   1.0  -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0
  0.0   0.0   1.0  -2.0   1.0   0.0   0.0   0.0   0.0   0.0
  0.0   0.0   0.0   1.0  -2.0   1.0   0.0   0.0   0.0   0.0
  0.0   0.0   0.0   0.0   1.0  -2.0   1.0   0.0   0.0   0.0
  0.0   0.0   0.0   0.0   0.0   1.0  -2.0   1.0   0.0   0.0
  0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0   1.0   0.0
  0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0   1.0
  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0
=#
```

FiniteDiff.jl assumes you're a smart cookie, and so if you used an
out-of-place function then it'll not mutate vectors at all, and is thus compatible
with objects like StaticArrays and will give you a fast Jacobian.

But if you wanted to use mutation, then we'd have to use the in-place function
`f` and call the mutating form:

```julia
x = rand(10)
output = zeros(10,10)
FiniteDiff.finite_difference_jacobian!(output,f,x)
output

#=
10×10 Array{Float64,2}:
 -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
  1.0  -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
  0.0   1.0  -2.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0
  0.0   0.0   1.0  -2.0   1.0   0.0   0.0   0.0   0.0   0.0
  0.0   0.0   0.0   1.0  -2.0   1.0   0.0   0.0   0.0   0.0
  0.0   0.0   0.0   0.0   1.0  -2.0   1.0   0.0   0.0   0.0
  0.0   0.0   0.0   0.0   0.0   1.0  -2.0   1.0   0.0   0.0
  0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0   1.0   0.0
  0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0   1.0
  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0  -2.0
=#
```

But what if you want this to be completely non-allocating on your mutating form?
Then you need to preallocate a cache:

```julia
cache = FiniteDiff.JacobianCache(x)
```

and now using this cache avoids allocating:

```julia
@time FiniteDiff.finite_difference_jacobian!(output,f,x,cache) # 0.000008 seconds (7 allocations: 224 bytes)
```

And that's pretty much it! Gradients and Hessians work similarly: out of place
doesn't index, and in-place avoids allocations. Either way, you're fast. GPUs
etc. all work.

### Tutorial 2: Fast Sparse Jacobians

Now let's exploit sparsity. If we knew the sparsity pattern we could write it
down analytically as a sparse matrix, but let's assume we don't. Thus we can
use [SparsityDetection.jl](https://github.com/JuliaDiffEq/SparsityDetection.jl)
to automatically get the sparsity pattern of the Jacobian as a sparse matrix:

```julia
using SparsityDetection, SparseArrays
in = rand(10)
out = similar(in)
sparsity_pattern = sparsity!(f,out,in)
sparsejac = Float64.(sparse(sparsity_pattern))
```

Then we can use [SparseDiffTools.jl](https://github.com/JuliaDiffEq/SparseDiffTools.jl)
to get the color vector:

```julia
using SparseDiffTools
colors = matrix_colors(sparsejac)
```

Now we can do sparse differentiation by passing the color vector and the sparsity
pattern:

```julia
sparsecache = FiniteDiff.JacobianCache(x,colorvec=colors,sparsity=sparsejac)
FiniteDiff.finite_difference_jacobian!(sparsejac,f,x,sparsecache)
```

Note that the number of `f` evaluations to fill a Jacobian is `1+maximum(colors)`.
By default, `colors=1:length(x)`, so in this case we went from 10 function calls
to 4. The sparser the matrix, the more the gain! We can measure this as well:

```julia
fcalls = 0
FiniteDiff.finite_difference_jacobian!(output,f,x,cache)
fcalls #11

fcalls = 0
FiniteDiff.finite_difference_jacobian!(sparsejac,f,x,sparsecache)
fcalls #4
```

### Tutorial 3: Fast Tridiagonal Jacobians

Handling dense matrices? Easy. Handling sparse matrices? Cool stuff. Automatically
specializing on the exact structure of a matrix? Even better. FiniteDiff can
specialize on types which implement the
[ArrayInterfaceCore.jl](https://github.com/JuliaDiffEq/ArrayInterfaceCore.jl) interface.
This includes:

- Diagonal
- Bidiagonal
- UpperTriangular and LowerTriangular
- Tridiagonal and SymTridiagonal
- [BandedMatrices.jl](https://github.com/JuliaMatrices/BandedMatrices.jl)
- [BlockBandedMatrices.jl](https://github.com/JuliaMatrices/BlockBandedMatrices.jl)

Our previous example had a Tridiagonal Jacobian, so let's use this. If we just
do

```julia
using ArrayInterfaceCore, LinearAlgebra
tridiagjac = Tridiagonal(output)
colors = matrix_colors(jac)
```

we get the analytical solution to the optimal matrix colors for our structured
Jacobian. Now we can use this in our differencing routines:

```julia
tridiagcache = FiniteDiff.JacobianCache(x,colorvec=colors,sparsity=tridiagjac)
FiniteDiff.finite_difference_jacobian!(tridiagjac,f,x,tridiagcache)
```

It'll use a special iteration scheme dependent on the matrix type to accelerate
it beyond general sparse usage.

### Tutorial 4: Fast Block Banded Matrices

Now let's showcase a difficult example. Say we had a large system of partial
differential equations, with a function like:

```julia
function pde(out, x)
	x = reshape(x, 100, 100)
	out = reshape(out, 100, 100)
	for i in 1:100
		for j in 1:100
			out[i, j] = x[i, j] + x[max(i -1, 1), j] + x[min(i+1, size(x, 1)), j] +  x[i, max(j-1, 1)]  + x[i, min(j+1, size(x, 2))]
		end
	end
	return vec(out)
end
x = rand(10000)
```

In this case, we can see that our sparsity pattern is a BlockBandedMatrix, so
let's specialize the Jacobian calculation on this fact:

```julia
using FillArrays, BlockBandedMatrices
Jbbb = BandedBlockBandedMatrix(Ones(10000, 10000), fill(100, 100), fill(100, 100), (1, 1), (1, 1))
colorsbbb = ArrayInterfaceCore.matrix_colors(Jbbb)
bbbcache = FiniteDiff.JacobianCache(x,colorvec=colorsbbb,sparsity=Jbbb)
FiniteDiff.finite_difference_jacobian!(Jbbb, pde, x, bbbcache)
```

And boom, a fast Jacobian filling algorithm on your special matrix.

## General Structure

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

## f Definitions

In all functions, the inplace form is `f!(dx,x)` while the out of place form is `dx = f(x)`.

## colorvec Vectors

colorvec vectors are allowed to be supplied to the Jacobian routines, and these are
the directional derivatives for constructing the Jacobian. For example, an accurate
NxN tridiagonal Jacobian can be computed in just 4 `f` calls by using
`colorvec=repeat(1:3,N÷3)`. For information on automatically generating colorvec
vectors of sparse matrices, see [SparseDiffTools.jl](https://github.com/JuliaDiff/SparseDiffTools.jl).

Hessian coloring support is coming soon!

## Scalar Derivatives

```julia
FiniteDiff.finite_difference_derivative(f, x::T, fdtype::Type{T1}=Val{:central},
    returntype::Type{T2}=eltype(x), f_x::Union{Nothing,T}=nothing)
```

## Multi-Point Derivatives

### Differencing Calls

```julia
# Cache-less but non-allocating if `fx` and `epsilon` are supplied
# fx must be f(x)
FiniteDiff.finite_difference_derivative(
    f,
    x          :: AbstractArray{<:Number},
    fdtype     :: Type{T1} = Val{:central},
    returntype :: Type{T2} = eltype(x),      # return type of f
    fx         :: Union{Nothing,AbstractArray{<:Number}} = nothing,
    epsilon    :: Union{Nothing,AbstractArray{<:Real}} = nothing;
    [epsilon_factor])

FiniteDiff.finite_difference_derivative!(
    df         :: AbstractArray{<:Number},
    f,
    x          :: AbstractArray{<:Number},
    fdtype     :: Type{T1} = Val{:central},
    returntype :: Type{T2} = eltype(x),
    fx         :: Union{Nothing,AbstractArray{<:Number}} = nothing,
    epsilon    :: Union{Nothing,AbstractArray{<:Real}}   = nothing;
    [epsilon_factor])

# Cached
FiniteDiff.finite_difference_derivative!(
    df::AbstractArray{<:Number},
    f,
    x::AbstractArray{<:Number},
    cache::DerivativeCache{T1,T2,fdtype,returntype};
    [epsilon_factor])
```

### Allocating and Non-Allocating Constructor

```julia
FiniteDiff.DerivativeCache(
    x          :: AbstractArray{<:Number},
    fx         :: Union{Nothing,AbstractArray{<:Number}} = nothing,
    epsilon    :: Union{Nothing,AbstractArray{<:Real}} = nothing,
    fdtype     :: Type{T1} = Val{:central},
    returntype :: Type{T2} = eltype(x))
```

This allocates either `fx` or `epsilon` if these are nothing and they are needed.
`fx` is the current call of `f(x)` and is required for forward-differencing
(otherwise is not necessary).

## Gradients

Gradients are either a vector->scalar map `f(x)`, or a scalar->vector map
`f(fx,x)` if `inplace=Val{true}` and `fx=f(x)` if `inplace=Val{false}`.

### Differencing Calls

```julia
# Cache-less
FiniteDiff.finite_difference_gradient(
    f,
    x,
    fdtype::Type{T1}=Val{:central},
    returntype::Type{T2}=eltype(x),
    inplace::Type{Val{T3}}=Val{true};
    [epsilon_factor])
FiniteDiff.finite_difference_gradient!(
    df,
    f,
    x,
    fdtype::Type{T1}=Val{:central},
    returntype::Type{T2}=eltype(df),
    inplace::Type{Val{T3}}=Val{true};
    [epsilon_factor])

# Cached
FiniteDiff.finite_difference_gradient!(
    df::AbstractArray{<:Number},
    f,
    x::AbstractArray{<:Number},
    cache::GradientCache;
    [epsilon_factor])
```

### Allocating Cache Constructor

```julia
FiniteDiff.GradientCache(
    df         :: Union{<:Number,AbstractArray{<:Number}},
    x          :: Union{<:Number, AbstractArray{<:Number}},
    fdtype     :: Type{T1} = Val{:central},
    returntype :: Type{T2} = eltype(df),
    inplace    :: Type{Val{T3}} = Val{true})
```

### Non-Allocating Cache Constructor

```julia
FiniteDiff.GradientCache(
    fx         :: Union{Nothing,<:Number,AbstractArray{<:Number}},
    c1         :: Union{Nothing,AbstractArray{<:Number}},
    c2         :: Union{Nothing,AbstractArray{<:Number}},
    c3         :: Union{Nothing,AbstractArray{<:Number}},
    fdtype     :: Type{T1} = Val{:central},
    returntype :: Type{T2} = eltype(fx),
    inplace    :: Type{Val{T3}} = Val{true})
```

Note that here `fx` is a cached function call of `f`. If you provide `fx`, then
`fx` will be used in the forward differencing method to skip a function call.
It is on you to make sure that you update `cache.fx` every time before
calling `FiniteDiff.finite_difference_gradient!`. If `fx` is an immutable, e.g. a scalar or 
a `StaticArray`, `cache.fx` should be updated using `@set` from [Setfield.jl](https://github.com/jw3126/Setfield.jl).
A good use of this is if you have a cache array for the output of `fx` already being used, you can make it alias
into the differencing algorithm here.

## Jacobians

Jacobians are for functions `f!(fx,x)` when using in-place `finite_difference_jacobian!`,
and `fx = f(x)` when using out-of-place `finite_difference_jacobian`. The out-of-place
jacobian will return a similar type as `jac_prototype` if it is not a `nothing`. For non-square
Jacobians, a cache which specifies the vector `fx` is required.

For sparse differentiation, pass a `colorvec` of matrix colors. `sparsity` should be a sparse
or structured matrix (`Tridiagonal`, `Banded`, etc. according to the ArrayInterfaceCore.jl specs)
to allow for decompression, otherwise the result will be the colorvec compressed Jacobian.

### Differencing Calls

```julia
# Cache-less
FiniteDiff.finite_difference_jacobian(
    f,
    x          :: AbstractArray{<:Number},
    fdtype     :: Type{T1}=Val{:central},
    returntype :: Type{T2}=eltype(x),
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    colorvec = 1:length(x),
    sparsity = nothing,
    jac_prototype = nothing)

finite_difference_jacobian!(J::AbstractMatrix,
    f,
    x::AbstractArray{<:Number},
    fdtype     :: Type{T1}=Val{:forward},
    returntype :: Type{T2}=eltype(x),
    f_in       :: Union{T2,Nothing}=nothing;
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    colorvec = 1:length(x),
    sparsity = ArrayInterfaceCore.has_sparsestruct(J) ? J : nothing)

# Cached
FiniteDiff.finite_difference_jacobian(
    f,
    x,
    cache::JacobianCache;
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    colorvec = cache.colorvec,
    sparsity = cache.sparsity,
    jac_prototype = nothing)

FiniteDiff.finite_difference_jacobian!(
    J::AbstractMatrix{<:Number},
    f,
    x::AbstractArray{<:Number},
    cache::JacobianCache;
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    colorvec = cache.colorvec,
    sparsity = cache.sparsity)
```

### Allocating Cache Constructor

```julia
FiniteDiff.JacobianCache(
              x,
              fdtype     :: Type{T1} = Val{:central},
              returntype :: Type{T2} = eltype(x),
              colorvec = 1:length(x)
              sparsity = nothing)
```

This assumes the Jacobian is square.

### Non-Allocating Cache Constructor

```julia
FiniteDiff.JacobianCache(
              x1 ,
              fx ,
              fx1,
              fdtype     :: Type{T1} = Val{:central},
              returntype :: Type{T2} = eltype(fx),
              colorvec = 1:length(x1),
              sparsity = nothing)
```

## Hessians

Hessians are for functions `f(x)` which return a scalar.

### Differencing Calls

```julia
#Cacheless
finite_difference_hessian(f, x::AbstractArray{<:Number},
    fdtype     :: Type{T1}=Val{:hcentral},
    inplace    :: Type{Val{T2}} = x isa StaticArray ? Val{true} : Val{false};
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep)

finite_difference_hessian!(H::AbstractMatrix,f,
    x::AbstractArray{<:Number},
    fdtype     :: Type{T1}=Val{:hcentral},
    inplace    :: Type{Val{T2}} = x isa StaticArray ? Val{true} : Val{false};
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep)

#Cached
finite_difference_hessian(
    f,x,
    cache::HessianCache{T,fdtype,inplace};
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep)

finite_difference_hessian!(H,f,x,
                           cache::HessianCache{T,fdtype,inplace};
                           relstep = default_relstep(fdtype, eltype(x)),
                           absstep = relstep)
```

### Allocating Cache Calls

```julia
HessianCache(x,fdtype::Type{T1}=Val{:hcentral},
                        inplace::Type{Val{T2}} = x isa StaticArray ? Val{true} : Val{false})
```

### Non-Allocating Cache Calls

```julia
HessianCache(xpp,xpm,xmp,xmm,
                      fdtype::Type{T1}=Val{:hcentral},
                      inplace::Type{Val{T2}} = x isa StaticArray ? Val{true} : Val{false})
```
