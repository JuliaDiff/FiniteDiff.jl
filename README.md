# DiffEqDiffTools

[![Join the chat at https://gitter.im/JuliaDiffEq/Lobby](https://badges.gitter.im/JuliaDiffEq/Lobby.svg)](https://gitter.im/JuliaDiffEq/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[![Build Status](https://travis-ci.org/JuliaDiffEq/DiffEqDiffTools.jl.svg?branch=master)](https://travis-ci.org/JuliaDiffEq/DiffEqDiffTools.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/t3risc94d2jqipd6?svg=true)](https://ci.appveyor.com/project/ChrisRackauckas/diffeqdifftools-jl)
[![Coverage Status](https://coveralls.io/repos/ChrisRackauckas/DiffEqDiffTools.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/ChrisRackauckas/DiffEqDiffTools.jl?branch=master)
[![codecov.io](http://codecov.io/github/ChrisRackauckas/DiffEqDiffTools.jl/coverage.svg?branch=master)](http://codecov.io/github/ChrisRackauckas/DiffEqDiffTools.jl?branch=master)
[![DiffEqDiffTools](http://pkg.julialang.org/badges/DiffEqDiffTools_0.6.svg)](http://pkg.julialang.org/?pkg=DiffEqDiffTools)

DiffEqDiffTools.jl is a component package in the DifferentialEquations ecosystem.
It holds the common tools for taking derivatives, Jacobians, etc. and utilizing
the traits from the ParameterizedFunctions when possible for increasing the
speed of calculations. Users interested in using this functionality should check
out [DifferentialEquations.jl](https://github.com/JuliaDiffEq/DifferentialEquations.jl/blob/master/src/DifferentialEquations.jl).

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

## Scalar Derivatives

```julia
finite_difference_derivative(f, x::T, fdtype::Type{T1}=Val{:central},
    returntype::Type{T2}=eltype(x), f_x::Union{Nothing,T}=nothing)
```

## Multi-Point Derivatives

### Differencing Calls

```julia
# Cache-less but non-allocating if `fx` and `epsilon` are supplied
# fx must be f(x)
finite_difference_derivative(
    f,
    x          :: AbstractArray{<:Number},
    fdtype     :: Type{T1} = Val{:central},
    returntype :: Type{T2} = eltype(x),      # return type of f
    fx         :: Union{Nothing,AbstractArray{<:Number}} = nothing,
    epsilon    :: Union{Nothing,AbstractArray{<:Real}} = nothing)

finite_difference_derivative!(
    df         :: AbstractArray{<:Number},
    f,
    x          :: AbstractArray{<:Number},
    fdtype     :: Type{T1} = Val{:central},
    returntype :: Type{T2} = eltype(x),
    fx         :: Union{Nothing,AbstractArray{<:Number}} = nothing,
    epsilon    :: Union{Nothing,AbstractArray{<:Real}}   = nothing)

# Cached
finite_difference_derivative!(df::AbstractArray{<:Number}, f,
                              x::AbstractArray{<:Number},
                              cache::DerivativeCache{T1,T2,fdtype,returntype})
```

### Allocating and Non-Allocating Constructor

```julia
DerivativeCache(
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

### Differencing Calls

```julia
# Cache-less
finite_difference_gradient(f, x, fdtype::Type{T1}=Val{:central},
                           returntype::Type{T2}=eltype(x),
                           inplace::Type{Val{T3}}=Val{true})
finite_difference_gradient!(df, f, x, fdtype::Type{T1}=Val{:central},
                            returntype::Type{T2}=eltype(df),
                            inplace::Type{Val{T3}}=Val{true})

# Cached
finite_difference_gradient!(df::AbstractArray{<:Number}, f,
                            x::AbstractArray{<:Number},
                            cache::GradientCache)
```

### Allocating Cache Constructor

```julia
GradientCache(
    df         :: Union{<:Number,AbstractArray{<:Number}},
    x          :: Union{<:Number, AbstractArray{<:Number}},
    fdtype     :: Type{T1} = Val{:central},
    returntype :: Type{T2} = eltype(df),
    inplace    :: Type{Val{T3}} = Val{true})
```

### Non-Allocating Cache Constructor

```julia
GradientCache(
    c1         :: Union{Nothing,AbstractArray{<:Number}},
    c2         :: Union{Nothing,AbstractArray{<:Number}},
    fx         :: Union{Nothing,<:Number,AbstractArray{<:Number}} = nothing,
    fdtype     :: Type{T1} = Val{:central},
    returntype :: Type{T2} = eltype(df),
    inplace    :: Type{Val{T3}} = Val{true})
```

Note that here `fx` is a cached function call of `f`. If you provide `fx`, then
`fx` will be used in the forward differencing method to skip a function call.
It is on you to make sure that you update `cache.fx` every time before
calling `finite_difference_gradient!`. A good use of this is if you have a
cache array for the output of `fx` already being used, you can make it alias
into the differencing algorithm here.

## Jacobians

### Differencing Calls

```julia
# Cache-less
finite_difference_jacobian(f, x::AbstractArray{<:Number},
                           fdtype     :: Type{T1}=Val{:central},
                           returntype :: Type{T2}=eltype(x),
                           inplace    :: Type{Val{T3}}=Val{true})

# Cached
finite_difference_jacobian(f,x,cache::JacobianCache)
finite_difference_jacobian!(J::AbstractMatrix{<:Number},f,
                            x::AbstractArray{<:Number},cache::JacobianCache)
```

### Allocating Cache Constructor

```julia
JacobianCache(
              x,
              fdtype     :: Type{T1} = Val{:central},
              returntype :: Type{T2} = eltype(x),
              inplace    :: Type{Val{T3}} = Val{true})
```

This assumes the Jacobian is square.

### Non-Allocating Cache Constructor

```julia
JacobianCache(
              x1 ,
              fx ,
              fx1,
              fdtype     :: Type{T1} = Val{:central},
              returntype :: Type{T2} = eltype(fx),
              inplace    :: Type{Val{T3}} = Val{true})
```
