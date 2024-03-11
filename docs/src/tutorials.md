# Tutorials

## Fast Dense Jacobians

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

## Fast Sparse Jacobians

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

## Fast Tridiagonal Jacobians

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

## Fast Block Banded Matrices

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