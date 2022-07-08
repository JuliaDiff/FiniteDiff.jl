using FiniteDiff, LinearAlgebra, SparseArrays, Test, LinearAlgebra,
  BlockBandedMatrices, ArrayInterfaceCore, BandedMatrices,
  ArrayInterfaceBlockBandedMatrices

fcalls = 0
function f(dx, x)
  global fcalls += 1
  for i in 2:length(x)-1
    dx[i] = x[i-1] - 2x[i] + x[i+1]
  end
  dx[1] = -2x[1] + x[2]
  dx[end] = x[end-1] - 2x[end]
  nothing
end
function oopf(x)
  global fcalls += 1
  vcat([-2x[1] + x[2]], [x[i-1] - 2x[i] + x[i+1] for i in 2:length(x)-1], [x[end-1] - 2x[end]])
end

function second_derivative_stencil(N)
  A = zeros(N, N)
  for i in 1:N, j in 1:N
    (j - i == -1 || j - i == 1) && (A[i, j] = 1)
    j - i == 0 && (A[i, j] = -2)
  end
  A
end

J = FiniteDiff.finite_difference_jacobian(oopf, rand(30))
@test J ≈ second_derivative_stencil(30)
_J = sparse(J)
@test fcalls == 31

_J2 = similar(_J)
fcalls = 0
FiniteDiff.finite_difference_jacobian!(_J2, f, rand(30), colorvec=repeat(1:3, 10))
@test fcalls == 4
@test _J2 ≈ _J

_J2 = similar(_J)
fcalls = 0
FiniteDiff.finite_difference_jacobian!(_J2, f, rand(30), Val{:central}, colorvec=repeat(1:3, 10))
@test fcalls == 6
@test _J2 ≈ _J

_J2 = similar(_J)
fcalls = 0
FiniteDiff.finite_difference_jacobian!(_J2, f, rand(30), Val{:complex}, colorvec=repeat(1:3, 10))
@test fcalls == 3
@test _J2 ≈ _J

_J2 = similar(_J)
_denseJ2 = collect(_J2)
fcalls = 0
FiniteDiff.finite_difference_jacobian!(_denseJ2, f, rand(30), colorvec=repeat(1:3, 10), sparsity=_J2)
@test fcalls == 4
@test sparse(_denseJ2) ≈ _J

_J2 = similar(_J)
_denseJ2 = collect(_J2)
fcalls = 0
FiniteDiff.finite_difference_jacobian!(_denseJ2, f, rand(30), Val{:central}, colorvec=repeat(1:3, 10), sparsity=_J2)
@test fcalls == 6
@test sparse(_denseJ2) ≈ _J

_J2 = similar(_J)
_denseJ2 = collect(_J2)
fcalls = 0
FiniteDiff.finite_difference_jacobian!(_denseJ2, f, rand(30), Val{:complex}, colorvec=repeat(1:3, 10), sparsity=_J2)
@test fcalls == 3
@test sparse(_denseJ2) ≈ _J

_J2 = similar(Tridiagonal(_J))
fcalls = 0
FiniteDiff.finite_difference_jacobian!(_J2, f, rand(30), colorvec=repeat(1:3, 10))
@test fcalls == 4
@test _J2 ≈ _J

_J2 = similar(_J2)
fcalls = 0
FiniteDiff.finite_difference_jacobian!(_J2, f, rand(30), Val{:central}, colorvec=repeat(1:3, 10))
@test fcalls == 6
@test _J2 ≈ _J

_J2 = similar(_J2)
fcalls = 0
FiniteDiff.finite_difference_jacobian!(_J2, f, rand(30), Val{:complex}, colorvec=repeat(1:3, 10))
@test fcalls == 3
@test _J2 ≈ _J

_Jb = BandedMatrices.BandedMatrix(similar(_J2), (1, 1))
FiniteDiff.finite_difference_jacobian!(_Jb, f, rand(30), colorvec=repeat(1:3, 10))
@test _Jb ≈ _J

_Jtri = Tridiagonal(similar(_J2))
FiniteDiff.finite_difference_jacobian!(_Jtri, f, rand(30), colorvec=repeat(1:3, 10))
@test _Jtri ≈ _J

#https://github.com/JuliaDiff/FiniteDiff.jl/issues/67#issuecomment-516871956
function f(out, x)
  x = reshape(x, 100, 100)
  out = reshape(out, 100, 100)
  for i in 1:100
    for j in 1:100
      out[i, j] = x[i, j] + x[max(i - 1, 1), j] + x[min(i + 1, size(x, 1)), j] + x[i, max(j - 1, 1)] + x[i, min(j + 1, size(x, 2))]
    end
  end
  return vec(out)
end
x = rand(10000)
Jbbb = BandedBlockBandedMatrix(Ones(10000, 10000), fill(100, 100), fill(100, 100), (1, 1), (1, 1))
Jsparse = sparse(Jbbb)
colorsbbb = ArrayInterfaceCore.matrix_colors(Jbbb)
FiniteDiff.finite_difference_jacobian!(Jbbb, f, x, colorvec=colorsbbb)
FiniteDiff.finite_difference_jacobian!(Jsparse, f, x, colorvec=colorsbbb)
@test Jbbb ≈ Jsparse
Jbb = BlockBandedMatrix(similar(Jsparse), fill(100, 100), fill(100, 100), (1, 1));
colorsbb = ArrayInterfaceCore.matrix_colors(Jbb)
FiniteDiff.finite_difference_jacobian!(Jbb, f, x, colorvec=colorsbb)
@test Jbb ≈ Jsparse


# Non-square Jacobian test.
# The Jacobian of f_nonsquare! has size (n, 2*n).
function f_nonsquare!(y, x)
  global fcalls += 1
  @assert length(x) == 2 * length(y)
  n = length(x) ÷ 2
  x1 = @view x[1:n]
  x2 = @view x[n+1:end]

  @. y = (x1 .- 3) .^ 2 .+ x1 .* x2 .+ (x2 .+ 4) .^ 2 .- 3
  return nothing
end

n = 4
x0 = vcat(ones(n) .* (1:n) .+ 0.5, ones(n) .* (1:n) .+ 1.5)
y0 = zeros(n)
rows = vcat([i for i in 1:n], [i for i in 1:n])
cols = vcat([i for i in 1:n], [i + n for i in 1:n])
sparsity = sparse(rows, cols, ones(length(rows)))
colorvec = vcat(fill(1, n), fill(2, n))

J_nonsquare1 = zeros(size(sparsity))
FiniteDiff.finite_difference_jacobian!(J_nonsquare1, f_nonsquare!, x0)

J_nonsquare2 = similar(sparsity)
for method in [Val(:forward), Val(:central), Val(:complex)]
  cache = FiniteDiff.JacobianCache(copy(x0), copy(y0), copy(y0), method; sparsity, colorvec)
  global fcalls = 0
  FiniteDiff.finite_difference_jacobian!(J_nonsquare2, f_nonsquare!, x0, cache)
  if method == Val(:central)
    @test fcalls == 2 * maximum(colorvec)
  elseif method == Val(:complex)
    @test fcalls == maximum(colorvec)
  else
    @test fcalls == maximum(colorvec) + 1
  end
  @test isapprox(J_nonsquare2, J_nonsquare1; rtol=1e-6)
end

## Non-sparse prototype 
# Test structralnz for dense matrix 
A = [[1 1; 0 1], [1 1 1], [1.0 1.0; 1.0 1.0; 1.0 1.0], [true true; true true]]
for a in A
  rows_index, cols_index = FiniteDiff._findstructralnz(a)
  rows_index2, cols_index2 = ArrayInterfaceCore.findstructralnz(sparse(a))
  @test (rows_index, cols_index) == (rows_index2, cols_index2)
end

# Square Jacobian
function _f(dx, x)
  dx .= [x[1]^2 + x[2]^2, x[1] + x[2]]
end
J = zeros(2, 2)
θ = [5.0, 3.0]
y0 = [0.0, 0.0]
cache = FiniteDiff.JacobianCache(copy(θ), copy(y0), copy(y0), Val(:forward); sparsity=[1 1; 1 1])
FiniteDiff.finite_difference_jacobian!(J, _f, θ, cache)
@test J ≈ [10 6; 1 1]

function _f2(dx, x)
  dx .= [x[1]^2 + x[2]^2, x[1]]
end
J = zeros(2, 2)
θ = [-3.0, 2.0]
y0 = [0.0, 0.0]
cache = FiniteDiff.JacobianCache(copy(θ), copy(y0), copy(y0), Val(:forward); sparsity=[1 1; 1 0])
FiniteDiff.finite_difference_jacobian!(J, _f2, θ, cache)
@test J ≈ [-6 4; 1 0]

# Rectangular Jacobian 
function _f3(dx, x)
  dx .= [x[1]^2 + x[2]^2 - x[1]]
end
J = zeros(1, 2)
θ = [-3.0, 2.0]
y0 = [0.0, 0.0]
cache = FiniteDiff.JacobianCache(copy(θ), copy(y0), copy(y0), Val(:forward); sparsity=[1 1])
FiniteDiff.finite_difference_jacobian!(J, _f3, θ, cache)
@test J ≈ [-7 4]

function _f4(dx, x)
  dx .= [x[1]^2 + x[2]^2 - x[1]; x[1]*x[2]; x[1]*x[3]; x[1]]
end
J = zeros(4, 3)
θ = [-3.0, 2.0, 13.3]
y0 = [0.0, 0.0, 0.0, 0.0]
cache = FiniteDiff.JacobianCache(copy(θ), copy(y0), copy(y0), Val(:forward); sparsity=[1 1 0; 1 1 0; 1 0 1; 1 0 0])
FiniteDiff.finite_difference_jacobian!(J, _f4, θ, cache)
@test J ≈ [-7.0 4.0 0; 2.0 -3.0 0.0; 13.3 0.0 -3.0; 1.0 0.0 0.0]

function _f5(dx, x)
    dx .= [x[1]^2 + x[2]^2]
end
J = zeros(1, 2)
θ = [5.0, 3.0]
y0 = [0.0]
cache = FiniteDiff.JacobianCache(copy(θ), copy(y0), copy(y0), Val(:forward); sparsity = [1 1])
FiniteDiff.finite_difference_jacobian!(J, _f5, θ, cache)
@test J ≈ [10.0 6.0]