using DiffEqDiffTools, LinearAlgebra, SparseArrays, Test, LinearAlgebra, BlockBandedMatrices, ArrayInterface, BandedMatrices

fcalls = 0
function f(dx,x)
  global fcalls += 1
  for i in 2:length(x)-1
    dx[i] = x[i-1] - 2x[i] + x[i+1]
  end
  dx[1] = -2x[1] + x[2]
  dx[end] = x[end-1] - 2x[end]
  nothing
end

function second_derivative_stencil(N)
  A = zeros(N,N)
  for i in 1:N, j in 1:N
      (j-i==-1 || j-i==1) && (A[i,j]=1)
      j-i==0 && (A[i,j]=-2)
  end
  A
end

J = DiffEqDiffTools.finite_difference_jacobian(f,rand(30))
@test J ≈ second_derivative_stencil(30)
_J = sparse(J)
@test fcalls == 31

_J2 = similar(_J)
fcalls = 0
DiffEqDiffTools.finite_difference_jacobian!(_J2,f,rand(30),colorvec=repeat(1:3,10))
@test fcalls == 4
@test _J2 ≈ _J

_J2 = similar(_J)
fcalls = 0
DiffEqDiffTools.finite_difference_jacobian!(_J2,f,rand(30),Val{:central},colorvec=repeat(1:3,10))
@test fcalls == 6
@test _J2 ≈ _J

_J2 = similar(_J)
fcalls = 0
DiffEqDiffTools.finite_difference_jacobian!(_J2,f,rand(30),Val{:complex},colorvec=repeat(1:3,10))
@test fcalls == 3
@test _J2 ≈ _J

_J2 = similar(_J)
_denseJ2 = collect(_J2)
fcalls = 0
DiffEqDiffTools.finite_difference_jacobian!(_denseJ2,f,rand(30),colorvec=repeat(1:3,10),sparsity=_J2)
@test fcalls == 4
@test sparse(_denseJ2) ≈ _J

_J2 = similar(_J)
_denseJ2 = collect(_J2)
fcalls = 0
DiffEqDiffTools.finite_difference_jacobian!(_denseJ2,f,rand(30),Val{:central},colorvec=repeat(1:3,10),sparsity=_J2)
@test fcalls == 6
@test sparse(_denseJ2) ≈ _J

_J2 = similar(_J)
_denseJ2 = collect(_J2)
fcalls = 0
DiffEqDiffTools.finite_difference_jacobian!(_denseJ2,f,rand(30),Val{:complex},colorvec=repeat(1:3,10),sparsity=_J2)
@test fcalls == 3
@test sparse(_denseJ2) ≈ _J

_J2 = similar(Tridiagonal(_J))
fcalls = 0
DiffEqDiffTools.finite_difference_jacobian!(_J2,f,rand(30),colorvec=repeat(1:3,10))
@test fcalls == 4
@test _J2 ≈ _J

_J2 = similar(_J2)
fcalls = 0
DiffEqDiffTools.finite_difference_jacobian!(_J2,f,rand(30),Val{:central},colorvec=repeat(1:3,10))
@test fcalls == 6
@test _J2 ≈ _J

_J2 = similar(_J2)
fcalls = 0
DiffEqDiffTools.finite_difference_jacobian!(_J2,f,rand(30),Val{:complex},colorvec=repeat(1:3,10))
@test fcalls == 3
@test _J2 ≈ _J

_Jb = BandedMatrices.BandedMatrix(similar(_J2),(1,1))
DiffEqDiffTools.finite_difference_jacobian!(_Jb, f, rand(30), colorvec=colorvec=repeat(1:3,10))
@test _Jb ≈ _J

_Jtri = Tridiagonal(similar(_J2))
DiffEqDiffTools.finite_difference_jacobian!(_Jtri, f, rand(30), colorvec=colorvec=repeat(1:3,10))
@test _Jtri ≈ _J

#https://github.com/JuliaDiffEq/DiffEqDiffTools.jl/issues/67#issuecomment-516871956
function f(out, x)
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
Jbbb = BandedBlockBandedMatrix(Ones(10000, 10000), (fill(100, 100), fill(100, 100)), (1, 1), (1, 1))
Jsparse = sparse(Jbbb)
colorsbbb = ArrayInterface.matrix_colors(Jbbb)
DiffEqDiffTools.finite_difference_jacobian!(Jbbb, f, x, colorvec=colorsbbb)
DiffEqDiffTools.finite_difference_jacobian!(Jsparse, f, x, colorvec=colorsbbb)
@test Jbbb ≈ Jsparse
Jbb = BlockBandedMatrix(similar(Jsparse),(fill(100, 100), fill(100, 100)),(1,1));
colorsbb = ArrayInterface.matrix_colors(Jbb)
DiffEqDiffTools.finite_difference_jacobian!(Jbb, f, x, colorvec=colorsbb)
@test Jbb ≈ Jsparse