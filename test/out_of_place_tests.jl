using DiffEqDiffTools, LinearAlgebra, SparseArrays, Test, StaticArrays

function f(x)
  xm1 = [0;x[1:end-1]]
  xp1 = [x[2:end];0]
  xm1 - 2x + xp1
end

function second_derivative_stencil(N)
  A = zeros(N,N)
  for i in 1:N, j in 1:N
      (j-i==-1 || j-i==1) && (A[i,j]=1)
      j-i==0 && (A[i,j]=-2)
  end
  A
end

x = @SVector ones(30)
J = DiffEqDiffTools.finite_difference_jacobian(f,x, Val{:forward}, eltype(x), Val{false})
@test J ≈ second_derivative_stencil(30)
_J = sparse(J)
DiffEqDiffTools.finite_difference_jacobian!(_J, f,x, Val{:forward}, eltype(x), Val{false},
    colorvec=repeat(1:3,10))
@test _J ≈ second_derivative_stencil(30)

J = DiffEqDiffTools.finite_difference_jacobian(f,x, Val{:central}, eltype(x), Val{false})
@test J ≈ second_derivative_stencil(30)
_J = sparse(J)
DiffEqDiffTools.finite_difference_jacobian!(_J, f,x, Val{:central}, eltype(x), Val{false},
    colorvec=repeat(1:3,10))
@test _J ≈ second_derivative_stencil(30)

J = DiffEqDiffTools.finite_difference_jacobian(f,x, Val{:complex}, eltype(x), Val{false})
@test J ≈ second_derivative_stencil(30)
_J = sparse(J)
DiffEqDiffTools.finite_difference_jacobian!(_J, f,x, Val{:complex}, eltype(x), Val{false},
    colorvec=repeat(1:3,10))
@test _J ≈ second_derivative_stencil(30)
