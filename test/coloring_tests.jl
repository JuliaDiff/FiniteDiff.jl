using DiffEqDiffTools, LinearAlgebra, SparseArrays, Test

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
DiffEqDiffTools.finite_difference_jacobian!(_J2,f,rand(30),color=repeat(1:3,10))
@test fcalls == 4
@test _J2 ≈ _J

_J2 = similar(_J)
fcalls = 0
DiffEqDiffTools.finite_difference_jacobian!(_J2,f,rand(30),Val{:central},color=repeat(1:3,10))
@test fcalls == 6
@test _J2 ≈ _J

_J2 = similar(_J)
fcalls = 0
DiffEqDiffTools.finite_difference_jacobian!(_J2,f,rand(30),Val{:complex},color=repeat(1:3,10)
  )
@test fcalls == 3
@test _J2 ≈ _J
