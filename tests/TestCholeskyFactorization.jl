include("../Cholesky-Factorization.jl")

import .CholeskyFactorization

using LinearAlgebra

X = Matrix(Hermitian(rand(8, 8) + I))
A = X' * X

println("\n\n***********************")
println("A")
println("***********************")
display(A)


l = CholeskyFactorization.RightLookingCholesky(A)

println("\n\n\n\nCholesky Factorization")
println("***********************")
println("L")
println("***********************")
display(l)

println("\n\n***********************")
println("A - LLᵀ")
println("***********************")

display(A - l * l')


l = CholeskyFactorization.BorderedCholesky(A)

println("\n\n\n\nBordered Cholesky Factorization")
println("***********************")
println("L")
println("***********************")
display(l)

println("\n\n***********************")
println("A - LLᵀ")
println("***********************")

display(A - l * l')
