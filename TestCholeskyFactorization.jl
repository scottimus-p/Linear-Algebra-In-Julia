include("Cholesky-Factorization.jl")

import .CholeskyFactorization

using LinearAlgebra

A = Matrix(Hermitian(rand(8, 8) + I))

println("\n\n***********************")
println("A")
println("***********************")
display(A)


l = CholeskyFactorization.Cholesky(A)

println("\n\n\n\nCholesky Factorization")
println("***********************")
println("L")
println("***********************")
display(l)

println("\n\n***********************")
println("A - LL^H")
println("***********************")

display(A - l * l')
