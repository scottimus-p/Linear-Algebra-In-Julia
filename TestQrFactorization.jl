include("QR-Factorization.jl")

import .QrFactorization

using LinearAlgebra

A = Matrix(rand(8, 6))

println("\n\n***********************")
println("A")
println("***********************")
display(A)


q, r = QrFactorization.GramSchmidt(A)

println("\n\n\n\nGram Schmidt Orthogonalization")
println("***********************")
println("Q")
println("***********************")
display(q)

println("\n\n***********************")
println("R")
println("***********************")
display(r)

println("\n\n***********************")
println("A - QR")
println("***********************")

display(A - q * r)


q, r = QrFactorization.ModifiedGramSchmidt(A)

println("\n\n\n\nModified Gram Schmidt Orthogonalization")
println("***********************")
println("Q")
println("***********************")
display(q)

println("\n\n***********************")
println("R")
println("***********************")
display(r)

println("\n\n***********************")
println("A - QR")
println("***********************")

display(A - q * r)


q, r = QrFactorization.Householder(A)

println("\n\n\n\nHouseholder Transforms")
println("***********************")
println("Q")
println("***********************")
display(q)

println("\n\n***********************")
println("R")
println("***********************")
display(r)

println("\n\n***********************")
println("A - QR")
println("***********************")

display(A - q * r)
