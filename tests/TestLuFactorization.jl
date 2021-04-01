include("LU-Factorization.jl")

import .LuFactorization

using LinearAlgebra

A = Matrix(rand(8, 8))

println("\n\n***********************")
println("A")
println("***********************")
display(A)


l, u = LuFactorization.LuGaussianElimination(A)

println("\n\n\n\nGuassian Elimination")
println("***********************")
println("L")
println("***********************")
display(l)

println("\n\n***********************")
println("U")
println("***********************")
display(u)

println("\n\n***********************")
println("A - LU")
println("***********************")

display(A - l * u)


l, u = LuFactorization.LuGaussTransforms(A)

println("\n\n\n\nGuassian Elimination Using Gauss Transforms")
println("***********************")
println("L")
println("***********************")
display(l)

println("\n\n***********************")
println("U")
println("***********************")
display(u)

println("\n\n***********************")
println("A - LU")
println("***********************")

display(A - l * u)


l, u, p = LuFactorization.LuGaussianEliminationWithPivots(A)

println("\n\n\n\nGuassian Elimination With Pivoting")
println("***********************")
println("L")
println("***********************")
display(l)

println("\n\n***********************")
println("U")
println("***********************")
display(u)

println("\n\n***********************")
println("A - LU")
println("***********************")

for i = 1:size(A)[2]
    global A = p[i] * A
end

display(A - l * u)


l, u, p = LuFactorization.LuGaussTransformsWithPivots(A)

println("\n\n\n\nGuassian Transforms With Pivot Transformations")
println("***********************")
println("L")
println("***********************")
display(l)

println("\n\n***********************")
println("U")
println("***********************")
display(u)

println("\n\n***********************")
println("A - LU")
println("***********************")

display(p * A - l * u)


l, u = LuFactorization.LuBordered(A)

println("\n\n\n\nBordered Algorithm")
println("***********************")
println("L")
println("***********************")
display(l)

println("\n\n***********************")
println("U")
println("***********************")
display(u)

println("\n\n***********************")
println("A - LU")
println("***********************")

display(A - l * u)
