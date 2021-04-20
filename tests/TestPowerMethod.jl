include("../Power-Method.jl")

using Random
using LinearAlgebra

rng = MersenneTwister(12345)

# generate a random positive definite matrix
size = 80
A = Matrix(rand(rng, size, size))
A = (A * A') + size * I(size)

λ, v, n = PowerMethod.PowMethod(A)

v̂ = A * v

println("************************")
println("Power Method")
println("************************")
println("λ = $λ")
println("\niterations: $n\n\nerror: $(norm(v̂ / λ - v))\n")


λ, v, n = PowerMethod.InversePowerMethod(A, 1E-5, 300)

v̂ = A * v

println("\n\n************************")
println("Inverse Power Method")
println("************************")
println("λ = $λ")
println("\niterations: $n\n\nerror: $(norm(v̂ / λ - v))\n")



λ, v, n = PowerMethod.ShiftedInversePowerMethod(A, 80, 1E-5, 300)

v̂ = A * v

println("\n\n************************")
println("Shifted Inverse Power Method")
println("************************")
println("λ = $λ")
println("\niterations: $n\n\nerror: $(norm(v̂ / λ - v))\n")



λ, v, n = PowerMethod.RayleighQuotientIteration(A, 1E-5, 300)

v̂ = A * v

println("\n\n************************")
println("Rayleigh Quotient Iteration")
println("************************")
println("λ = $λ")
#println("v = ")
#display(v)
println("\niterations: $n\n\nerror: $(norm(v̂ / λ - v))\n")

display(eigen(A).values)
