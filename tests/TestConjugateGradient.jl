include("../Conjugate-Gradient.jl")

using LinearAlgebra

function CreatePoissonMatrix(N)

    npoints = N * N

    A = zeros(npoints, npoints) # this is inefficient, we should use a sparse matrix

    A[1:1+npoints:end] .= 4

    A[2:1+npoints:end] .= -1
    A[1+npoints:1+npoints:end] .= -1

    A[2+(1+npoints)*(N-1):(1+npoints)*N:end] .= 0
    A[1+npoints+(1+npoints)*(N-1):(1+npoints)*N:end] .= 0

    A[(N+1):1+npoints:end-npoints*N] .= -1
    A[1+N*npoints:1+npoints:end] .= -1

    return A
end

function PlaceFinB(N, F)

    b = zeros(N*N, 1)

    ii = 1
    for i=2:N+1
        for j=2:N+1
            b[ii] = F[i, j]
            ii += 1
        end
    end

    h = 1 / (N+1)
    b = h*h*b
end


N = 50
h = 1 / (N + 1)

α = 2
β = 3

F = Matrix{Float64}(undef, N + 2, N + 2)
x = h * (0:N+1)
y = h * (0:N+1)

for i = 1:N+2
    for j = 1:N+2
        F[i, j] = (α^2 + β^2) * (π^2 * sin(α * π * x[i]) * sin(β * π * y[j]))
    end
end

A = CreatePoissonMatrix(N)
b = PlaceFinB(N, F)

x, iters = ConjugateGradient.CG(A, b)

error = norm(x - A \ b)

println("Iterations: ", iters)
println("Error: ", error)
