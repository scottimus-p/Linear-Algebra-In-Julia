# This file currently performs a Jacobi iteration algorithm for a Poisson's
# equation on a unit square domain , i.e. 0 ≤ x,y ≤ 1

# todos:
# 1) generalize the implementation
#    a) any formula, not just Poisson's equation
#    b) arbitrary number of mesh points and iterations
#    c) arbitrary domain
# 2) add Successive Over-Relaxation implementation
# 3) turn this into a module rather than a script

using LinearAlgebra
using DelimitedFiles

NumberOfIterations = 500

# Set parameters for Poisson's equation
alpha = 2
beta = 3

NumberOfMeshPoints = 50

h = 1 / (NumberOfMeshPoints + 1)

x = h * collect(1:NumberOfMeshPoints + 2)
y = h * collect(1:NumberOfMeshPoints + 2)

F = zeros(NumberOfMeshPoints + 2, NumberOfMeshPoints + 2)
U = zeros(NumberOfMeshPoints + 2, NumberOfMeshPoints + 2)

for i = 2:(NumberOfMeshPoints + 2)
    for j = 2:(NumberOfMeshPoints + 2)
        F[i, j] = (alpha^2 + beta^2) * pi^2 * sin(alpha * pi * x[i]) * sin(beta * pi * y[j])
    end
end

for i = 1:NumberOfIterations
    U_old = U
    for j = 2:(NumberOfMeshPoints + 1)
        for k = 2:(NumberOfMeshPoints + 1)
            U[j, k] = (h^2 * F[j, k] + U_old[j, k - 1] + U_old[j - 1, k] + U_old[j + 1, k] + U_old[j, k + 1]) / 4
        end
    end
end

writedlm( "U.csv",  U, ',')
