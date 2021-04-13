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


function Jacobi(Func::Function, NumberOfMeshPoints, iterations)
# performs a Jacobi iteration to solve the system of equations
# Func must be have two parameters

    h = 1 / (NumberOfMeshPoints + 1)

    x = h * collect(1:NumberOfMeshPoints + 2)
    y = h * collect(1:NumberOfMeshPoints + 2)

    F = zeros(NumberOfMeshPoints + 2, NumberOfMeshPoints + 2)
    U = zeros(NumberOfMeshPoints + 2, NumberOfMeshPoints + 2)

    for i = 2:(NumberOfMeshPoints + 2)
        for j = 2:(NumberOfMeshPoints + 2)
            F[i, j] = Func(x[i], y[j])
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

    return U
end

NumberOfIterations = 500


function PoissonEquation(x, y, α, β)
    return (α^2 + β^2) * π^2 * sin(α * π * x) * sin(β * π * y)
end

myFunc(α, β) = function(x, y)
    return PoissonEquation(x, y, α, β)
end

U = Jacobi(myFunc(2, 3), 50, 500)

writedlm( "U.csv",  U, ',')
