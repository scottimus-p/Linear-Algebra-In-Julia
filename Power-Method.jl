module PowerMethod

using LinearAlgebra

"""
Power Method

uses the Power Method to determine the largest eigenvalue of matrix A
and its associated eigenvector
"""
function PowMethod(A, tolerance = 1E-10, maxiters = 50)

    rows, cols = size(A)

    if (rows != cols)
        throw(error("PowMethod: matrix A must be square"))
    end

    # Make initial vector a unit-length vector
    v = zeros(rows, 1)
    v[1] = 1

    # initialize the variable so we'll get through at least one iteration of the while below
    change = tolerance + 1
    iters = 0

    while (iters < maxiters && change > tolerance)
        v_old = v

        v = A * v
        v = v / norm(v)

        change = norm(v - v_old)
        iters += 1
    end

    λ = v' * A * v

    return λ, v, iters
end



"""
Inverse Power Method

uses the Inverse Power Method to determine the smallest eigenvalue of matrix A
and its associated eigenvector
"""
function InversePowerMethod(A::Matrix{Float64}, tolerance = 1E-10, maxiters = 50)

    rows, cols = size(A)

    if (rows != cols)
        throw(error("InversePowerMethod: matrix A must be square"))
    end

    # Make initial vector a unit-length vector
    v = zeros(rows, 1)
    v[1] = 1

    # initialize the variable so we'll get through at least one iteration of the while below
    change = tolerance + 1
    iters = 0

    F = lu(A)

    while (iters < maxiters && change > tolerance)
        v_old = v

        v = F.U \ (F.L \ v)
        v = v / norm(v)

        change = norm(v - v_old)
        iters += 1
    end

    λ = v' * A * v

    return λ, v, iters

end


"""
Shifted Inverse Power Method

uses the Shifted Inverse Power Method to determine the eigenvalue of matrix A closest to ρ
and its associated eigenvector
"""
function ShiftedInversePowerMethod(A::Matrix{Float64}, ρ, tolerance = 1E-10, maxiters = 50)

    rows, cols = size(A)

    if (rows != cols)
        throw(error("ShiftedInversePowerMethod: matrix A must be square"))
    end

    # Make initial vector a unit-length vector
    v = zeros(rows, 1)
    v[1] = 1

    # initialize the variable so we'll get through at least one iteration of the while below
    change = tolerance + 1
    iters = 0

    F = lu(A - ρ * I(rows))

    while (iters < maxiters && change > tolerance)
        v_old = v

        v = F.U \ (F.L \ v)
        v = v / norm(v)

        change = norm(v - v_old)
        iters += 1
    end

    λ = v' * A * v

    return λ, v, iters

end



"""
Rayleigh Quotient Iteration

uses the Rayleigh Quotient Iteration to determine an eigenvalue of matrix A
and its associated eigenvector
"""
function RayleighQuotientIteration(A::Matrix{Float64}, tolerance = 1E-10, maxiters = 50)

    rows, cols = size(A)

    if (rows != cols)
        throw(error("RayleighQuotientIteration: matrix A must be square"))
    end

    # Make initial vector a unit-length vector
    v = zeros(rows, 1)
    v[1] = 1

    # initialize the variable so we'll get through at least one iteration of the while below
    change = tolerance + 1
    iters = 0

    while (iters < maxiters && change > tolerance)
        v_old = v

        ρ =  v' * A * v

        v = (A - ρ .* I(rows)) \ v
        v = v / norm(v)

        change = norm(v - v_old)
        iters += 1
    end

    λ = v' * A * v

    return λ, v, iters

end

end
