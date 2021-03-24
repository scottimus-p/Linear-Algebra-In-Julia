# The code in this file performs a LU factorization of a real-valued matrix using
# several different algorithms

module LuFactorization

using LinearAlgebra

function LuGaussianElimination(A)
# this function performs an LU factorization on matrix A using Gaussian elimination

    M = copy(A)
    rows, cols = size(M)

    for i = 1:rows
        M[i + 1:rows, i] /= M[i, i]
        M[i + 1:rows, i + 1:cols] -= M[i + 1:rows, i:i] * M[i:i, i + 1:cols]
    end

    U = UpperTriangular(M)
    L = UnitLowerTriangular(M)

    return L, U
end


function LuGaussianEliminationWithPivots(A)
# this function performs an LU factorization on matrix A using Gaussian elimination

    M = copy(A)
    rows, cols = size(M)

    Pivots = Array{Matrix{Float64}}(undef, cols)

    for i = 1:rows
        # determine what row to pivot with and then get the matrix which will swap
        # rows. we choose the row that has the largest value in the first column
        # of the current submatrix. by choosing the largest absolute value, we limit
        # error from values close to zero
        PivotRow = FindIndexWithMaxAbs(M[i:rows, i:i]) + i - 1
        Pivots[i] = Matrix{Float64}(I, rows, cols)
        Pivots[i][i:rows, i:cols] = ElementaryPivotMatrix(PivotRow - i + 1, rows - i + 1, cols - i + 1)

        M[i:i, 1:cols], M[PivotRow:PivotRow, 1:cols] = M[PivotRow:PivotRow, 1:cols], M[i:i, 1:cols]

        M[i + 1:rows, i:i] /= M[i, i]
        M[i + 1:rows, i + 1:cols] -= M[i + 1:rows, i:i] * M[i:i, i + 1:cols]
    end

    U = UpperTriangular(M)
    L = UnitLowerTriangular(M)

    return L, U, Pivots
end


function LuGaussTransforms(A)
# this function performs an LU factorization on matrix A using Gauss transforms. it is
# essentially the same as the LuGaussianElimination function except the Gaussian
# elimination is applied via a series of matrix transformations

    M = copy(A)
    rows, cols = size(M)

    # this will be an array of the Gauss transform matrices that will be calculated
    Gausses = Array{Matrix{Float64}}(undef, cols)

    for i = 1:(rows - 1)
        Gausses[i] = GaussTransform(M, i)
        M = Gausses[i] * M
    end

    U = M

    # explicitly form the L in the LU factorization by applying the
    # inverse of the Gauss transforms. this is because:
    # L(n) * L(n-1) * ... * L(1) * A = U
    # implies
    # A = inv(L(1)) * inv(L(2)) * ... inv(L(n)) * U
    L = Matrix{Float64}(I, rows, cols)
    for i = (rows - 1):-1:1
        L = inv(Gausses[i]) * L
    end

    return L, U
end


function GaussTransform(A, k)
# this function returns a Gauss transform matrix which, when applied
# to matrix A, takes a multiple of the row indexed with k and adds them
# to the other rows

    rows, cols = size(A)

    L = Matrix{Float64}(I, rows, cols)
    L[(k + 1):rows, k] = A[(k + 1):rows, k] / -A[k, k]

    return L
end


function LuGaussTransformsWithPivots(A)
# this function performs the same process as LuGaussTransforms except it
# implements row pivoting to reduce error introduced due to floating
# point arithmetic error

    M = copy(A)
    rows, cols = size(M)

    # this will be an array of the Gauss transform matrices that will be calculated
    Gausses = Array{Matrix{Float64}}(undef, cols)
    Pivots = Array{Matrix{Float64}}(undef, cols)

    for i = 1:rows
        # determine what row to pivot with and then get the matrix which will swap
        # rows. we choose the row that has the largest value in the first column
        # of the current submatrix. by choosing the largest absolute value, we limit
        # error from values close to zero
        PivotRow = FindIndexWithMaxAbs(M[i:rows, i:i])
        Pivots[i] = Matrix{Float64}(I, rows, cols)
        Pivots[i][i:rows, i:cols] = ElementaryPivotMatrix(PivotRow, rows - i + 1, cols - i + 1)

        # perform the pivot by applying the pivot matrix, but no need to actually use the matrix
        # transform, we can just swap instead because that's what we know the transformation would
        # do
        M[i:i, 1:cols], M[PivotRow+i-1:PivotRow+i-1, 1:cols] = M[PivotRow+i-1:PivotRow+i-1, 1:cols], M[i:i, 1:cols]

        # now proceed as usual with our pivoted matrix by getting the Gauss
        # transform and using it to perform Gaussian elimination
        Gausses[i] = GaussTransform(M, i)
        M = Gausses[i] * M
    end

    U = M

    # explicitly form the L in the LU factorization by applying the
    # inverse of the Gauss transforms. this is because:
    # L(n) * P(n) * L(n-1) * P(n-1) * ... * L(1) * P(1) * A = U
    # implies
    # A = P(1) * inv(L(1)) * P(2) * inv(L(2)) * ... * P(n) * inv(L(n)) * U
    L = Matrix{Float64}(I, rows, cols)
    P = Matrix{Float64}(I, rows, cols)
    for i = rows:-1:1
        L = inv(Gausses[i]) * L
        P = Pivots[i] * P
    end

    return L, U, P
end


function FindIndexWithMaxAbs(vector)
    rows = size(vector)[1]

    maxValue = -Inf
    maxIndex = 0

    for i = 1:rows
        if abs(vector[i]) > maxValue
            maxValue = abs(vector[i])
            maxIndex = i
        end
    end

    return maxIndex
end


function ElementaryPivotMatrix(p, rows, cols)
    P = zeros(rows, cols)

    P[1, p] = 1

    for i = 2:(p - 1)
        P[i, i] = 1
    end

    P[p, 1] = 1

    for i = (p + 1):rows
        P[i, i] = 1
    end

    return P
end


function LuBordered(A)
# this function performs an LU factorization on the matrix A using the Bordered
# LU factorization algorithm

    M = copy(A)
    rows, cols = size(M)

    for i = 1:rows
        L = UnitLowerTriangular(M[1:i-1, 1:i-1])
        U = UpperTriangular(M[1:i-1, 1:i-1])

        M[1:i-1, i:i] = inv(L) * M[1:i-1, i:i]
        M[i:i, 1:i-1] = M[i:i, 1:i-1] * inv(U)
        M[i:i, i:i] = M[i:i, i:i] - M[i:i, 1:i-1] * M[1:i-1, i:i]
    end

    L = UnitLowerTriangular(M)
    U = UpperTriangular(M)
    return L, U
end


function totalPermute(p)
    Ps = Matrix{Float64}(I, size(p)[1], size(p)[1])
    for i = 1:size(p)[1]
        Ps = P[i] * Ps
    end

    return Ps
end

end # end of module declaration
