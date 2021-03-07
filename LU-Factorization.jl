# The code in this file performs a LU factorization of a real-valued matrix.
#
# At the end of the file is code that runs the factorizations and checks the accuracy

using LinearAlgebra


function LuGaussianElimination(A)

    M = copy(A)
    rows, cols = size(M)

    for i = 1:(rows - 1)
        M[i + 1:rows, i] /= M[i, i]
        M[i + 1:rows, i + 1:cols] -= M[i + 1:rows, i] * M[i, i + 1:cols]'
    end

    u = UpperTriangular(M)

    # the lower triangular matrix is the lower part of our transformed matrix with
    # 1's along the diagonal
    l = Matrix{Float64}(I, rows, cols)
    l[2:rows, 1:cols - 1] += LowerTriangular(M[2:rows, 1:cols - 1])

    return l, u
end


function LuGaussTransforms(A)

    M = copy(A)
    rows, cols = size(M)

    # this will be an array of the Gauss transform matrices that will be calculated
    Gausses = Array{Matrix{Float64}}(undef, cols)

    for i = 1:(rows - 1)
        Gausses[i] = GaussTransform(A, i)
        M = Gausses[i] * M
    end

    U = M

    # explicitly form the L in the LU factorization by applying the
    # inverse of the Gauss transforms. this is because:
    # L(n) * L(n-1) * ... * L(1) * A = U
    # implies
    # A = inv(L(0)) * inv(L(1)) * ... inv(L(n)) * U
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

    M = copy(A)
    rows, cols = size(M)

    # this will be an array of the Gauss transform matrices that will be calculated
    Gausses = Array{Matrix{Float64}}(undef, cols)

    for i = 1:(rows - 1)
        pivot = Matrix{Float64}(undef, rows, 1)
        pivot = [k for k = 1:rows]

        Gausses[i] = GaussTransform(A, i)
        M = Gausses[i] * M
    end

    U = M

    # explicitly form the L in the LU factorization by applying the
    # inverse of the Gauss transforms. this is because:
    # L(n) * L(n-1) * ... * L(1) * A = U
    # implies
    # A = inv(L(0)) * inv(L(1)) * ... inv(L(n)) * U
    L = Matrix{Float64}(I, rows, cols)
    for i = (rows - 1):-1:1
        L = inv(Gausses[i]) * L
    end

    return L, U
end

function LuBordered(A)

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

function swap(a, b)
    tmp = a
    a = b
    b = tmp
end

function PermutationMatrix(permuteVector)
    size = size(permuteVector)
    P = Matrix{Float64}(0, size, size)

    for i = 1:size
        P[i, permuteVector[i]] = 1
    end

    return P
end


# Let's test out this code that we wrote
A = [2. -1. 1.; -2. 2. 1; 4. -4. 1]

println("\n\n***********************")
println("A")
println("***********************")
display(A)


 l, u = LuBordered(A)

println("\n\n***********************")
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
