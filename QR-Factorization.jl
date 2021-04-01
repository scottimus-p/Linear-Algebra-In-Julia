# The code in this file performs a QR factorization of a real-valued matrix using three
# different methods:
#
# 1) Gram-Schmidt orthogonalization
# 2) A modified version of Gram-Schmidt that's supposed to be more efficient and have less floating-point error
# 3) QR factorization via Householder matrices

module QrFactorization

using LinearAlgebra


function GramSchmidt(A)
# performs a QR factorization of matrix A using the Gram-Schmidt algorithm

# this algorithm only works for rows ≥ cols

    M = copy(A)
    rows, cols = size(M)

    Q = zeros(rows, rows)
    R = zeros(rows, cols)

    for i = 1:cols
        tmp = M[1:rows, i:i]
        R[1:i, i:i] = Q[1:rows, 1:i]' * M[1:rows, i:i]
        tmp -= Q[1:rows, 1:i] * R[1:i, i:i]

        R[i, i] = VectorTwoNorm(tmp)
        Q[1:rows, i:i] = tmp / R[i, i]
    end

    return Q, R
end



function ModifiedGramSchmidt(A)
# performs a QR factorization of matrix A using a modified version of the Gram-Schmidt algorithm
# overwrites values in A as it goes along rather than creating an additional matrix Q

# this algorithm only works for rows ≥ cols

    M = copy(A)
    rows, cols = size(M)

    R = zeros(cols, cols)

    for i = 1:cols
        for j = 1:(i - 1)
            R[j:j, i:i] = M[1:rows, j:j]' * M[1:rows, i:i]
        end

        M[1:rows, i:i] -= M[1:rows, 1:i] * R[1:i, i:i]

        R[i, i] = VectorTwoNorm(M[1:rows, i:i])
        M[1:rows, i:i] = M[1:rows, i:i] / R[i, i]
    end

    return M, R
end



function Householder(A)
# performs a QR factorization of matrix A using Householder matrices

    R = copy(A)
    rows, cols = size(R)

    # this will be an array of the Householder matrices that will be calculated
    Householders = Array{Matrix{Float64}}(undef, cols)

    # calculate the Householder matrices and update the R matrix
    for i = 1:cols
        H = HouseholderMatrix(R[i:rows, i:i])
        Householders[i] = H
        R[i:rows, i:cols] = H * R[i:rows, i:cols]
    end

    # successively multiply the Householder matrices to get the Q matrix
    Q = Matrix(I, rows, rows)
    for i = reverse(1:cols)
        Q = ExpandHouseholder(Householders[i], rows, rows) * Q
    end

    return Q, R
end



function VectorTwoNorm(v)
# calculates the 2-norm of a vector v (i.e. the Euclidean length)
    accum = 0

    for i = 1:size(v)[1]
        accum += v[i] ^ 2
    end

    return accum ^ 0.5
end



function HouseholderVector(v)
# returns a vector that defines the plane that "mirrors" a given vector onto a scaled standard basis vector
    reflected = VectorTwoNorm(v) * StandardBasisVector(size(v)[1], 1)
    return (v - reflected) / VectorTwoNorm(v - reflected)
end



function HouseholderMatrix(v)
# returns a Householder matrix for vector v
    u = HouseholderVector(v)
    return Matrix(I, size(u)[1], size(u)[1]) - 2 * u * u'
end



function ExpandHouseholder(H, rows, cols)
# this function takes a Householder matrix (which is always square) and expands by adding
# an identity matrix in the top-left, the original Householder in the bottom-right, and zeros
# elsewhere so that the resulting matrix has dimensions (rows, cols). In other words, the
# resulting matrix will be:
#   I | 0
#   - + -
#   0 | H

    m = Matrix{Float64}(I, rows, cols)
    m[rows - size(H)[1] + 1:rows, cols - size(H)[2] + 1:cols] = H

    return m
end



function StandardBasisVector(dim, i)
# returns the i-th standard basis vector of size (dim)

    return Matrix(I, dim, dim)[1:dim, i:i]
end


end # end of module declaration
