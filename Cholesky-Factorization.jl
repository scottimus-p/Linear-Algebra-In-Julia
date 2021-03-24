# The code in this file performs a Cholesky factorization of a real-valued matrix.

module CholeskyFactorization

using LinearAlgebra


function Cholesky(A)
    # this function performs an Cholesky factorization on the matrix A using a
    # right-looking algorithm

    M = copy(A)
    rows, cols = size(M)

    @assert rows == cols

    for i = 1:rows
        M[i, i] = sqrt(M[i, i]) #choose the positive square root

        M[i+1:rows, i:i] = M[i+1:rows, i:i] / M[i:i, i:i]

        M[i+1:rows, i+1:cols] = M[i+1:rows, i+1:cols] - M[i+1:rows, i:i] * M[i+1:rows, i:i]'
    end

    return LowerTriangular(M)
end

end # end of module declaration
