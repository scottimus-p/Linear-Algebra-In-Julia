# The code in this file performs a Cholesky factorization of a real-valued matrix.

module CholeskyFactorization

using LinearAlgebra


function RightLookingCholesky(A)
# this function performs an Cholesky factorization on the matrix A using a
# right-looking algorithm

# using the matrix partitioning shown below, each iteration through the FOR loop
# examines α₁₁ and a₂₁ and solves for the lower triangular, overwriting the results
# in the original matrix along the way
#
#  A₀₀ | a₀₁ | A₀₂
# -----+-----+-----
# a₁₀ᵀ | α₁₁ | a₁₂ᵀ
# -----+-----+-----
#  A₂₀ | a₂₁ | A₂₂

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

function BorderedCholesky(A)
# this function performs an Cholesky factorization on the matrix A using a
# bordered algorithm such that A = LLᵀ

# using the matrix partitioning shown below, each iteration through the FOR loop
# examines a₁₀ᵀ, a₀₁ and α₁₁ (note that a₁₀ᵀ == a₀₁ because A must be symmetric
# positive definite to perform a Cholesky factorization) and solves for the lower
# triangular, overwriting results in the original matrix along the way
#
#  A₀₀ | a₀₁
# -----+-----
# a₁₀ᵀ | α₁₁

    M = copy(A)
    rows, cols = size(M)

    @assert rows == cols

    for i = 1:rows
        if i > 1
            M[i:i, 1:i-1] = transpose(inv(LowerTriangular(M[1:i-1, 1:i-1])) * M[1:i-1, i:i])
        end

        M[i, i] = sqrt(M[i, i] - (M[i:i, 1:i-1] * transpose(M[i:i, 1:i-1]))[1,1])
    end

    return LowerTriangular(M)

end

end # end of module declaration
