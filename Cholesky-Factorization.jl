# The code in this file performs a Cholesky factorization of a real-valued matrix.
#
# At the end of the file is code that runs the factorizations and checks the accuracy

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



# Let's test out this code that we wrote
A = Matrix(Hermitian(rand(6,6) + I))

println("\n\n***********************")
println("A")
println("***********************")
display(A)


 l = Cholesky(A)

println("\n\n***********************")
println("L")
println("***********************")
display(l)

println("\n\n***********************")
println("A - LL^H")
println("***********************")

display(A - l * l')
