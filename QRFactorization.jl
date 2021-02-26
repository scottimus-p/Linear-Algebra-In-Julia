using LinearAlgebra

function GramSchmidt(A)
  rows = size(A)[1]
  cols = size(A)[2]

  Q = zeros(rows, rows)
  R = zeros(rows, cols)

  for i = 1:cols
      tmp = A[1:rows, i]
      for j = 1:(i - 1)
          R[j, i] = transpose(Q[1:rows, j]) * A[1:rows, i]

          tmp -= R[j, i] * Q[1:rows, j]
      end

      R[i, i] = VectorTwoNorm(tmp)
      Q[1:rows, i] = tmp / R[i,i]
  end

  return Q, R
end

function ModifiedGramSchmidt(A)
    M = copy(A)

    rows = size(M)[1]
    cols = size(M)[2]

    R = zeros(cols, cols)

    for i = 1:cols
        for j = 1:(i - 1)
            R[j, i] = transpose(M[1:rows, j]) * M[1:rows, i]

            M[1:rows, i] -= R[j, i] * M[1:rows, j]
        end

        R[i, i] = VectorTwoNorm(M[1:rows, i])
        M[1:rows, i] = M[1:rows, i] / R[i,i]
    end

    return A, R
end

function VectorTwoNorm(v)
  accum = 0

  for i = 1:size(v)[1]
    accum += v[i] ^ 2
  end

  return accum ^ 0.5
end

function HouseHolderQr(A)
    rows = size(A)[1]
    cols = size(A)[2]

    R = copy(A)
    Householders = Array{Matrix{Float64}}(undef, cols)

    for i = 1:cols
        H = HouseholderMatrix(R[i:rows, i])
        Householders[i] = H
        R[i:rows, i:cols] = H * R[i:rows, i:cols]
    end

    Q = Matrix(I, rows, rows)
    for i = reverse(1:cols)
        Q = Expand(Householders[i], rows, rows) * Q
    end

    return Q, R
end

function HouseholderVector(v)
    reflected = VectorTwoNorm(v) * StandardBasisVector(size(v)[1], 1)
    return (v - reflected) / VectorTwoNorm(v - reflected)
end

function HouseholderMatrix(v)
    u = HouseholderVector(v)
    return Matrix(I, size(u)[1], size(u)[1]) - 2 * u * u'
end

function Expand(A, rows, cols)
    tmp = copy(A)
    m = Matrix{Float64}(I, rows, cols)
    m[rows - size(A)[1] + 1:rows, cols - size(A)[2] + 1:cols] = A

    return m
end


function StandardBasisVector(dim, i)
    return Matrix(I, dim, dim)[i, 1:dim]
end


A = [1. -1. 4.; 1. 4. -2.; 1. 4. 2.; 1. -1. 0.]

#A = [0.5 1; 0 1]

println("\n\n***********************")
println("A")
println("***********************")
display(A)


q, r = HouseHolderQr(A)

println("\n\n***********************")
println("Q")
println("***********************")
display(q)

println("\n\n***********************")
println("R")
println("***********************")
display(r)

println("\n\n***********************")
println("A - QR")
println("***********************")
display(A - q * r)
