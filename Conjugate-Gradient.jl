


module ConjugateGradient

using LinearAlgebra

function CG(A, b)
# solves the linear system Ax = b using the conjugate gradient method

# returns both the solution and the number of iterations required to
# achieve that solution

    rows, cols = size(A)

    x = zeros(rows, 1)
    r = b

    r_prev = Matrix{Float64}(undef, rows, 1)
    p_prev = Matrix{Float64}(undef, rows, 1)

    n_iters = 0

    tolerance = eps() * norm(b)

    while norm(r) > tolerance
        if n_iters == 0
            p = r
        else
            γ = ((r' * r) / (r_prev' * r_prev))[1,1]
            p = r + γ * p_prev
        end

        α = ((r' * r) / (p' * A * p))[1, 1]

        r_prev = r
        p_prev = p

        x = x + α * p
        r = r - α * A * p
        n_iters += 1
    end

    return x, n_iters
end

end
