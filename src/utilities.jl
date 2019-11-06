
"""
    dimsRange, dimsDomain = size(op)

Return a tuple containg the dimensions of the operator `op`.
"""
size(op::AbstractLinearTransform) = (op.dimsRange, op.dimsDomain)
function size(op::AbstractLinearTransform, space::Symbol)
    if space == :domain
        op.dimsDomain
    elseif space == :range
        op.dimsRange
    else
        error("Unknown option")
    end
end

function size(op::AbstractLinearTransform, dim::Int)
    if dim == 1
        op.dimsRange
    elseif dim == 2
        op.dimsDomain
    else
        error("Unknown dim $dim")
    end
end

"""
    show(io, op)

Print a description of the generalized linear operator.
"""
function show(io::IO, op::AbstractLinearTransform)
    println(io,"Generalized linear transform")
    println(io,"  dim  range: ", op.dimsDomain)
    print(io,"  dim domain: ", op.dimsRange)
end


"""
    drandn(op::AbstractLinearTransform)

Return a pseudorandom vector in the domain of A.

See also [`rrandn`](@ref).
"""
drandn(op::AbstractLinearTransform) = randn(size(op,:domain))


"""
    rrandn(op::AbstractLinearTransform)

Return a pseudorandom vector in the domain of A.

See also [`drandn`](@ref).
"""
rrandn(op::AbstractLinearTransform) = randn(size(op,:range))


"""
    dottest(op::AbstractLinearTransform, [k=100])

Generate random vectors X and Y (from the domain and range of A), and verifies
that (A*X)'*Y = X'*(A'*Y) within a tolerance of 1E-10. This can help detect
errors in the operator; it canot be used to guarantee correctness.

The function returns true when the test succeeded and false if it failed.
"""
function dottest(A::AbstractLinearTransform, k=100; verbose=false)

    tol = √eps()
    err = 0.
    kpass = 0

    for i=1:k
        x   = drandn(A)
        y   = rrandn(A)
        z1  = dot(A*x, y)
        z2  = dot(x, A'*y)
        err = max(err, norm(z1-z2,Inf))
        if err < tol
            kpass += 1
        end
    end

    if verbose
        if kpass < k
            println("FAILED on $(k-kpass) out of $k tests")
            println("  maximum absolute difference of $err")
        else
            println("PASSED!");
        end
    end

    status = kpass != 0
end


"""
    Matrix(op::AbstractLinearTransform)

Return a matrix representation of the operator, if the sizes are
1-dimensional, and not tuples.
"""
function Matrix(op::AbstractLinearTransform)
    m, n = size(op)
    !(length(m) == length(n) == 1) && error("Dims cannot be tuples")
    T = eltype(op)
    mat = Matrix{T}(undef, (m, n))
    v = fill(zero(T), n)
    @inbounds for i = 1:n
        v[i] = one(T)
        mul!(view(mat, :, i), op, v)
        v[i] = zero(T)
    end
    return mat
end


