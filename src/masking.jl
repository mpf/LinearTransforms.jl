
# Mask operator functions ##################################

struct opMaskFactored{T} <: AbstractLinearTransform{T} end
"""
    op = opMaskFactored(M::SparseMatrixCSC)

Create a masking linear operator that applies to factored matrices.

- forward mode: `op*(U,V)` returns a vector of the elements of the matrix `U*V'` indexed by the mask;

- adjoint mode: `op'*y` returns a sparse matrix with the elements of `y` scattered into the mask pattern.
"""
function opMaskFactored{T}(M::SparseMatrixCSC) where {T}

    # forward operation is a gather.
    function forward(nnzM, I, J, U, V)
        z = zeros(nnzM)
        for (u,v) in zip(eachcol(U), eachcol(V))
            z += [ u[i]*v[j] for (i,j) in zip(I, J) ]
        end
        return z
    end

    # Adjoint operation is scatter.
    adjoint(I, J, w) = sparse(I, J, w)

    dimsDomain = size(M)
    dimsRange = (nnz(M),)
    I, J = findnz(M)[1:2]
    prod = uv -> forward(nnz(M), I, J, uv[1], uv[2])
    tprod = w -> adjoint(I, J, w)
    LinearTransform{T}(dimsRange, dimsDomain, prod, tprod)
end
opMaskFactored(M::SparseMatrixCSC) = opMaskFactored{Float64}(M::SparseMatrixCSC)

struct opMask{T} <: AbstractLinearTransform{T} end
"""
    op = opMask(M::SparseMatrixCSC)

Create a masking linear operator.

- forward mode: `op*(X)` returns a vector of the elements of the matrix `X` indexed by the mask;

- adjoint mode: `op'*y` returns a sparse matrix with the elements of `y` scattered into the mask pattern.
"""
function opMask{T}(M::SparseMatrixCSC) where {T}
    dimsDomain = size(M)
    dimsRange = (nnz(M),)
    I, J = findnz(M)[1:2]
    prod = X -> [X[i,j] for (i,j) in zip(I,J)]
    tprod = w -> sparse(I, J, w, dimsDomain...)
    LinearTransform{T}(dimsRange, dimsDomain, prod, tprod)
end
opMask(M::SparseMatrixCSC) = opMask{Float64}(M)
