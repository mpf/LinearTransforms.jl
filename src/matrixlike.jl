
# Operator in product form ############################

"""
    op = opProductForm(R::AbstractMatrix, L::Abstract)

Create a linear operator equivalent to `R*L'`:

- forward mode: `op*x` returns (R*L')*x
- adjoint mode: `op'*y` returns (L*R')*y

In both cases, only matrix-vector products are used.
"""
function opProductForm(R::AbstractMatrix{T}, L::AbstractMatrix{T}) where {T}
    mR, rR = size(R)
    mL, rL = size(L)
    rR != rL && error("Dimensions not compatible")
    prod = x -> R*(L'*x)
    tprod = y -> L*(R'*y)
    LinearTransform{T}(mR, mL, prod, tprod)
end

# Operator on a matrix ####################################
# This isn't very useful, since we could just use the matrix
# itself. But it helps for testing.testing.testing

function opMatrix(A::AbstractMatrix{T}) where {T}
    m, n = size(A)
    LinearTransform{T}(m, n, x->A*x, y->A'*y)
end
