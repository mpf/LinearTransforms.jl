
"Multiplication"
*(op::AbstractLinearTransform, x) = op.prod(x)

"Adjoint operator"
adjoint(op::AbstractLinearTransform) = LinearTransform(op.dimsRange, op.dimsDomain, op.tprod, op.prod)

"Unary addition"
+(op::AbstractLinearTransform) = op

"Unary negation"
function -(op::AbstractLinearTransform)
    prod = v -> -op.prod(v)
    tprod = w -> -op.tprod(w)
    LinearTransform(op.dimsRange, op.dimsDomain, prod, tprod)
end
