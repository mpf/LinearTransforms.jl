
"Abstract linear operator."
abstract type AbstractLinearTransform{T} end

"Operator dimensions are either integer or tuples of integers."
DimType = Union{Int,Tuple{Vararg{Int}}}

"General linear operator"
struct LinearTransform{T} <: AbstractLinearTransform{T}
    dimsRange::DimType
    dimsDomain::DimType
    prod::Function
    tprod::Function
end

"Constructor with default Float64 element type"
LinearTransform(dimsRange::DimType, dimsDomain::DimType, prod::Function, tprod::Function) = LinearTransform{Float64}(dimsRange, dimsDomain, prod, tprod)


eltype(::AbstractLinearTransform{T}) where {T} = T
isreal(op::AbstractLinearTransform) = eltype(op) <: Real
