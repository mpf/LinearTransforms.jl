"""
Linear Functions
"""
module LinearTransforms

using SparseArrays, LinearAlgebra

export AbstractLinearTransform, LinearTransform, opMask,
    opMaskFactored,
    drandn, rrandn, dottest


import Base: *, -, +
import Base: show, size, adjoint, eltype, isreal, Matrix

include("abstractlinop.jl")
include("utilities.jl")
include("masking.jl")
include("operations.jl")

end # module

