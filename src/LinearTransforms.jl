"""
Linear Functions
"""
module LinearTransforms

using SparseArrays, LinearAlgebra

export AbstractLinearTransform, LinearTransform, opMask, opMaskFactored, opProductForm,
    opMatrix, drandn, rrandn, dottest

import Base: *, -, +
import Base: show, size, adjoint, eltype, isreal

include("abstractlinop.jl")
include("utilities.jl")
include("matrixlike.jl")
include("masking.jl")
include("operations.jl")

end # module

