module IR2Reg

export prox!, shifted, shift!

using ADNLPModels
using LinearAlgebra
using NLPModels
using Printf
using ProxTV
using ProximalOperators
# using RegularizedOptimization
using ShiftedProximalOperators
using SolverCore
using RegularizedProblems
using Quadmath

function __init__()
    #empty for now
end

# main library functions
include("ir2reg_algo.jl")
include("utils.jl")
include("normLp.jl")
include("normTVp.jl")

end
