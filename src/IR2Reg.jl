module IR2Reg

using ADNLPModels
using LinearAlgebra
using NLPModels
using Printf
using ProxTV
using ProximalOperators
using RegularizedOptimization
using ShiftedProximalOperators
using SolverCore


function __init__()
    #empty for now
end

# main library functions
include("ir2reg_algo.jl")
include("utils.jl")

end
