module IR2Reg

export NormLp, inexact_prox!, ShiftedNormLp, shifted, fun_name, fun_expr, fun_params, shift!


using ADNLPModels
using LinearAlgebra
using NLPModels
using Printf
# using ProxTV
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

end
