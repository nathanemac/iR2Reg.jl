# NormLp

struct NormLp{T1,T2}
    λ::T1
    p::T2

    function NormLp(λ::T1, p::T2) where {T1,T2}
        if λ isa Real
            λ < 0 && error("λ must be nonnegative")
        elseif λ isa AbstractArray
            eltype(λ) <: Real || error("Elements of λ must be real")
            any(λ .< 0) && error("All elements of λ must be nonnegative")
        else
            error("λ must be a real scalar or array")
        end

        p >= 1 || error("p must be greater than one")
        new{T1,T2}(λ, p)
    end
end

function (h::NormLp)(x::AbstractArray)
    return ProxTV.LPnorm(x, length(x), h.p)
end

"""
    inexact_prox!( h::NormLp, x::AbstractArray, y::AbstractArray, objGap::Real)
    Given a reference signal y and a penalty parameter lambda, solves the proximity operator

        min_x 0.5 ||x-y||^2 + λ||x||_p ,

    for the Lp norm. To do so a Projected Newton algorithm is used.

    Inputs:
        - h: NormLp object.
        - y: reference signal.
        - x: array in which to store the solution.
        - lambda: penalty parameter.
        - p: degree of the Lp norm.
        - objGap: desired quality of the solution in terms of duality gap.
"""
function inexact_prox!(h::NormLp, y::AbstractArray, x::AbstractArray, objGap::Real)
    n = length(y)
    ws = ProxTV.newWorkspace(n)
    positive = all(x -> x >= 0, y) ? 1 : 0
    info = []

    ProxTV.PN_LPp(y, h.λ, x, info, n, h.p, ws, positive, objGap)
    return x
end


# ShiftedNormLp

mutable struct ShiftedNormLp{
    R<:Real,
    V0<:AbstractVector{R},
    V1<:AbstractVector{R},
    V2<:AbstractVector{R},
} #<: ShiftedProximableFunction
    h::NormLp{R,R}
    xk::V0
    sj::V1
    sol::V2
    shifted_twice::Bool
    xsy::V2

    function ShiftedNormLp(
        h::NormLp{R,R},
        xk::AbstractVector{R},
        sj::AbstractVector{R},
        shifted_twice::Bool,
    ) where {R<:Real}
        sol = similar(xk)
        xsy = similar(xk)
        new{R,typeof(xk),typeof(sj),typeof(sol)}(h, xk, sj, sol, shifted_twice, xsy)
    end
end

shifted(h::NormLp{R,R}, xk::AbstractVector{R}) where {R<:Real} =
    ShiftedNormLp(h, xk, zero(xk), false)

shifted(
    ψ::ShiftedNormLp{R,V0,V1,V2},
    sj::AbstractVector{R},
) where {R<:Real,V0<:AbstractVector{R},V1<:AbstractVector{R},V2<:AbstractVector{R}} =
    ShiftedNormLp(ψ.h, ψ.xk, sj, true)

fun_name(ψ::ShiftedNormLp) = "shifted Lp norm"
fun_expr(ψ::ShiftedNormLp) = "t ↦ ‖xk + sj + t‖ₚ"
fun_params(ψ::ShiftedNormLp) = "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)"


"""
    shift!(ψ, x)

Update the shift of a shifted NormLp object.
"""
function shift!(ψ::ShiftedNormLp, shift::AbstractVector{R}) where {R<:Real}
    if ψ.shifted_twice
        ψ.sj .= shift
    else
        ψ.xk .= shift
    end
    return ψ
end

function (ψ::ShiftedNormLp)(y::AbstractVector)
    @. ψ.xsy = ψ.xk + ψ.sj + y
    return ψ.h(ψ.xsy)
end

function inexact_prox!(ψ::ShiftedNormLp, y::AbstractArray, x::AbstractArray, objGap::Real)
    n = length(y)
    ws = ProxTV.newWorkspace(n)
    info = []

    # Adjust y by the shift
    y_shifted = y .+ ψ.xk .+ ψ.sj
    positive = all(v -> v >= 0, y_shifted) ? 1 : 0

    # Applies the proximal operator to the shifted object
    ProxTV.PN_LPp(y_shifted, ψ.h.λ, x, info, n, ψ.h.p, ws, positive, objGap)
    return x
end
