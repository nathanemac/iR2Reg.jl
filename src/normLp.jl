# NormLp and ShiftedNormLp Implementation

"""
    NormLp(λ::Real or AbstractArray, p::Real)

Represents the Lp norm with parameter `p` and scaling factor `λ`.
"""
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

        p >= 1 || error("p must be greater than or equal to one")
        new{T1,T2}(λ, p)
    end
end

# Allows NormLp objects to be called as functions
function (h::NormLp)(x::AbstractArray)
    return h.λ * ProxTV.LPnorm(x, length(x), h.p)
end

"""
    ShiftedNormLp

A mutable struct representing a shifted NormLp function.
"""
mutable struct ShiftedNormLp{
    R<:Real,
    T<:Real,
    V0<:AbstractVector{R},
    V1<:AbstractVector{R},
    V2<:AbstractVector{R},
}
    h::NormLp{R,T}
    xk::V0
    sj::V1
    sol::V2
    shifted_twice::Bool
    xsy::V2

    function ShiftedNormLp(
        h::NormLp{R,T},
        xk::AbstractVector{R},
        sj::AbstractVector{R},
        shifted_twice::Bool,
    ) where {R<:Real,T<:Real}
        sol = similar(xk)
        xsy = similar(xk)
        new{R,T,typeof(xk),typeof(sj),typeof(sol)}(h, xk, sj, sol, shifted_twice, xsy)
    end
end

"""
    shifted(h::NormLp, xk::AbstractVector)

Creates a ShiftedNormLp object with initial shift `xk`.
"""
shifted(h::NormLp{R,T}, xk::AbstractVector{R}) where {R<:Real,T<:Real} =
    ShiftedNormLp(h, xk, zero(xk), false)

"""
    shifted(ψ::ShiftedNormLp, sj::AbstractVector)

Creates a ShiftedNormLp object by adding a second shift `sj`.
"""
shifted(
    ψ::ShiftedNormLp{R,T,V0,V1,V2},
    sj::AbstractVector{R},
) where {
    R<:Real,
    T<:Real,
    V0<:AbstractVector{R},
    V1<:AbstractVector{R},
    V2<:AbstractVector{R},
} = ShiftedNormLp(ψ.h, ψ.xk, sj, true)

# Functions to get the name, expression, and parameters of the function
fun_name(ψ::ShiftedNormLp) = "shifted Lp norm"
fun_expr(ψ::ShiftedNormLp) = "t ↦ λ * ‖xk + sj + t‖ₚ"
fun_params(ψ::ShiftedNormLp) = "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)"

"""
    shift!(ψ::ShiftedNormLp, shift::AbstractVector)

Updates the shift of a ShiftedNormLp object.
"""
function shift!(ψ::ShiftedNormLp, shift::AbstractVector{R}) where {R<:Real}
    if ψ.shifted_twice
        ψ.sj .= shift
    else
        ψ.xk .= shift
    end
    return ψ
end

# Allows ShiftedNormLp objects to be called as functions
function (ψ::ShiftedNormLp)(y::AbstractVector)
    @. ψ.xsy = ψ.xk + ψ.sj + y
    return ψ.h(ψ.xsy)
end

"""
    prox!(y::AbstractArray, ψ::ShiftedNormLp, q::AbstractArray, σ::Real; objGap=1e-4)

Computes the proximity operator of a shifted Lp norm.

Inputs:
    - `y`: Array in which to store the result.
    - `ψ`: ShiftedNormLp object.
    - `q`: Vector to which the proximity operator is applied.
    - `σ`: Scaling factor.
    - `objGap`: Desired quality of the solution in terms of duality gap (default `1e-5`).
"""
function prox!(
    y::AbstractArray,
    ψ::ShiftedNormLp,
    q::AbstractArray,
    σ::Real;
    objGap::Real = 1e-5,
)
    n = length(y)
    ws = ProxTV.newWorkspace(n)

    # Allocate info array (based on C++ code)
    info = zeros(Float64, 3)

    # Compute y_shifted = xk + sj + q
    y_shifted = ψ.xk .+ ψ.sj .+ q

    # Adjust lambda to account for σ (multiply λ by σ)
    lambda_scaled = ψ.h.λ * σ

    # Check if all elements of y_shifted are non-negative
    positive = Int32(all(v -> v >= 0, y_shifted) ? 1 : 0)

    # Allocate the x vector to store the intermediate solution
    x = similar(y)

    # Call the PN_LPp function from ProxTV package
    ProxTV.PN_LPp(y_shifted, lambda_scaled, x, info, n, ψ.h.p, ws, positive, objGap)

    # Compute s = x - xk - sj
    s = x .- ψ.xk .- ψ.sj

    # Store the result in y
    y .= s


    return y
end
