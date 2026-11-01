fermionic_Δt(f::AbstractBoundedFunction; β::Real, N::Int, t::Real, μ::Real=0) = fermionic_Δt(f, β, t, N, μ)
fermionic_Δt(f::AbstractBoundedFunction, disperse::Function; β::Real, N::Int, t::Real, μ::Real=0) = fermionic_Δt(f, disperse, β, t, N, μ)

"""
    fermionic_Δt(bath::AbstractFermionicNormalBath; N::Int, t::Real)
    fermionic_Δt(f::AbstractBoundedFunction; β::Real, N::Int, t::Real, μ::Real=0)
    fermionic_Δt(f::AbstractBoundedFunction, disperse::Function; β::Real, N::Int, t::Real, μ::Real=0)

Fermionic hybridization function Δᵢⱼᵘᵛ on the real-time (closed Keldysh)
contour.

The Feynman-Vernon influence functional has the form

    I[ā, a] = e^{Σᵤᵥ Σᵢⱼ āᵢᵘ Δᵢⱼᵘᵛ aⱼᵛ},

where the minus sign is absorbed into the definition of Δ compared with the
usual convention; `i`, `j` are discrete real-time step indices (nodes
tᵢ = i·δt, δt = t/N) and `u`, `v ∈ (:+, :-)` label the forward/backward
branch.  Returns a `RealCorrelationFunction` whose four blocks are selected by
`branch(x; b1, b2)`.

Arguments:
- `f`:  the bath spectrum function A(ε) as an `AbstractBoundedFunction`;
- `disperse`:  optional dispersion applied to the integration variable,
  `ε -> disperse(ε)` (default `identity`);
- `β`:  the inverse temperature;
- `μ`:  the chemical potential;
- `t`:  the total evolution time;
- `N`:  number of discrete real-time steps, such that δt = t/N.

The `bath` convenience form reads `β` and `μ` from the bath object.
"""
fermionic_Δt(f::AbstractBoundedFunction, β::Real, t::Real, N::Int, μ::Real) = fermionic_Δt(f, identity, β, t, N, μ)

function fermionic_Δt(f::AbstractBoundedFunction, disperse::Function, β::Real, t::Real, N::Int, μ::Real)
    β = convert(Float64, β)
    μ = convert(Float64, μ)
    δt = t / N
    g₁(ε) = fermidirac(β, μ - disperse(ε)); g₂(ε) = fermidirac(β, disperse(ε) - μ)
    fⱼₖ(Δk) = _fⱼₖ_r_disperse(f, disperse, Δk, δt)
    fⱼⱼ = _fⱼⱼ_r_disperse(f, disperse, δt)
    fₖₖ = _fₖₖ_r_disperse(f, disperse, δt)

    # The four branch blocks share four distinct Toeplitz vectors:
    #   G₊₊ = CM(a, b)   G₊₋ = CM(c, b)   G₋₊ = CM(a, d)   G₋₋ = CM(c, d)
    # computed once here (previously each was quadruplicated, doubling the
    # quadgk cost).  Backward-sector entries use the negative-lag cell kernel
    # fⱼₖ(-i) with the spectrum un-conjugated — the physical correlator carries
    # the same weight at forward and backward lags; conjugation is only used
    # for the branch DERIVATION, which is exact for real-valued spectra.
    a = zeros(ComplexF64, N+1)
    b = zeros(ComplexF64, N+1)
    a[1] = quadgkwrapper(-fⱼⱼ * g₁)
    b[1] = quadgkwrapper(fₖₖ * g₂)
    for i = 1:N
        a[i+1] = quadgkwrapper(-fⱼₖ(i) * g₁)
        b[i+1] = quadgkwrapper(fⱼₖ(-i) * g₂)
    end
    # secondary pair computed by its own quadrature with the spectrum
    # un-conjugated — complex-spectrum safe (BCS anomalous coherence weights),
    # and for real-valued spectra bit-identical to conj(b)/conj(a)
    c = zeros(ComplexF64, N+1)
    d = zeros(ComplexF64, N+1)
    c[1] = quadgkwrapper(fⱼⱼ * g₂)
    d[1] = quadgkwrapper(-fₖₖ * g₁)
    for i = 1:N
        c[i+1] = quadgkwrapper(fⱼₖ(i) * g₂)
        d[i+1] = quadgkwrapper(-fⱼₖ(-i) * g₁)
    end
    # 统一约定：G₊₋ = CM(−c,−b)、G₋₊ = CM(−a,−d)（对角块由 branch 推导）
    return RealCorrelationFunction(CorrelationMatrix(.-c, .-b), CorrelationMatrix(.-a, .-d))
end
