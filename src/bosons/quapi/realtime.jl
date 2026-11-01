bosonic_Δt(f::AbstractBoundedFunction; β::Real, N::Int, t::Real, μ::Real=0) = bosonic_Δt(f, β, N, t/N, μ)
bosonic_Δt(f::AbstractBoundedFunction, disperse::Function; β::Real, N::Int, t::Real, μ::Real=0) = bosonic_Δt(f, disperse, β, N, t/N, μ)

"""
    bosonic_Δt(bath::AbstractBosonicNormalBath; N::Int, t::Real)
    bosonic_Δt(f::AbstractBoundedFunction; β::Real, N::Int, t::Real, μ::Real=0)
    bosonic_Δt(f::AbstractBoundedFunction, disperse::Function; β::Real, N::Int, t::Real, μ::Real=0)

Bosonic hybridization function Δᵢⱼᵘᵛ on the real-time (closed Keldysh)
contour.

Same convention as `fermionic_Δt`: the influence functional reads
I = e^{Σᵤᵥ Σᵢⱼ sᵢᵘ Δᵢⱼᵘᵛ sⱼᵛ} (spin indices suppressed) with the minus sign
absorbed into Δ; `i`, `j` are discrete real-time step indices (nodes
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

Note:  for μ = 0 the Bose factor carries a 1/ε singularity at ε = 0, and the
small-ε branch of the integrand diverges as `1/(β ε)`, which is `Inf` exactly
at ε = 0.  Spectra with support touching ε = 0 (a flat band, an ohmic-like
edge, or a delta peak at zero frequency) can therefore make the quadrature
raise a `DomainError`; shifting the band edge slightly above 0 avoids the
instability.
"""
bosonic_Δt(f::AbstractBoundedFunction, β::Real, N::Int, δt::Real, μ::Real) = bosonic_Δt(f, identity, β, N, δt, μ)

function bosonic_Δt(f::AbstractBoundedFunction, disperse::Function, β::Real, N::Int, δt::Real, μ::Real)
    β = convert(Float64, β)
    μ = convert(Float64, μ)
    fⱼₖ(Δk) = _bosonic_fⱼₖ_r_disperse(f, disperse, β, μ, Δk, δt)
    fₖⱼ(Δk) = _bosonic_fₖⱼ_r_disperse(f, disperse, β, μ, Δk, δt)
    fⱼⱼ = _bosonic_fⱼⱼ_r_disperse(f, disperse, β, μ, δt)
    fₖₖ = _bosonic_fₖₖ_r_disperse(f, disperse, β, μ, δt)

    # primary Toeplitz pair only: the remaining branch blocks are derived by
    # the unified conjugation convention in branch (the historical four-block
    # storage of the cross branches is bit-identically reproduced for the
    # real-valued bosonic spectra — verified in the alignment tests), halving
    # the quadgk cost of the construction
    ηⱼₖ = zeros(ComplexF64, N+1)
    ηₖⱼ = zeros(ComplexF64, N+1)
    ηⱼₖ[1] = quadgkwrapper(-fⱼⱼ)
    ηₖⱼ[1] = quadgkwrapper(-fₖₖ)
    for i = 1:N
        ηⱼₖ[i+1] = quadgkwrapper(-fⱼₖ(i))
        ηₖⱼ[i+1] = quadgkwrapper(-fₖⱼ(-i))
    end
    # secondary pair by conjugation — exact for the real-valued bosonic
    # spectral densities; stored as the unified cross pair
    #   G₊₋ = CM(−conj(b), −b)、G₋₊ = CM(−a, −conj(a))
    return RealCorrelationFunction(CorrelationMatrix(.-conj.(ηₖⱼ), .-ηₖⱼ),
                                   CorrelationMatrix(.-ηⱼₖ, .-conj.(ηⱼₖ)))
end
