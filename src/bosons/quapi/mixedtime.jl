"""
    bosonic_Δm(f::AbstractBoundedFunction; β::Real, Nτ::Int, t::Real, Nt::Int, μ::Real=0, δτ::Real=β/Nτ)
    bosonic_Δm(f::AbstractBoundedFunction, disperse::Function; β::Real, Nτ::Int, t::Real, Nt::Int, μ::Real=0, δτ::Real=β/Nτ)

Bosonic hybridization function Δᵢⱼᵘᵛ on the Kadanoff-Baym (L-shaped) contour.

Same convention as `fermionic_Δm`: the influence functional reads
I = e^{Σᵤᵥ Σᵢⱼ sᵢᵘ Δᵢⱼᵘᵛ sⱼᵛ} (spin indices suppressed) with the minus sign
absorbed into Δ; `u`, `v ∈ (:+, :-, :τ)` label the forward/backward real-time
branch (Nt+1 nodes, δt = t/Nt) and the imaginary-time branch (Nτ nodes,
δτ = β/Nτ).  Returns a `MixedCorrelationFunction`.

Arguments:
- `f`:  the bath spectrum function A(ε) as an `AbstractBoundedFunction`;
- `disperse`:  optional dispersion applied to the integration variable,
  `ε -> disperse(ε)` (default `identity`);
- `β`:  the inverse temperature;
- `μ`:  the chemical potential;
- `Nτ`:  number of imaginary-time steps, δτ = β/Nτ;
- `t`:  the total real evolution time;
- `Nt`:  number of real-time steps, δt = t/Nt;
- `δτ`:  the imaginary-time step size (defaults to β/Nτ).

Note:  for μ = 0 the Bose factor carries a 1/ε singularity at ε = 0, and the
small-ε branch of the integrand diverges as `1/(β ε)`, which is `Inf` exactly
at ε = 0.  Spectra with support touching ε = 0 (a flat band, an ohmic-like
edge, or a delta peak at zero frequency) can therefore make the quadrature
raise a `DomainError`; shifting the band edge slightly above 0 avoids the
instability.
"""
bosonic_Δm(f::AbstractBoundedFunction; β::Real, Nτ::Int, t::Real, Nt::Int, μ::Real=0, δτ::Real=β/Nτ) = bosonic_Δm(
                                        f, identity, β=β, Nτ=Nτ, t=t, Nt=Nt, μ=μ, δτ=δτ)

function bosonic_Δm(f::AbstractBoundedFunction, disperse::Function; β::Real, Nτ::Int, t::Real, Nt::Int, μ::Real=0, δτ::Real=β/Nτ)
    δt = t / Nt
    # real time
    fⱼₖ(Δk) = _bosonic_fⱼₖ_r_disperse(f, disperse, β, μ, Δk, δt)
    fₖⱼ(Δk) = _bosonic_fₖⱼ_r_disperse(f, disperse, β, μ, Δk, δt)
    fⱼⱼ = _bosonic_fⱼⱼ_r_disperse(f, disperse, β, μ, δt)
    fₖₖ = _bosonic_fₖₖ_r_disperse(f, disperse, β, μ, δt)

    # imaginary time
    hⱼₖ(Δk::Int) = _bosonic_fⱼₖ_i_disperse(f, disperse, β, μ, Δk, δτ)
    hₖⱼ(Δk::Int) = _bosonic_fₖⱼ_i_disperse(f, disperse, β, μ, Δk, δτ)
    hⱼⱼ = _bosonic_fⱼⱼ_i_disperse(f, disperse, β, μ, δτ)
    hₖₖ = _bosonic_fₖₖ_i_disperse(f, disperse, β, μ, δτ)

    # mixed time
    lⱼₖ(j::Int, k::Int) = _bosonic_lⱼₖ_disperse(f, disperse, β, μ, j, k, δt, δτ)
    lₖⱼ(k::Int, j::Int) = _bosonic_lₖⱼ_disperse(f, disperse, β, μ, k, j, δt, δτ)

    N, M = Nt, Nτ

    # real time
    ηⱼₖ = zeros(ComplexF64, Nt+1)
    ηₖⱼ = zeros(ComplexF64, Nt+1)
    ηⱼₖ₂ = zeros(ComplexF64, Nt+1)
    ηₖⱼ₂ = zeros(ComplexF64, Nt+1)

    ηⱼₖ[1] = quadgkwrapper(-fⱼⱼ)
    for i = 1:Nt
        ηⱼₖ[i+1] = quadgkwrapper(-fⱼₖ(i))
    end
    ηₖⱼ[1] = quadgkwrapper(-fₖₖ)
    for i = 1:Nt
        ηₖⱼ[i+1] = quadgkwrapper(-fₖⱼ(-i))
    end
    # secondary pair (lags: un-conjugated negative-lag route; contacts: the
    # bosonic contact kernels fⱼⱼ/fₖₖ are not a conjugate-symmetric pair, so
    # the contacts are the conjugates of the primary contacts — exact for the
    # real-valued bosonic spectral densities)
    for i = 1:Nt
        ηⱼₖ₂[i+1] = quadgkwrapper(-fₖⱼ(i))
    end
    for i = 1:Nt
        ηₖⱼ₂[i+1] = quadgkwrapper(-fⱼₖ(-i))
    end
    ηⱼₖ₂[1] = conj(ηₖⱼ[1])
    ηₖⱼ₂[1] = conj(ηⱼₖ[1])

    # imag time
    ξⱼₖ = zeros(ComplexF64, M) # j >= k
    ξₖⱼ = zeros(ComplexF64, M)

    ξⱼₖ[1] = quadgkwrapper(-hⱼⱼ)
    for k in 2:M
        ξⱼₖ[k] = quadgkwrapper(-hⱼₖ(k-1))
    end

    ξₖⱼ[1] = quadgkwrapper(-hₖₖ)
    for k in 2:M
        ξₖⱼ[k] = quadgkwrapper(-hₖⱼ(k-1))
    end

    # mix time
    ζⱼₖ = zeros(ComplexF64, M, N+1)
    ζₖⱼ = zeros(ComplexF64, N+1, M)

    for j = 1:M, k = 1:N+1
        ζⱼₖ[j,k] = quadgkwrapper(lⱼₖ(j-1,k-1))
        ζₖⱼ[k,j] = quadgkwrapper(lₖⱼ(k-1,j-1))
    end
    # 统一约定：G₊₋ = CM(−ηⱼₖ₂, −ηₖⱼ)、G₋₊ = CM(−ηⱼₖ, −ηₖⱼ₂)（对角块由 branch 推导）
    MixedCorrelationFunction(CorrelationMatrix(-ηⱼₖ₂, -ηₖⱼ), CorrelationMatrix(-ηⱼₖ, -ηₖⱼ₂),
                             -im .* ζₖⱼ, -im .* ζⱼₖ, CorrelationMatrix(-ξⱼₖ, -ξₖⱼ))
end
