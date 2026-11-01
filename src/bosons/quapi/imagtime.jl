bosonic_Δτ(f::AbstractBoundedFunction; β::Real, N::Int, μ::Real=0, δτ::Real=β/N) = bosonic_Δτ(f, β, N, μ, δτ)
bosonic_Δτ(f::AbstractBoundedFunction, disperse::Function; β::Real, N::Int, μ::Real=0, δτ::Real=β/N) = bosonic_Δτ(f, disperse, β, N, μ, δτ)

"""
    bosonic_Δτ(bath::AbstractBosonicNormalBath; N::Int, δτ::Real=bath.β/N)
    bosonic_Δτ(f::AbstractBoundedFunction; β::Real, N::Int, μ::Real=0, δτ::Real=β/N)
    bosonic_Δτ(f::AbstractBoundedFunction, disperse::Function; β::Real, N::Int, μ::Real=0, δτ::Real=β/N)

Bosonic hybridization function Δᵢⱼ on the imaginary-time axis.

Same convention as `fermionic_Δτ`: the influence functional reads
I = e^{Σᵢⱼ sᵢ Δᵢⱼ sⱼ} (spin/flavor indices suppressed) with the minus sign
absorbed into Δ, `i`, `j` discrete imaginary-time step indices on a grid of
`N` steps δτ = β/N.  Returns an `ImagCorrelationFunction` wrapping the
Toeplitz `CorrelationMatrix`.

Arguments:
- `f`:  the bath spectrum function A(ε) as an `AbstractBoundedFunction`;
- `disperse`:  optional dispersion applied to the integration variable,
  `ε -> disperse(ε)` (default `identity`);
- `β`:  the inverse temperature;
- `μ`:  the chemical potential, absorbed into the spectrum via `spectrumshift`;
- `N`:  number of discrete imaginary-time steps, such that δτ = β/N;
- `δτ`:  the step size (defaults to β/N).

Note:  for μ = 0 the Bose factor carries a 1/ε singularity at ε = 0, and the
small-ε branch of the integrand evaluates `1/(β ε)`, which is `Inf` exactly at
ε = 0.  Spectra with support touching ε = 0 (a flat band, an ohmic-like edge,
or a delta peak at zero frequency) can therefore make the quadrature raise a
`DomainError`; shifting the band edge slightly above 0 avoids the instability.
"""
bosonic_Δτ(f0::AbstractBoundedFunction, β::Real, N::Int, μ::Real, δτ::Real=β / N) = bosonic_Δτ(f0, identity, β, N, μ, δτ)

function bosonic_Δτ(f0::AbstractBoundedFunction, disperse::Function, β::Real, N::Int, μ::Real, δτ::Real=β / N)
    # f′, lb, ub = f0.f, lowerbound(f0), upperbound(f0)
    β = convert(Float64, β)
    μ = convert(Float64, μ)
    f = spectrumshift(f0, μ)
    # δτ = β / N

    fⱼₖ(Δk::Int) = _bosonic_fⱼₖ_i_disperse(f, disperse, β, 0, Δk, δτ)
    fₖⱼ(Δk::Int) = _bosonic_fₖⱼ_i_disperse(f, disperse, β, 0, Δk, δτ)
    fⱼⱼ = _bosonic_fⱼⱼ_i_disperse(f, disperse, β, 0, δτ)
    fₖₖ = _bosonic_fₖₖ_i_disperse(f, disperse, β, 0, δτ)

    # j >= k
    L = N
    tmp = quadgkwrapper(fⱼⱼ)
    T = typeof(tmp)
    ηⱼₖ = zeros(T, L)
    ηⱼₖ[1] = tmp
    for k = 1:L-1
        ηⱼₖ[k+1] = quadgkwrapper(fⱼₖ(k))
    end

    ηₖⱼ = zeros(T, L)
    ηₖⱼ[1] = quadgkwrapper(fₖₖ)
    for k = 1:L-1
        ηₖⱼ[k+1] = quadgkwrapper(fₖⱼ(k))
    end
    ImagCorrelationFunction(CorrelationMatrix{T}(ηⱼₖ, ηₖⱼ))
end
