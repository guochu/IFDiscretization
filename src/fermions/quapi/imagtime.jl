fermionic_Δτ(f::AbstractBoundedFunction; β::Real, N::Int, μ::Real=0, δτ::Real=β/N) = fermionic_Δτ(f, β, N, μ, δτ)
fermionic_Δτ(f::AbstractBoundedFunction, disperse::Function; β::Real, N::Int, μ::Real=0, δτ::Real=β/N) = fermionic_Δτ(f, disperse, β, N, μ, δτ)

"""
    fermionic_Δτ(bath::AbstractFermionicNormalBath; N::Int, δτ::Real=bath.β/N)
    fermionic_Δτ(f::AbstractBoundedFunction; β::Real, N::Int, μ::Real=0, δτ::Real=β/N)
    fermionic_Δτ(f::AbstractBoundedFunction, disperse::Function; β::Real, N::Int, μ::Real=0, δτ::Real=β/N)

Fermionic hybridization function Δᵢⱼ on the imaginary-time axis.

The Feynman-Vernon influence functional has the form

    I[ā, a] = e^{Σᵢⱼ āᵢ Δᵢⱼ aⱼ},

where the minus sign is absorbed into the definition of Δ compared with the
usual convention; `i`, `j` are discrete imaginary-time step indices (nodes
τᵢ = (i-1)δτ).  Returns an `ImagCorrelationFunction` wrapping the Toeplitz
`CorrelationMatrix`.

Arguments:
- `f`:  the bath spectrum function A(ε) as an `AbstractBoundedFunction`
  (e.g. `DiscreteSpectrum`, `DiracDelta` or `bounded`);
- `disperse`:  optional dispersion applied to the integration variable,
  `ε -> disperse(ε)` (default `identity`);
- `β`:  the inverse temperature;
- `μ`:  the chemical potential, absorbed into the spectrum via `spectrumshift`;
- `N`:  number of discrete imaginary-time steps, such that δτ = β/N;
- `δτ`:  the step size (defaults to β/N).

The `bath` convenience form reads `β` and `μ` from the bath object.
"""
fermionic_Δτ(f0::AbstractBoundedFunction, β::Real, N::Int, μ::Real, δτ::Real=β / N) = fermionic_Δτ(f0, identity, β, N, μ, δτ)

function fermionic_Δτ(f0::AbstractBoundedFunction, disperse::Function, β::Real, N::Int, μ::Real, δτ::Real=β / N)
    # f′, lb, ub = f0.f, lowerbound(f0), upperbound(f0)
    β = convert(Float64, β)
    μ = convert(Float64, μ)
    f = spectrumshift(f0, μ)
    # δτ = β / N

    fⱼₖ(Δk::Int) = _fermionic_fⱼₖ_i_disperse(f, disperse, β, 0, Δk, δτ)
    fₖⱼ(Δk::Int) = _fermionic_fₖⱼ_i_disperse(f, disperse, β, 0, Δk, δτ)
    fⱼⱼ = _fermionic_fⱼⱼ_i_disperse(f, disperse, β, 0, δτ)
    fₖₖ = _fermionic_fₖₖ_i_disperse(f, disperse, β, 0, δτ)

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
