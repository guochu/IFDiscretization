
"""
    fermionic_Δm(bath::AbstractFermionicNormalBath; Nτ::Int, t::Real, Nt::Int, δτ::Real=bath.β/Nτ)
    fermionic_Δm(f::AbstractBoundedFunction; β::Real, Nτ::Int, t::Real, Nt::Int, μ::Real=0, δτ::Real=β/Nτ)
    fermionic_Δm(f::AbstractBoundedFunction, disperse::Function; β::Real, Nτ::Int, t::Real, Nt::Int, μ::Real=0, δτ::Real=β/Nτ)

Fermionic hybridization function Δᵢⱼᵘᵛ on the Kadanoff-Baym (L-shaped)
contour.

The Feynman-Vernon influence functional has the form

    I[ā, a] = e^{Σᵤᵥ Σᵢⱼ āᵢᵘ Δᵢⱼᵘᵛ aⱼᵛ},

where the minus sign is absorbed into the definition of Δ compared with the
usual convention; `i`, `j` are discrete step indices and `u`, `v ∈ (:+, :-, :τ)`
label the branch: `:+`/`:-` the forward/backward real-time branch (Nt+1 nodes
tᵢ = i·δt, δt = t/Nt), `:τ` the imaginary-time branch (Nτ nodes
τᵢ = (i-1)δτ, δτ = β/Nτ).  Returns a `MixedCorrelationFunction`.

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

The `bath` convenience form reads `β` and `μ` from the bath object.
"""
fermionic_Δm(f::AbstractBoundedFunction; β::Real, Nτ::Int, t::Real, Nt::Int, μ::Real=0, δτ::Real=β/Nτ) = fermionic_Δm(
                f, identity, β=β, Nτ=Nτ, t=t, Nt=Nt, μ=μ, δτ=δτ)

function fermionic_Δm(f::AbstractBoundedFunction, disperse::Function; β::Real, Nτ::Int, t::Real, Nt::Int, μ::Real=0, δτ::Real=β/Nτ)
    δt = t / Nt
    g₁(ε) = fermidirac(β, μ - disperse(ε))
    g₂(ε) = fermidirac(β, disperse(ε) - μ)
    # real time
    fⱼₖ(Δk) = _fⱼₖ_r_disperse(f, disperse, Δk, δt)
    fⱼⱼ = _fⱼⱼ_r_disperse(f, disperse, δt)
    fₖₖ = _fₖₖ_r_disperse(f, disperse, δt)
    # imaginary time
    hⱼₖ(Δk::Int) = _fermionic_fⱼₖ_i_disperse(f, disperse, β, μ, Δk, δτ)
    hₖⱼ(Δk::Int) = _fermionic_fₖⱼ_i_disperse(f, disperse, β, μ, Δk, δτ)
    hⱼⱼ = _fermionic_fⱼⱼ_i_disperse(f, disperse, β, μ, δτ)
    hₖₖ = _fermionic_fₖₖ_i_disperse(f, disperse, β, μ, δτ)

    # mixed time
    lⱼₖ(j::Int, k::Int) = _fermionic_lⱼₖ_disperse(f, disperse, β, μ, j, k, δt, δτ)
    lₖⱼ(k::Int, j::Int) = _fermionic_lₖⱼ_disperse(f, disperse, β, μ, k, j, δt, δτ)

    N, M = Nt, Nτ

    # real time
    ηⱼₖ = zeros(ComplexF64, Nt+1)
    ηₖⱼ = zeros(ComplexF64, Nt+1)
    ηⱼₖ₂ = zeros(ComplexF64, Nt+1)
    ηₖⱼ₂ = zeros(ComplexF64, Nt+1)

    ηⱼₖ[1] = quadgkwrapper(-fⱼⱼ * g₁)
    for i = 1:Nt
        ηⱼₖ[i+1] = quadgkwrapper(-fⱼₖ(i) * g₁)
    end
    ηₖⱼ[1] = quadgkwrapper(fₖₖ * g₂)
    for i = 1:Nt
        ηₖⱼ[i+1] = quadgkwrapper(fⱼₖ(-i) * g₂)
    end
    # secondary pair (un-conjugated spectrum; equals conj of the primary pair
    # for real-valued spectra)
    ηⱼₖ₂[1] = quadgkwrapper(fⱼⱼ * g₂)
    for i = 1:Nt
        ηⱼₖ₂[i+1] = quadgkwrapper(fⱼₖ(i) * g₂)
    end
    ηₖⱼ₂[1] = quadgkwrapper(-fₖₖ * g₁)
    for i = 1:Nt
        ηₖⱼ₂[i+1] = quadgkwrapper(-fⱼₖ(-i) * g₁)
    end

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
