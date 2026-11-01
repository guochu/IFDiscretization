println("------------------------------------")
println("|      Imaginary-time QuAPI        |")
println("------------------------------------")

using Test, LinearAlgebra

"""
    quapi_adt(β, ε, Δ, ωs, λs; N)

Imaginary-time QuAPI for the spin-boson model

    H = ε/2 σz + Δ/2 σx + Σ_k ω_k a_k†a_k + (σz/2) Σ_k λ_k (a_k† + a_k)

The Feynman-Vernon influence functional is discretized on N time slices
(δτ = β/N) and the full sum over the 2^N spin paths is assembled into an
augmented density tensor (ADT) of rank N and physical dimension 2.  Returns
the partition function Z, the Matsubara Green's function C[m+1] = ⟨σz(τ_m)σz(0)⟩
at τ_m = m δτ, and δτ.

Sign convention: following src/QuAPI.jl, the minus sign in the influence
functional exponent I[s] = exp[-∫∫ s Δ s] is absorbed into the hybridization
matrix η, so the discretized path weight is exp[+sᵀ η s].
"""
function quapi_adt(β, ε, Δ, ωs, λs; N)
    σx = [0.0 1.0; 1.0 0.0]
    σz = [1.0 0.0; 0.0 -1.0]
    δτ = β / N

    # discretized bath correlation η[j+1, k+1] = Δ_{jk}
    corr = bosonic_Δτ(DiscreteSpectrum(ωs, λs .^ 2), β=β, N=N, μ=0)
    η = corr.data

    # short-time propagator of the isolated spin
    Hs = ε/2 * σz + Δ/2 * σx
    W = exp(-δτ * Hs)

    # enumerate the 2^N spin paths s_j ∈ {±1/2}
    masks = UInt32.(0:2^N-1)
    s = Matrix{Float64}(undef, N, 2^N)
    for j in 0:N-1
        bit = (masks .>> j) .& 1
        s[j+1, :] .= ifelse.(bit .== 0, 0.5, -0.5)
    end

    # influence functional I[s] = exp[+sᵀηs]
    IF = exp.(sum(s .* (η * s), dims=1))

    # product of short-time spin propagators along the path (periodic s_N = s_0)
    prodW = ones(2^N)
    for j in 0:N-1
        i1 = (masks .>> j) .& 1 .+ 1
        i2 = (masks .>> ((j + 1) % N)) .& 1 .+ 1
        prodW .*= [W[i1[m], i2[m]] for m in 1:2^N]
    end

    # augmented density tensor: rank-N, physical dimension 2 per site
    ADT = reshape(IF[:] .* prodW, ntuple(_ -> 2, N))
    Z = sum(ADT)

    # ⟨σz(τ_m)σz(0)⟩ obtained by contracting the ADT with σz at sites m and 0
    σ0 = ifelse.((masks .& 1) .== 0, 1.0, -1.0)     # σz(s_0)
    C = zeros(N)
    for m in 0:N-1
        σm = ifelse.((masks .>> m) .& 1 .== 0, 1.0, -1.0)
        C[m+1] = sum(ADT .* reshape(σm .* σ0, size(ADT))) / Z
    end
    return Z, C, δτ
end

"""
    ed_hamiltonian(ε, Δ, ωs, λs; d)

Full spin-boson Hamiltonian on the truncated Fock space (d boson levels per
mode), to be diagonalized exactly as a reference.
"""
function ed_hamiltonian(ε, Δ, ωs, λs; d)
    σx = [0.0 1.0; 1.0 0.0]
    σz = [1.0 0.0; 0.0 -1.0]
    I2 = Matrix{Float64}(I, 2, 2)
    K = length(ωs)

    a  = [bosonaoperator(K, k; d=d) for k in 1:K]
    ad = [bosonadagoperator(K, k; d=d) for k in 1:K]
    n  = [bosondensityoperator(K, k; d=d) for k in 1:K]
    Ib = Matrix{Float64}(I, d^K, d^K)

    Hs = ε/2 * σz + Δ/2 * σx
    Hb = sum(ωs[k] * n[k] for k in 1:K)
    Hc = kron(σz/2, sum(λs[k] * (ad[k] + a[k]) for k in 1:K))
    H = kron(Hs, Ib) + kron(I2, Hb) + Hc
    return (H + H') / 2
end

"""
    quapi_ibm(β, ε, ωs, λs; N)

Independent-boson limit of the spin-boson model (tunneling Δ = 0) solved with
the discretized QuAPI influence functional.  Since σz is conserved when Δ = 0,
the only non-trivial Matsubara correlator is the spin-flip (fermionic) one

    G(τ_m) = ⟨d(τ_m) d†(0)⟩,   d = σ_-,  d† = σ_+.

It is given by the single-flip path integral: the spin takes +1/2 on [0, τ_m)
and -1/2 on [τ_m, β) (one domain wall), with weight

    W[1,1]^m W[2,2]^(N-m) × exp[+sᵀηs] / Z,

where W = exp(-δτ (ε/2) σz) is the diagonal short-time spin propagator and Z
receives contributions only from the two constant spin paths (since W is
diagonal).  Returns Z, G and δτ.

Note on normalization: exp[+sᵀηs] is built with the free-bath partition
function factored out (Ση = βΣλ²/ω exactly), so the returned Z is the true
partition function divided by Z_bath; the correlators (ratios) are unaffected.
"""
function quapi_ibm(β, ε, ωs, λs; N)
    δτ = β / N
    σz = [1.0 0.0; 0.0 -1.0]
    corr = bosonic_Δτ(DiscreteSpectrum(ωs, λs .^ 2), β=β, N=N, μ=0)
    η = corr.data
    W = exp(-δτ * (ε/2) * σz)                    # diagonal (Δ = 0)
    # partition function: only the constant spin paths contribute
    Z = (W[1,1]^N + W[2,2]^N) * exp(sum(η) / 4)
    # single-flip path: s_j = +1/2 for j < m, -1/2 for j ≥ m
    G = zeros(N)
    for m in 0:N-1
        sp = [(j < m ? 0.5 : -0.5) for j in 0:N-1]
        G[m+1] = W[1,1]^m * W[2,2]^(N-m) * exp(dot(sp, η, sp)) / Z
    end
    return Z, G, δτ
end

@testset "imaginary-time QuAPI (spin-boson) vs exact diagonalization" begin
    β, ε, Δ = 2.0, 0.5, 0.5
    d = 12   # boson Fock cutoff for ED
    cases = (("1 mode", [1.0], [0.5]),
             ("2 modes", [1.0, 2.0], [0.5, 0.3]))

    for (label, ωs, λs) in cases
        @testset "$label" begin
            K = length(ωs)
            H = ed_hamiltonian(ε, Δ, ωs, λs; d=d)
            cache = eigencache(H)
            σz_full = kron([1.0 0.0; 0.0 -1.0], Matrix{Float64}(I, d^K, d^K))

            # ED reference at τ = β/2 (used for the convergence check)
            ged_mid = correlation_2op_1τ(H, σz_full, σz_full, [β/2], cache; β=β, reverse=false)[1]

            # QuAPI on the fine grid: full curve vs ED
            N = 12
            Z, C, δτ = quapi_adt(β, ε, Δ, ωs, λs; N=N)
            @test Z > 0
            @test C[1] ≈ 1.0 atol=1.0e-10    # ⟨σz(0)σz(0)⟩ = 1 by construction

            τs = [(m) * δτ for m in 0:N-1]
            ged = correlation_2op_1τ(H, σz_full, σz_full, τs, cache; β=β, reverse=false)
            @test C ≈ ged atol=2.0e-3

            # convergence: finer discretization (smaller δτ) ⇒ smaller error
            Ncoarse = 6
            _, Ccoarse, _ = quapi_adt(β, ε, Δ, ωs, λs; N=Ncoarse)
            err_coarse = abs(Ccoarse[Ncoarse÷2+1] - ged_mid)
            err_fine   = abs(C[N÷2+1] - ged_mid)
            @test err_fine < err_coarse
            @test err_fine < 1.0e-3
        end
    end
end

@testset "independent-boson limit (Δ=0) vs ImpurityModelBase analytic solution" begin
    β, ε = 2.0, 0.5
    d = 12   # boson Fock cutoff for ED
    cases = (("1 mode", [1.0], [0.5]),
             ("2 modes", [1.0, 2.0], [0.5, 0.3]))

    for (label, ωs, λs) in cases
        @testset "$label" begin
            N = 16
            Z, G, δτ = quapi_ibm(β, ε, ωs, λs; N=N)
            τs = [(m) * δτ for m in 0:N-1]

            # ImpurityModelBase analytic solution for the independent-boson model.
            # Our spin-boson Hamiltonian maps onto a fermionic impurity (n=(σz+1)/2)
            # with level ϵ_d = ε + Δ, where Δ = Σλ²/ω is the polaron shift absorbed
            # inside the analytic solution, so that its renormalized level ϵ_d - Δ = ε
            # equals the bare level of the spin-boson Hamiltonian.
            Δ = sum(λs .^ 2 ./ ωs)
            spec = DiscreteSpectrum(ωs, λs .^ 2)
            Gan = [independentbosons_Gτ(spec, τ, β=β, ϵ_d=ε + Δ, bands=1) for τ in τs]
            @test G ≈ Gan atol=1.0e-8

            # cross-check against exact diagonalization of the same spin-boson
            # Hamiltonian (Δ_tun = 0), i.e. ⟨d(τ)d†(0)⟩
            H = ed_hamiltonian(ε, 0.0, ωs, λs; d=d)
            cache = eigencache(H)
            K = length(ωs)
            σm = [0.0 0.0; 1.0 0.0]
            σp = [0.0 1.0; 0.0 0.0]
            d_full  = kron(σm, Matrix{Float64}(I, d^K, d^K))
            dp_full = kron(σp, Matrix{Float64}(I, d^K, d^K))
            Ged = correlation_2op_1τ(H, d_full, dp_full, τs, cache; β=β, reverse=false)
            @test G ≈ Ged atol=1.0e-8

            # partition function with the free-bath Z factored out (see quapi_ibm)
            @test Z ≈ 2 * cosh(β*ε/2) * exp(β * sum(λs .^ 2 ./ (4 .* ωs))) atol=1.0e-8
        end
    end
end
