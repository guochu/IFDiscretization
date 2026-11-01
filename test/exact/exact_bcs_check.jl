# =============================================================================
# exact/exact_bcs_check.jl —— BCS 严格装配检验
# ← QuAPI/exactif/test/exact_bcs_check.jl 移植：include 链改 using
#   IFDiscretization（父层），`M=` 关键字改 `gfsolver=EigenSolver(...)`；
#   mb_generator 锚来自 _anchor_utils.jl（brute.jl 拷贝）；_quad_nodes 经
#   IFD. 限定访问（内部 helper）。
#
# T1  Fock ED 定义级锚定（1 模浴，8 维）：多体 H = mb_generator(h₊) ⊕ εn↓
#     ⊕ Δ(c†↑c†↓+h.c.)（JW，配对符号 +1 由 EOM i∂c↑⊃+Δc↓† 锁定），
#     ρ = e^{−βH}/Z，轮廓收缩 ⟨T_C d(t_v)d†(t_w)⟩ 逐位对照四 Nambu 块
#     （(1,2)/(2,1) 连同提取符号 sgn 一起锚定；实/复 Δ）
# T2  Δ=0 退化：Nambu 粒子扇区解耦 ⟹ (1,1) 环 = 费米杂质 GF 闭式
# T3  V→0：⟨dd†⟩ → ½、反常 → 0
# T4  μ≠0 ⟹ ArgumentError
# =============================================================================
include(joinpath(@__DIR__, "_anchor_utils.jl"))
using Test, LinearAlgebra

Γ0, Wb, β = 0.15, 3.0, 2.0
f = bounded(ω -> Γ0, -Wb, Wb)

"1 模 BCS 浴的多体 H（16 维；模式序 d↑=1, c↑=2, d↓=3, c↓=4，JW）"
function bcs_ed_h(ε::Real, V::Real, Δ::Number)
    h2 = [0.0 V; V ε]
    G = mb_generator(h2)                        # 星形 (杂质, c) 二模生成元
    H = zeros(ComplexF64, 16, 16)
    H += kron(G, Matrix{ComplexF64}(I, 4, 4))   # ↑ 扇区（bit 1,2）
    H += kron(Matrix{ComplexF64}(I, 4, 4), G)   # ↓ 扇区（bit 3,4）
    ann(k) = begin                              # mode k 湮没（JW，1-based）
        D = zeros(ComplexF64, 16, 16)
        for s in 0:15
            (s >> (k - 1)) & 1 == 1 || continue
            sgn = (-1)^count_ones(s & ((1 << (k - 1)) - 1))
            D[s ⊻ (1 << (k - 1)) + 1, s + 1] = sgn
        end
        return D
    end
    c2, c4 = ann(2), ann(4)
    H += Δ * (c2' * c4') + conj(Δ) * (c4 * c2)  # Δ(c†↑c†↓ + h.c.)
    return Hermitian(H)
end

@testset "exact_bcs 严格装配（Nambu/BdG）" begin

    @testset "T1 Fock ED 定义级锚定（Δt）" begin
        N, t = 3, 0.6
        # ED 与装配必须同一 h₊：GL-1 离散的 (V, E) 直接取自 _quad_nodes
        ε1, V1, _ = IFD._quad_nodes(f, 0.0, 1)
        ε, V = ε1[1], V1[1]
        for Δ in (0.3, 0.2 + 0.15im)
            H = bcs_ed_h(ε, V, Δ)
            vals, vecs = eigen(H)
            ρ = vecs * Diagonal(exp.(-β * vals)) * vecs' / sum(exp.(-β * vals))
            ann(k) = begin
                D = zeros(ComplexF64, 16, 16)
                for s in 0:15
                    (s >> (k - 1)) & 1 == 1 || continue
                    sgn = (-1)^count_ones(s & ((1 << (k - 1)) - 1))
                    D[s ⊻ (1 << (k - 1)) + 1, s + 1] = sgn
                end
                return D
            end
            dup, ddn = ann(1), ann(3)              # d↑（槽1）、d↓（模式 3）
            Heis(τ) = vecs * Diagonal(exp.(-im * vals * τ)) * vecs'
            ex = exact_bcs_Δt(f; β=β, N=N, t=t, Δ=Δ, gfsolver=EigenSolver(1))
            n1 = N + 1
            posf(i) = i; posb(i) = N + (N - i) + 1
            # Nambu 槽位物理算符：Φ₁ = d↑（湮没）、Φ₂ = d†↓（产生）——
            # 条目 (p,q) = ⟨T_C Φ_p(t_v) Φ_q†(t_w)⟩：
            # (1,1)=⟨d↑d↑†⟩、(2,2)=⟨d†↓d↓⟩、(1,2)=⟨d↑d↓⟩、(2,1)=⟨d†↓d†↑⟩
            slotop(p, dagger) = p == 1 ? (dagger ? dup' : dup) :
                                        (dagger ? ddn : ddn')
            function edC(v::Int, w::Int, p::Int, q::Int)
                pv = v <= n1 ? posf(v - 1) : posb(v - n1 - 1)
                pw = w <= n1 ? posf(w - 1) : posb(w - n1 - 1)
                # 反支 i 的物理时间 = (N−i)δt（i=0 为转折点 t）
                tv = v <= n1 ? (v - 1) * t / N : (N - (v - n1 - 1)) * t / N
                tw = w <= n1 ? (w - 1) * t / N : (N - (w - n1 - 1)) * t / N
                A = Heis(tv)' * slotop(p, false) * Heis(tv)    # Φ_p(t_v)
                B = Heis(tw)' * slotop(q, true) * Heis(tw)     # Φ_q†(t_w)
                if pv >= pw
                    return tr(ρ * A * B)                       # ⟨Φ_p(t_v) Φ_q†(t_w)⟩
                else
                    return -tr(ρ * B * A)                      # 反对易翻号
                end
            end
            worst = 0.0
            for (blk, p, q) in ((ex.uu, 1, 1), (ex.dd, 2, 2), (ex.ud, 1, 2), (ex.du, 2, 1))
                for i in 0:N, j in 0:N
                    e11 = edC(n1 + j + 1, i + 1, p, q)
                    worst = max(worst, abs(blk.G₊₋[j+1, i+1] - e11))
                    e22 = edC(j + 1, n1 + i + 1, p, q)
                    worst = max(worst, abs(blk.G₋₊[j+1, i+1] - e22))
                end
            end
            @test worst < 1e-12
        end
    end

    @testset "T2 Δ=0 退化：环 = 费米杂质 GF 闭式" begin
        ex = exact_bcs_Δτ(f; β=β, N=6, Δ=0.0, gfsolver=EigenSolver(48))
        Es, Vsq, _ = IFD._quad_nodes(f, 0.0, 48)
        n = 49
        h0 = zeros(ComplexF64, n, n)
        for k in 1:48
            h0[1, 1+k] = Vsq[k]; h0[1+k, 1] = conj(Vsq[k]); h0[1+k, 1+k] = Es[k]
        end
        ev = eigen(Hermitian(h0))
        a2 = abs2.(ev.vectors[1, :])
        G0(τ) = τ > 0 ? sum(a2[p] * exp(-ev.values[p] * τ) / (1 + exp(-β * ev.values[p]))
                            for p in 1:n) :
                -sum(a2[p] * exp(-ev.values[p] * (τ + β)) / (1 + exp(-β * ev.values[p]))
                     for p in 1:n)
        worst = maximum(abs.(ex.uu.data.ηⱼₖ .- [G0((d - 1) * β / 6) for d in 1:6]))
        @test worst < 1e-12
    end

    @testset "T3 V→0：⟨dd†⟩→½、反常→0" begin
        ex = exact_bcs_Δτ(bounded(ω -> 0.0, -Wb, Wb); β=β, N=6, Δ=0.3, gfsolver=EigenSolver(16))
        @test abs(ex.uu.data.ηₖⱼ[1] + 0.5) < 1e-12        # G(0⁻) = −1−G(β⁻) → −½
        @test maximum(abs.(ex.uu.data.ηⱼₖ[2:end] .- 0.5)) < 1e-12
        @test maximum(abs.(ex.ud.data.ηⱼₖ)) < 1e-12
        @test maximum(abs.(ex.du.data.ηⱼₖ)) < 1e-12
    end

    @testset "T4 μ≠0 ⟹ ArgumentError" begin
        @test_throws ArgumentError exact_bcs_Δτ(f; β=β, N=6, Δ=0.3, μ=0.3)
        @test_throws ArgumentError exact_bcs_Δt(f; β=β, N=6, t=0.8, Δ=0.3, μ=0.3)
        @test_throws ArgumentError exact_bcs_Δm(f; β=β, Nτ=5, t=0.6, Nt=3, Δ=0.3, μ=0.3)
    end

end
println("exact_bcs 严格装配检验全部通过 ✓")
