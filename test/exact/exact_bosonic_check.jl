# =============================================================================
# exact/exact_bosonic_check.jl —— 玻色严格装配检验
# ← QuAPI/exactif/test/exact_bosonic_check.jl 移植：include 链改 using
#   IFDiscretization（父层），`M=` 关键字改 `gfsolver=EigenSolver(...)`，
#   `_boson_modes` 经 IFD. 限定访问（内部 helper）。
#
# T1  FDT 谱恒等式：G>(t) − G<(t) = Σ_k w_k e^{−iω_k t}（boson 因子相消，
#     存储布局下 = G₊₋.ηₖⱼ − G₊₋.ηⱼₖ 共轭组合）；接触值差 = Σw_k = ∫f
# T2  KMS 周期性（玻色 + 号）：ηₖⱼ[d] = ηⱼₖ 环对位（G(−τ) = G(β−τ) 构造）
# T3  GL-M 收敛：Δτ 条目对照 QuadGK 高精度谱积分，M=8/16/32 单调下降
# T4  零谱 ⟹ 条目精确为零
# T5  μ 双路径自洽：Δt 热移（ω>μ）与 Δτ spectrumshift（ω>0）各自的
#     KMS/FDT 恒等式
# =============================================================================
using Test, LinearAlgebra, QuadGK

Γ0, Wb, β = 0.15, 3.0, 2.0
f = bounded(ω -> Γ0, 0.5, Wb)      # 正带（避开 μ=0 红外奇异）

@testset "exact_bosonic 严格装配" begin

    @testset "T1 FDT 谱恒等式" begin
        M = 64; t = 0.8
        ex = exact_bosonic_Δt(f; β=β, N=8, t=0.8, gfsolver=EigenSolver(M))
        ωs, ws = IFD._boson_modes(f, 0.0, M)
        raw(τ) = sum(w * exp(-im * ω * τ) for (ω, w) in zip(ωs, ws))
        for i in (1, 3, 8)
            τ = (i - 1) * (t / 8)
            # 存储布局：G₊₋.ηₖⱼ = −G>、G₊₋.ηⱼₖ = −conj(G<)
            # G> − G< = −ex.G₊₋.ηₖⱼ − conj(ex.G₊₋.ηⱼₖ) = raw(τ)
            @test abs((-ex.G₊₋.ηₖⱼ[i]) + conj(ex.G₊₋.ηⱼₖ[i]) - raw(τ)) < 1e-13
        end
        # 接触值差 = Σw = 解析 ∫f = Γ0·(带宽)
        @test abs((-ex.G₊₋.ηₖⱼ[1]) + conj(ex.G₊₋.ηⱼₖ[1]) - Γ0 * (Wb - 0.5)) < 1e-10
    end

    @testset "T2 KMS 周期性（玻色 + 号）" begin
        ex = exact_bosonic_Δτ(f; β=β, N=8, gfsolver=EigenSolver(64))
        # 环对位镜像：ηₖⱼ[d] = G(β−(d−1)δτ) = ηⱼₖ[N−d+2]（d≥2）
        @test all(ex.data.ηₖⱼ[d] .== ex.data.ηⱼₖ[8 - d + 2] for d in 2:8)
        # 接触跳变：G(0⁺) − G(β⁻) = Σw（[b,b†] δ 源，调用方簿记）
        @test abs(ex.data.ηⱼₖ[1] - ex.data.ηₖⱼ[1] - Γ0 * (Wb - 0.5)) < 1e-10
    end

    @testset "T3 GL-M 收敛（对照高精度谱积分）" begin
        τ = 1.3
        ref(τ) = quadgk(ε -> Γ0 * (1 + boseeinstein(β, 0, ε)) * exp(-ε * τ),
                        0.5, Wb; rtol=1e-13)[1]
        prev = Inf
        for M in (8, 16, 32)
            ex = exact_bosonic_Δτ(f; β=β, N=4, gfsolver=EigenSolver(M))
            d = abs(ex.data.ηⱼₖ[3] - ref(τ))     # τ = 2δτ = β/2 = 1.0?? 取 N=4: δτ=0.5
            d = abs(ex.data.ηⱼₖ[4] - ref(1.5))   # τ = 3δτ = 1.5
            @test d < prev
            prev = d
        end
        @test prev < 1e-8
    end

    @testset "T4 零谱 ⟹ 条目精确为零" begin
        f0 = bounded(ω -> 0.0, 0.5, Wb)
        eτ = exact_bosonic_Δτ(f0; β=β, N=6, gfsolver=EigenSolver(16))
        @test maximum(abs.(eτ.data.ηⱼₖ)) == 0.0
        et = exact_bosonic_Δt(f0; β=β, N=6, t=0.8, gfsolver=EigenSolver(16))
        @test maximum(abs.(et.G₊₋.ηₖⱼ)) == 0.0
        em = exact_bosonic_Δm(f0; β=β, Nτ=4, t=0.6, Nt=3, gfsolver=EigenSolver(16))
        @test maximum(abs.(em.G₊τ)) == 0.0
    end

    @testset "T5 μ 双路径各自自洽" begin
        μ = 0.2
        et = exact_bosonic_Δt(f; β=β, N=6, t=0.8, μ=μ, gfsolver=EigenSolver(64))
        ωs, ws = IFD._boson_modes(f, μ, 64)
        raw(τ) = sum(w * exp(-im * ω * τ) for (ω, w) in zip(ωs, ws))
        @test abs((-et.G₊₋.ηₖⱼ[4]) + conj(et.G₊₋.ηⱼₖ[4]) - raw((3 * 0.8) / 6)) < 1e-13
        eτ = exact_bosonic_Δτ(f; β=β, N=6, μ=μ, gfsolver=EigenSolver(64))
        @test all(abs(eτ.data.ηₖⱼ[d] - eτ.data.ηⱼₖ[6 - d + 2]) < 1e-12 for d in 2:6)
        em = exact_bosonic_Δm(f; β=β, Nτ=5, t=0.6, Nt=3, μ=μ, gfsolver=EigenSolver(64))
        @test all(abs(em.Gττ.ηₖⱼ[d] - em.Gττ.ηⱼₖ[5 - d + 2]) < 1e-12 for d in 2:5)
    end

end
println("exact_bosonic 严格装配检验全部通过 ✓")
