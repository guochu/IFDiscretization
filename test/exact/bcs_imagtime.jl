# =============================================================================
# exact/bcs_imagtime.jl —— BCS 虚时 ExactEqualDistantKernel 金标准
#
# 定义级锚定（主断言）：exact_bcs_Δτ 的四分量环 = 16 维 BCS Fock ED 的
# Matsubara 收缩 ⟨T_τ Φ_p(τ)Φ_q†(0)⟩（Φ = (d↑, d†↓)），实/复 Δ 逐位：
#   ηⱼₖ[1] = −𝒢(β⁻)（环对角 KMS 像，δ 源约定）、ηⱼₖ[d≥2] = 𝒢(τ_d)、
#   ηₖⱼ[1] = 0（对角由 ηⱼₖ[1] 独担——CorrelationMatrix 求和读法，冻结核
#   同款）、ηₖⱼ[d≥2] = −𝒢(β−τ_d)（KMS 上三角）。
# 结构检查：
#   Δ=0 退化 —— uu 环 = 费米杂质 GF 闭式（h₀ 本征）；dd = −uuᵀ（洞侧
#   Nambu(2,2) partner，含对角接触）；δ 源与冻结契约同款。
#   V→0    —— 解耦极限 ⟨dd†⟩ → ½、反常 → 0。
#   μ≠0    —— ArgumentError（BCS 只支持 μ=0）。
#   dispatch —— Δτ(bath; scheme=ExactEqualDistantKernel) == 直接调用（δτ 透传）。
# =============================================================================

include(joinpath(@__DIR__, "_bcs_ed_utils.jl"))

const β = 2.0
const Wb = 3.0
f = bounded(ω -> 0.15, -Wb, Wb)
edmake = bcs_ed_factory(f, β)

@testset "BCS 虚时 ExactEqualDistantKernel 金标准（exact_bcs_Δτ vs Fock ED）" begin

    # ---- T1 定义锚：四分量环 == ED Matsubara 收缩（实/复 Δ） ----
    @testset "T1 定义锚：环 == Fock ED（1 模，实/复 Δ）" begin
        N = 4
        for Δv in (0.3, 0.2 + 0.15im)
            ed, opH, ρ = edmake(Δv; Nt=1, t=0.1, Nτ=N)
            ex = IFD.exact_bcs_Δτ(f; β=β, N=N, Δ=Δv, gfsolver=IFD.EDToulouseSolver(1))
            worst = 0.0
            for (blk, p, q) in bcs_blocks(ex)
                d = blk.data
                Gβ = tr(ρ * opH((:τ, N + 1); p=p, dag_=false) *
                        opH((:τ, 1); p=q, dag_=true))      # 𝒢_pq(β⁻)
                worst = max(worst, abs(d.ηⱼₖ[1] - (-Gβ)))  # 环对角 = −𝒢(β⁻)
                worst = max(worst, abs(d.ηₖⱼ[1]))          # 对角由 ηⱼₖ[1] 独担
                for i in 2:N
                    worst = max(worst, abs(d.ηⱼₖ[i] - ed((:τ, i), (:τ, 1), p, q)))
                    worst = max(worst, abs(d.ηₖⱼ[i] - ed((:τ, 1), (:τ, i), p, q)))
                end
            end
            @test worst < 1e-12
        end
    end

    # ---- T2 Δ=0 退化：uu = 费米杂质 GF 闭式；dd = −uuᵀ ----
    @testset "T2 Δ=0 退化：uu == 闭式 GF，dd == −uuᵀ" begin
        N = 6
        ex = IFD.exact_bcs_Δτ(f; β=β, N=N, Δ=0.0, gfsolver=IFD.EDToulouseSolver(48))
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
        δτ = β / N
        # 环矩阵读法（对角 = ηⱼₖ[1] + ηₖⱼ[1]，其余按三角）
        ring(c) = [i > j ? c.data.ηⱼₖ[i - j + 1] :
                   i < j ? c.data.ηₖⱼ[j - i + 1] :
                   c.data.ηⱼₖ[1] + c.data.ηₖⱼ[1] for i in 1:N, j in 1:N]
        uu = ring(ex.uu)
        worst_uu = maximum(abs(uu[i, j] - G0((i - j) * δτ)) for i in 1:N, j in 1:N)
        @test worst_uu < 1e-12
        dd = ring(ex.dd)
        # 洞侧 Nambu(2,2) partner：𝒢₂₂(τ) = −𝒢₁₁(−τ) + 正则反对易接触——
        # 非对角 dd = −uuᵀ 精确；对角差 δ 源：dd[i,i] = −uu[i,i] − 1
        off = maximum(abs(dd[i, j] + uu[j, i]) for i in 1:N, j in 1:N if i != j)
        diag = maximum(abs(dd[i, i] + uu[i, i] + 1) for i in 1:N)
        @test off < 1e-12
        @test diag < 1e-12
        @test maximum(abs.(ex.ud.data.ηⱼₖ)) < 1e-12             # 反常消失
        @test maximum(abs.(ex.du.data.ηⱼₖ)) < 1e-12
    end

    # ---- T3 V→0：解耦极限 ----
    @testset "T3 V→0：⟨dd†⟩→½、反常→0" begin
        ex = IFD.exact_bcs_Δτ(bounded(ω -> 0.0, -Wb, Wb); β=β, N=6, Δ=0.3,
                              gfsolver=IFD.EDToulouseSolver(16))
        @test abs(ex.uu.data.ηⱼₖ[1] + 0.5) < 1e-12        # G(0⁻) = −1−G(β⁻) → −½
        @test maximum(abs.(ex.uu.data.ηⱼₖ[2:end] .- 0.5)) < 1e-12
        @test maximum(abs.(ex.ud.data.ηⱼₖ)) < 1e-12
        @test maximum(abs.(ex.du.data.ηⱼₖ)) < 1e-12
    end

    # ---- T4 μ≠0 ⟹ ArgumentError（三入口） ----
    @testset "T4 μ≠0 ⟹ ArgumentError" begin
        @test_throws ArgumentError IFD.exact_bcs_Δτ(f; β=β, N=6, Δ=0.3, μ=0.3)
        @test_throws ArgumentError IFD.exact_bcs_Δt(f; β=β, N=6, t=0.8, Δ=0.3, μ=0.3)
        @test_throws ArgumentError IFD.exact_bcs_Δm(f; β=β, Nτ=5, t=0.6, Nt=3, Δ=0.3, μ=0.3)
    end

    # ---- dispatch：δτ 透传 + 默认一致 ----
    @testset "统一入口：Δτ(bath; scheme) == 直接调用（δτ 透传）" begin
        bC = BCSBath(f, β=β, μ=0.0, Δ=0.4)
        e = IFD.ExactEqualDistantKernel(IFD.EDToulouseSolver(16))
        x1 = IFD.Δτ(bC, N=5, scheme=e)
        y1 = IFD.exact_bcs_Δτ(f, β=β, N=5, Δ=0.4, gfsolver=IFD.EDToulouseSolver(16))
        @test x1.uu == y1.uu && x1.dd == y1.dd && x1.ud == y1.ud && x1.du == y1.du
        x2 = IFD.Δτ(bC, N=5, δτ=β / 5, scheme=e)
        @test x2.uu == y1.uu                              # δτ = β/N 缺省路径
        y2 = IFD.exact_bcs_Δτ(f, β=β, N=5, Δ=0.4, δτ=β / 10,
                              gfsolver=IFD.EDToulouseSolver(16))
        x3 = IFD.Δτ(bC, N=5, δτ=β / 10, scheme=e)
        @test x3.uu == y2.uu && x3.ud == y2.ud            # δτ 旋钮真实生效
    end

end
println("BCS 虚时 ExactEqualDistantKernel 金标准通过 ✓（T1 ED 锚 + T2 退化 + T3/T4/dispatch）")
