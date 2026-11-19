# =============================================================================
# exact/bcs_mixedtime.jl —— BCS 混合 ExactEqualDistantKernel 金标准
#
# 定义级锚定（主断言）：exact_bcs_Δm 的九分支块（ττ 环 + 四 Keldysh + 四 τ
# 交叉）= 16 维 BCS Fock ED 的 L 轮廓收缩 ⟨T_C Φ_p(v)Φ_q†(w)⟩（Φ = (d↑, d†↓)；
# 轮廓 τ→+→−，物理时间 Heisenberg：τ 支虚时、Keldysh 支实时（− 支物理时间
# (Nt−a)δt）+ 轮廓位排序 + ρ 前置；实/复 Δ）。ττ 环对角 = −𝒢(β⁻)（KMS 像）。
#
# 内部一致性：Δm 的 ττ 块 == exact_bcs_Δτ 的环（逐位）；四 Keldysh 块 ==
# exact_bcs_Δt 的对应块（逐位）；δτ 旋钮透传（统一入口 == 直接调用）。
#
# 闭式备忘（src 头注详）：Gτ± 走 lesser 分支（τ 早于一切 Keldysh 点，有界）；
# G±τ 走 greater 闭式 [W(±t)·e^{+𝓗τ}·M₀]（KMS 上半平面延拓；τ∈[0,β] 有界）。
# =============================================================================

include(joinpath(@__DIR__, "_bcs_ed_utils.jl"))

const β = 2.0
const Wb = 3.0
f = bounded(ω -> 0.15, -Wb, Wb)
edmake = bcs_ed_factory(f, β)

@testset "BCS 混合 ExactEqualDistantKernel 金标准（exact_bcs_Δm vs Fock ED）" begin

    # ---- T1 定义锚：九块全网格（实/复 Δ） ----
    @testset "T1 定义锚：九块 == Fock ED L 轮廓收缩（1 模，实/复 Δ）" begin
        Nτ, Nt, t = 3, 2, 0.6
        for Δv in (0.3, 0.2 + 0.15im)
            ed, opH, ρ = edmake(Δv; Nt=Nt, t=t, Nτ=Nτ)
            ex = IFD.exact_bcs_Δm(f; β=β, Nτ=Nτ, t=t, Nt=Nt, Δ=Δv,
                                  gfsolver=IFD.EDToulouseSolver(1))
            worst = 0.0
            for (blk, p, q) in bcs_blocks(ex), b1 in (:τ, :+, :-), b2 in (:τ, :+, :-)
                B = IFD.branch(blk, b1, b2)
                if b1 === :τ && b2 === :τ
                    Gβ = tr(ρ * opH((:τ, Nτ + 1); p=p, dag_=false) *
                            opH((:τ, 1); p=q, dag_=true))
                    for i in 1:Nτ, j in 1:Nτ
                        e = i == j ? -Gβ : ed((:τ, i), (:τ, j), p, q)
                        worst = max(worst, abs(B[i, j] - e))
                    end
                else
                    ii = b1 === :τ ? (1:Nτ) : (1:Nt+1)
                    jj = b2 === :τ ? (1:Nτ) : (1:Nt+1)
                    for i in ii, j in jj
                        v1 = b1 === :τ ? (:τ, i) : (b1, i - 1)
                        w1 = b2 === :τ ? (:τ, j) : (b2, j - 1)
                        worst = max(worst, abs(B[i, j] - ed(v1, w1, p, q)))
                    end
                end
            end
            @test worst < 1e-12
        end
    end

    # ---- T2 内部一致性：ττ == Δτ 环、Keldysh == Δt（逐位） ----
    @testset "T2 内部一致性：ττ == Δτ、Keldysh == Δt（逐位）" begin
        Nτ, Nt, t = 3, 2, 0.6
        exm = IFD.exact_bcs_Δm(f; β=β, Nτ=Nτ, t=t, Nt=Nt, Δ=0.35,
                               gfsolver=IFD.EDToulouseSolver(16))
        exτ = IFD.exact_bcs_Δτ(f; β=β, N=Nτ, Δ=0.35, gfsolver=IFD.EDToulouseSolver(16))
        ext = IFD.exact_bcs_Δt(f; β=β, N=Nt, t=t, Δ=0.35, gfsolver=IFD.EDToulouseSolver(16))
        for (bm, bτ) in ((exm.uu, exτ.uu), (exm.dd, exτ.dd), (exm.ud, exτ.ud), (exm.du, exτ.du))
            @test IFD.branch(bm, :τ, :τ) == IFD.branch(bτ, :τ, :τ)
        end
        for (bm, bt) in ((exm.uu, ext.uu), (exm.dd, ext.dd), (exm.ud, ext.ud), (exm.du, ext.du))
            for b1 in (:+, :-), b2 in (:+, :-)
                @test IFD.branch(bm, b1, b2) == IFD.branch(bt, b1, b2)
            end
        end
    end

    # ---- dispatch：统一入口 == 直接调用（δτ 透传） ----
    @testset "统一入口：Δm(bath; scheme) == 直接调用（δτ 透传）" begin
        bC = BCSBath(f, β=β, μ=0.0, Δ=0.4)
        e = IFD.ExactEqualDistantKernel(IFD.EDToulouseSolver(16))
        x = IFD.Δm(bC, Nτ=3, t=0.6, Nt=3, scheme=e)
        y = IFD.exact_bcs_Δm(f, β=β, Nτ=3, t=0.6, Nt=3, Δ=0.4,
                             gfsolver=IFD.EDToulouseSolver(16))
        @test x isa BCSCorrelationFunction{ExactMixedCorrelationFunction}
        @test x.uu == y.uu && x.dd == y.dd && x.ud == y.ud && x.du == y.du
        x2 = IFD.Δm(bC, Nτ=3, t=0.6, Nt=3, δτ=β / 6, scheme=e)
        y2 = IFD.exact_bcs_Δm(f, β=β, Nτ=3, t=0.6, Nt=3, Δ=0.4, δτ=β / 6,
                              gfsolver=IFD.EDToulouseSolver(16))
        @test x2.uu == y2.uu && x2.ud == y2.ud            # δτ 旋钮真实生效
    end

end
println("BCS 混合 ExactEqualDistantKernel 金标准通过 ✓（T1 ED 锚 + T2 一致性 + dispatch）")
