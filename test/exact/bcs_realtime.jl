# =============================================================================
# exact/bcs_realtime.jl —— BCS 实时 ExactEqualDistantKernel 金标准
#
# 定义级锚定（主断言）：exact_bcs_Δt 的四 Nambu 块 × 四分支对 × 全网格 =
# 16 维 BCS Fock ED 的轮廓收缩 ⟨T_C Φ_p(t_v)Φ_q†(t_w)⟩（Φ = (d↑, d†↓)；
# 物理时间 Heisenberg + 轮廓位排序 + ρ 前置；实/复 Δ）。
#
# 存储约定（本文件钉死，接 test/api/bcs_grassmann.jl 的装配槽位语义）：
#   block(b1,b2)[i,j] = ⟨T_C Φ_p(b1,i)Φ_q†(b2,j)⟩ —— row = Φ_p 侧、
#   col = Φ_q† 侧；等时点走 greater/0⁺ 侧；块存装配级严格容器
#   ExactRealCorrelationFunction（四块独立，branch 无损切片）。
#
# 结构检查：等时恒等 G₊₊[i,i] == G₋₋[i,i]（同一物理时刻的算符）；V→0
#   对角 → ½、反常 → 0；统一入口 dispatch == 直接调用。
# =============================================================================

using Test, LinearAlgebra
using ImpurityModelBase
using IFDiscretization
import IFDiscretization as IFD

include(joinpath(@__DIR__, "_bcs_ed_utils.jl"))

const β = 2.0
const Wb = 3.0
f = bounded(ω -> 0.15, -Wb, Wb)
edmake = bcs_ed_factory(f, β)

@testset "BCS 实时 ExactEqualDistantKernel 金标准（exact_bcs_Δt vs Fock ED）" begin

    # ---- T1 定义锚：四块 × 四分支对 × 全网格（实/复 Δ） ----
    @testset "T1 定义锚：raw 轮廓收缩 == Fock ED（1 模，实/复 Δ）" begin
        N, t = 3, 0.6
        for Δv in (0.3, 0.2 + 0.15im)
            ed, opH, ρ = edmake(Δv; Nt=N, t=t, Nτ=1)
            ex = IFD.exact_bcs_Δt(f; β=β, N=N, t=t, Δ=Δv,
                                  gfsolver=IFD.EDToulouseSolver(1))
            worst = 0.0
            for (blk, p, q) in bcs_blocks(ex), b1 in (:+, :-), b2 in (:+, :-)
                B = IFD.branch(blk, b1, b2)
                for i in 0:N, j in 0:N
                    worst = max(worst, abs(B[i+1, j+1] - ed((b1, i), (b2, j), p, q)))
                end
            end
            @test worst < 1e-12
        end
    end

    # ---- T2 结构：等时恒等（同物理时刻 ⟹ +/− 支算符相同） ----
    @testset "T2 等时恒等：G₊₊[i,i] == G₋₋[i,i]" begin
        ex = IFD.exact_bcs_Δt(f; β=β, N=4, t=0.8, Δ=0.35,
                              gfsolver=IFD.EDToulouseSolver(16))
        for (blk, p, q) in bcs_blocks(ex)
            @test all(isapprox(IFD.branch(blk, :+, :+)[i, i], IFD.branch(blk, :-, :-)[i, i];
                               atol=1e-12) for i in 1:5)
        end
    end

    # ---- T3 V→0：对角 → ½、反常 → 0 ----
    @testset "T3 V→0：⟨dd†⟩→½、反常→0" begin
        ex = IFD.exact_bcs_Δt(bounded(ω -> 0.0, -Wb, Wb); β=β, N=4, t=0.8, Δ=0.3,
                              gfsolver=IFD.EDToulouseSolver(16))
        @test maximum(abs(IFD.branch(ex.uu, :+, :+)[i, i] - 0.5) for i in 1:5) < 1e-12
        @test maximum(abs.(IFD.branch(ex.ud, :+, :+))) < 1e-12
        @test maximum(abs.(IFD.branch(ex.du, :-, :-))) < 1e-12
    end

    # ---- dispatch：统一入口 == 直接调用 ----
    @testset "统一入口：Δt(bath; scheme) == 直接调用" begin
        bC = BCSBath(f, β=β, μ=0.0, Δ=0.4)
        e = IFD.ExactEqualDistantKernel(IFD.EDToulouseSolver(16))
        x = IFD.Δt(bC, N=4, t=0.8, scheme=e)
        y = IFD.exact_bcs_Δt(f, β=β, N=4, t=0.8, Δ=0.4, gfsolver=IFD.EDToulouseSolver(16))
        @test x isa BCSCorrelationFunction{ExactRealCorrelationFunction}
        @test x.uu == y.uu && x.dd == y.dd && x.ud == y.ud && x.du == y.du
    end

end
println("BCS 实时 ExactEqualDistantKernel 金标准通过 ✓（T1 ED 锚 + T2/T3 结构 + dispatch）")
