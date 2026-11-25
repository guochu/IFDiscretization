# =============================================================================
# exact/bcs_realtime.jl —— BCS 实时 ExactEqualDistantKernel 金标准
#
# exact_bcs_Δt（FormB Nambu 翻倍，2026-11-25 恢复）的包内锚定阶梯：
#
#   T1 定义锚   Δ=0 退化逐位（**对称谱**）：uu/dd == exact_fermionic_Δt
#               （同 f/β/N/t/谱）机器精度（Δ=0 时 BCS 浴 = 两个独立数守恒浴，
#               ↓ 扇区读出 dd = −GA[↓in,↓out] 直读式经此锚钉死）；ud/du ≡ 0。
#   T2 二倍锚   ν_BCS(Δ=0) == ν_ferm² 机器精度（对称谱；BdG 翻倍 det 每对模
#               计两次，det(I+e^{−β𝓗}) = e^{βΣλ}·Z² 的规范形式）。
#   T3 结构     Δ≠0：Pf(G) 非零 / ud,du Δ-连续 / 复 Δ（𝓗 厄米 ⟹ 演化幺正）。
#   T4 守卫     μ≠0 ArgumentError；零谱 ArgumentError；离散谱直取。
#   T5 参照自检 退役 raw 收缩闭式（_bcs_raw_reference.jl，D4 裁决）复活
#               旧 T1：四块 × 四支对 × 全网格 == 16 维 Fock ED 轮廓收缩
#               （实/复 Δ）——参照面自身有效，为 GTEMPO 消费面立项保留锚机器。
#   T6 分派     Δt(BCSBath; scheme=ExactEqualDistantKernel(s)) == 直连 [1] 逐位。
#
# ⚠️ 已知限制（不静默，见 src/bcs/exact/realtime.jl 头注 ② 与 README todolist）：
#   ① **Δ=0 费米退化当前仅在镜像简并谱下成立**（单模 ε=0 / 等权对称 GL）；
#      一般离散谱（非对称 ε 或不等权重，如 Dirac ε≠0 / 对称 ε 不等权）下
#      dd 与 ν_BCS 偏离费米退化 O(0.006–0.04)——根因 = 热边取实时 BdG 形
#      e^{−β𝓗_B}（空穴扇区携 e^{+βε} 的 Nambu 雅可比记账），物理闭合边应为
#      BCS 玻尔兹曼核的 Gaussian（镜像简并时两者重合）。修复路线见
#      plan rev 2（GaussianStates purify 式玻尔兹曼核推导）——这是核可用于
#      一般谱的前置。uu 全谱逐位锚定 ✓（实时缝合成分为准）。
#   ② ED 端到端证书（带踢 Z≡1 + 门序列 n̄ 对拍 + 阶数标度）依赖 BCS 消费
#      契约定义（GTEMPO exact-BCS 消费面立项）；异常块（ud/du）取向的
#      数值终裁随该立项闭环。本文件锁定消费无关层。
#
# 铁律：参照侧（ED 链 / raw 参照）禁任何谱恒等式加速；全套件单线程。
# =============================================================================

include(joinpath(@__DIR__, "_bcs_ed_utils.jl"))
include(joinpath(@__DIR__, "_bcs_raw_reference.jl"))

const β = 2.0

@testset "BCS 实时 ExactEqualDistantKernel 金标准（FormB Nambu 翻倍）" begin

    f1 = bounded(ω -> 0.15, -3.0, 3.0)             # GL-1（单节点 ε=0，对称）
    fsym = DiscreteSpectrum([-1.3, 1.3], [0.2025, 0.36])    # 对称双线（K=2）
    f2 = bounded(ω -> 1.0, -1.0, 1.0)              # GL-2（对称）
    fd = DiracDelta(ω=0.8, α=0.4)                  # 非对称单模（限制用例，见头注①）

    # ---- T1 定义锚：Δ=0 退化逐位 == 费米 FormB 核（镜像简并谱，见头注①） ----
    @testset "T1 Δ=0 定义锚（镜像简并谱）：uu/dd == 费米核逐位、ud/du ≡ 0" begin
        for (f, M) in ((f1, 1), (f2, 2)), (N, t) in ((2, 0.6), (3, 0.6), (4, 0.8))
            kern, nd = exact_bcs_Δt(f; β=β, N=N, t=t, Δ=0.0,
                                    gfsolver=IFD.EDToulouseSolver(M))
            fkern, fnd = exact_fermionic_Δt(f; β=β, N=N, t=t, μ=0,
                                            gfsolver=IFD.EDToulouseSolver(M))
            for b1 in (:+, :-), b2 in (:+, :-)
                @test maximum(abs.(branch(kern.uu, b1, b2) - branch(fkern, b1, b2))) < 1e-12
                @test maximum(abs.(branch(kern.dd, b1, b2) - branch(fkern, b1, b2))) < 1e-12
                @test maximum(abs.(branch(kern.ud, b1, b2))) == 0.0
                @test maximum(abs.(branch(kern.du, b1, b2))) == 0.0
            end
        end
        # 一般离散谱（对称 ε、不等权重）：Δ=0 下结构成立（uu 锚、ud/du 零、
        # rank/Pf 非 0）——dd 与 ν 的费米退化属热边开口（头注①），不作断言
        kern, nd = exact_bcs_Δt(fsym; β=β, N=3, t=0.6, Δ=0.0,
                                gfsolver=IFD.EDToulouseSolver(2))
        fkern, _ = exact_fermionic_Δt(fsym; β=β, N=3, t=0.6, μ=0,
                                      gfsolver=IFD.EDToulouseSolver(2))
        @test maximum(abs.(branch(kern.uu, :+, :+) - branch(fkern, :+, :+))) < 1e-12
        @test maximum(abs.(branch(kern.ud, :+, :+))) == 0.0
        @test maximum(abs.(branch(kern.du, :+, :+))) == 0.0
        @test abs(nd) > 1e-12
    end

    # ---- T2 二倍锚：ν_BCS(Δ=0) == ν_ferm²（镜像简并谱，机器精度） ----
    @testset "T2 二倍锚：ν_BCS(Δ=0) == ν_ferm²（镜像简并谱）" begin
        for (f, M) in ((f1, 1), (f2, 2)), (N, t) in ((2, 0.6), (3, 0.6))
            _, nd = exact_bcs_Δt(f; β=β, N=N, t=t, Δ=0.0,
                                 gfsolver=IFD.EDToulouseSolver(M))
            _, fnd = exact_fermionic_Δt(f; β=β, N=N, t=t, μ=0,
                                        gfsolver=IFD.EDToulouseSolver(M))
            @test abs(nd - fnd^2) < 1e-11 * max(1.0, abs(fnd^2))
        end
    end

    # ---- T3 结构：Δ≠0 Pf 非零 / Δ-连续 / 复 Δ ----
    @testset "T3 结构：Pf 非零、Δ-连续、复 Δ" begin
        for Δv in (0.3, 0.2 + 0.15im)
            kern, nd = exact_bcs_Δt(fd; β=β, N=3, t=0.6, Δ=Δv,
                                    gfsolver=IFD.EDToulouseSolver(2))
            @test abs(nd) > 1e-12
            @test all(isfinite, branch(kern.uu, :+, :+))
            @test all(isfinite, branch(kern.ud, :+, :-))
        end
        kern1, _ = exact_bcs_Δt(fd; β=β, N=3, t=0.6, Δ=0.3, gfsolver=IFD.EDToulouseSolver(2))
        kern2, _ = exact_bcs_Δt(fd; β=β, N=3, t=0.6, Δ=0.3 + 1e-7, gfsolver=IFD.EDToulouseSolver(2))
        @test maximum(abs.(branch(kern1.ud, :+, :+) - branch(kern2.ud, :+, :+))) < 1e-5
        @test maximum(abs.(branch(kern1.du, :+, :+) - branch(kern2.du, :+, :+))) < 1e-5
        # 小 Δ 下反常块幅度随 Δ 缩（连续归零的粗界）
        kern3, _ = exact_bcs_Δt(fd; β=β, N=3, t=0.6, Δ=0.001, gfsolver=IFD.EDToulouseSolver(2))
        @test maximum(abs.(branch(kern3.ud, :+, :+))) < 0.05
    end

    # ---- T4 守卫：μ≠0 / 零谱 / 离散谱直取 ----
    @testset "T4 守卫" begin
        @test_throws ArgumentError exact_bcs_Δt(f1; β=β, N=2, t=0.6, Δ=0.3, μ=0.5,
                                                gfsolver=IFD.EDToulouseSolver(2))
        @test_throws ArgumentError exact_bcs_Δt(bounded(ω -> 0.0, -1.0, 1.0); β=β, N=2,
                                                t=0.6, Δ=0.3, gfsolver=IFD.EDToulouseSolver(4))
        # 离散谱直取（不抛 GL 求积错误）
        kern, nd = exact_bcs_Δt(DiracDelta(ω=0.5, α=0.3); β=β, N=2, t=0.6, Δ=0.3,
                                gfsolver=IFD.EDToulouseSolver(8))
        @test abs(nd) > 1e-12
    end

    # ---- T5 参照自检：退役 raw 收缩闭式 == 16 维 Fock ED（旧 T1 复活） ----
    @testset "T5 raw 参照自检：raw 闭式 == Fock ED（1 模，实/复 Δ）" begin
        Wb = 3.0
        fr = bounded(ω -> 0.15, -Wb, Wb)
        edmake = bcs_ed_factory(fr, β)
        N, t = 3, 0.6
        for Δv in (0.3, 0.2 + 0.15im)
            ed, opH, ρ = edmake(Δv; Nt=N, t=t, Nτ=1)
            ex = raw_reference_bcs_Δt(fr; β=β, N=N, t=t, Δ=Δv,
                                      gfsolver=IFD.EDToulouseSolver(1))
            worst = 0.0
            for (blk, p, q) in bcs_blocks(ex), b1 in (:+, :-), b2 in (:+, :-)
                B = branch(blk, b1, b2)
                for i in 0:N, j in 0:N
                    worst = max(worst, abs(B[i+1, j+1] - ed((b1, i), (b2, j), p, q)))
                end
            end
            @test worst < 1e-12
        end
    end

    # ---- T6 分派：统一入口 == 直连 [1] 逐位 ----
    @testset "T6 分派：Δt(BCSBath; scheme=Exact) == 直连" begin
        bC = BCSBath(f2, β=β, μ=0.0, Δ=0.4)
        e = ExactEqualDistantKernel(EDToulouseSolver(8))
        x = Δt(bC, N=3, t=0.6, scheme=e)
        y = exact_bcs_Δt(f2; β=β, N=3, t=0.6, Δ=0.4, gfsolver=EDToulouseSolver(8))
        @test x isa BCSCorrelationFunction{ExactRealCorrelationFunction}
        @test x.uu == y[1].uu && x.dd == y[1].dd && x.ud == y[1].ud && x.du == y[1].du
        # QuAPI 分派不受影响
        q = Δt(bC, N=3, t=0.6)
        @test q isa BCSCorrelationFunction
    end

end
println("BCS 实时金标准通过 ✓（T1/T2 对称谱锚 + T3/T4 结构守卫 + T5 raw 参照自检 + T6 分派；非对称谱热边与 ED 端到端证书 = 已知开口，见头注）")
