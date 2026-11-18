# =============================================================================
# debug/parity_vs_quapi.jl —— 与 QuAPI 的逐位对拍（移植忠实性防线）
#
# IFDiscretization 将来要替代 QuAPI，正式测试套件（test/）不依赖 QuAPI；
# 本脚本是过渡期的交叉验证工具：QuAPI 以限定名共存（IFD.xxx / QuAPI.xxx），
# 三统计 × 三轮廓的冻结核逐一逐位比对（两包同名类型跨模块 == 会回落 ===，
# 必须经 samebits 逐字段比较）；另有腿指标访问器的逐元素对拍（普通/BCS/玻色
# × 三轮廓，含 2026-11-03 BCS 有向指数系数语义）。
#
# 运行：julia --project=. debug/parity_vs_quapi.jl   （IFDiscretization 包根目录）
# =============================================================================
using Test
using LinearAlgebra
using ImpurityModelBase
using IFDiscretization
import IFDiscretization as IFD
import QuAPI

# ---- 跨包逐位比较（不同模块的同名类型之间 == 会回落 ===，必须逐字段比） ----
samebits(a::IFD.CorrelationMatrix, b::QuAPI.CorrelationMatrix) =
    a.ηⱼₖ == b.ηⱼₖ && a.ηₖⱼ == b.ηₖⱼ
samebits(a::IFD.ImagCorrelationFunction, b::QuAPI.ImagCorrelationFunction) =
    samebits(a.data, b.data)
function samebits(a::IFD.RealCorrelationFunction, b::QuAPI.RealCorrelationFunction)
    samebits(a.G₊₋, b.G₊₋) && samebits(a.G₋₊, b.G₋₊)
end
function samebits(a::IFD.MixedCorrelationFunction, b::QuAPI.MixedCorrelationFunction)
    samebits(a.G₊₋, b.G₊₋) && samebits(a.G₋₊, b.G₋₊) &&
        a.G₊τ == b.G₊τ && a.Gτ₊ == b.Gτ₊ && samebits(a.Gττ, b.Gττ)
end
function samebits(a::IFD.BCSCorrelationFunction, b::QuAPI.BCSCorrelationFunction)
    samebits(a.cc, b.cc) && samebits(a.cn, b.cn) &&
        samebits(a.nc, b.nc) && samebits(a.nn, b.nn)
end

β = 2.0
fF = semicircular()                                          # 费米谱（含 ε=0 无碍）
fB = bounded(ω -> exp(-(ω - 1.5)^2 / (2 * 0.4^2)) / sqrt(2π * 0.4^2), 0.5, 3.0)
bF = fermionicbath(fF, β=β, μ=0.3)
bB = bosonicbath(fB, β=β, μ=0.2)
bC = BCSBath(fF, β=β, μ=0.0, Δ=0.4)

@testset "存储容器与冻结核：与 QuAPI 逐位对拍（忠实移植锚）" begin
    # ---- 虚时 ----
    @test samebits(IFD.fermionic_Δτ(fF, β=β, N=10, μ=0.3), QuAPI.fermionic_Δτ(fF, β=β, N=10, μ=0.3))
    @test samebits(IFD.bosonic_Δτ(fB, β=β, N=6, μ=0.2), QuAPI.bosonic_Δτ(fB, β=β, N=6, μ=0.2))
    @test samebits(IFD.bcs_Δτ(fF, β=β, N=8, μ=0.0, Δ=0.4), QuAPI.bcs_Δτ(fF, β=β, N=8, μ=0.0, Δ=0.4))
    @test samebits(IFD.Δτ(bF, N=10), QuAPI.Δτ(bF, N=10))
    @test samebits(IFD.Δτ(bB, N=6), QuAPI.Δτ(bB, N=6))
    @test samebits(IFD.Δτ(bC, N=8), QuAPI.Δτ(bC, N=8))
    # disperse 变体（BCS 冻结核内部路径）
    @test samebits(IFD.fermionic_Δτ(fF, ϵ -> abs(ϵ), β=β, N=6, μ=0.1),
                   QuAPI.fermionic_Δτ(fF, ϵ -> abs(ϵ), β=β, N=6, μ=0.1))
    # ---- 实时 ----
    @test samebits(IFD.fermionic_Δt(fF, β=β, N=6, t=1.2, μ=0.3), QuAPI.fermionic_Δt(fF, β=β, N=6, t=1.2, μ=0.3))
    @test samebits(IFD.bosonic_Δt(fB, β=β, N=5, t=1.2, μ=0.2), QuAPI.bosonic_Δt(fB, β=β, N=5, t=1.2, μ=0.2))
    @test samebits(IFD.bcs_Δt(fF, β=β, N=6, t=1.2, Δ=0.4), QuAPI.bcs_Δt(fF, β=β, N=6, t=1.2, Δ=0.4))
    @test samebits(IFD.Δt(bF, N=6, t=1.2), QuAPI.Δt(bF, N=6, t=1.2))
    @test samebits(IFD.Δt(bB, N=5, t=1.2), QuAPI.Δt(bB, N=5, t=1.2))
    @test samebits(IFD.Δt(bC, N=6, t=1.2), QuAPI.Δt(bC, N=6, t=1.2))
    # ---- 混合 ----
    @test samebits(IFD.fermionic_Δm(fF, β=β, Nτ=5, Nt=4, t=0.8, μ=0.3),
                   QuAPI.fermionic_Δm(fF, β=β, Nτ=5, Nt=4, t=0.8, μ=0.3))
    @test samebits(IFD.bosonic_Δm(fB, β=β, Nτ=4, Nt=3, t=0.8, μ=0.2),
                   QuAPI.bosonic_Δm(fB, β=β, Nτ=4, Nt=3, t=0.8, μ=0.2))
    @test samebits(IFD.bcs_Δm(fF, β=β, Nτ=5, Nt=4, t=0.8, Δ=0.4),
                   QuAPI.bcs_Δm(fF, β=β, Nτ=5, Nt=4, t=0.8, Δ=0.4))
    @test samebits(IFD.Δm(bF, Nτ=5, Nt=4, t=0.8), QuAPI.Δm(bF, Nτ=5, Nt=4, t=0.8))
    @test samebits(IFD.Δm(bB, Nτ=4, Nt=3, t=0.8), QuAPI.Δm(bB, Nτ=4, Nt=3, t=0.8))
    @test samebits(IFD.Δm(bC, Nτ=5, Nt=4, t=0.8), QuAPI.Δm(bC, Nτ=5, Nt=4, t=0.8))
    # ---- 统一入口默认行为 = QuAPI 裸 bath 便捷方法 ----
    @test samebits(IFD.Δτ(bB, N=6), QuAPI.Δτ(bB, N=6))
    @test samebits(IFD.Δτ(bB, N=6, scheme=IFD.QuAPI()), QuAPI.Δτ(bB, N=6))
end

@testset "exact 路线（EigenSolver=64）对拍 exactif 原版语义" begin
    # exactif 原版函数不在 QuAPI 模块内（研究目录），此处以相同谱闭式性质
    # 自检 + 与 IFD.exact_* 直接调用互证（gfsolver 接口同参逐位）。
    e = IFD.ExactIF(IFD.EigenSolver(64))
    @test IFD.Δτ(bB, N=5, scheme=e) ==
          IFD.exact_bosonic_Δτ(fB, β=β, N=5, μ=0.2, gfsolver=IFD.EigenSolver(64))
    @test IFD.Δτ(bF, N=5, scheme=e) ==
          IFD.exact_fermionic_Δτ(fF, β=β, N=5, μ=0.3, gfsolver=IFD.EigenSolver(64))
end

@testset "腿指标访问器：与 QuAPI 逐元素对拍（含 2026-11-03 BCS 有向指数系数语义）" begin
    # 各分支的 j 范围：branch 块的阶（虚时只有 :τ；实时 :+/:− 同长；混合 :τ 短一格系）
    jran(x, b) = 1:length(IFD.branch(x; b1=b, b2=b).ηⱼₖ)
    # 普通费米：三轮廓 × 全腿对逐元素（a-ā 收缩语义：a-ā 取 +Δ、ā-a 取 −Δ、
    # 同 conj 为零；实时顶点位移、τ/mixed raw）
    for (xc, yc) in ((IFD.Δτ(bF, N=6), QuAPI.Δτ(bF, N=6)),
                     (IFD.Δt(bF, N=5, t=1.2), QuAPI.Δt(bF, N=5, t=1.2)),
                     (IFD.Δm(bF, Nτ=4, Nt=3, t=0.8), QuAPI.Δm(bF, Nτ=4, Nt=3, t=0.8)))
        brs = IFD.branches(xc)
        mkc = (b, j, c) -> IFD.GrassmannContourIndex(j, b, 1, c)
        mkq = (b, j, c) -> QuAPI.GrassmannContourIndex(j, b, 1, c)
        for b1 in brs, j1 in jran(xc, b1), c1 in (false, true),
            b2 in brs, j2 in jran(xc, b2), c2 in (false, true)
            @test xc[mkc(b1, j1, c1), mkc(b2, j2, c2)] ==
                  yc[mkq(b1, j1, c1), mkq(b2, j2, c2)]
        end
    end
    # BCS：有向 IF 指数系数（(ā↑,a↑)/(ā↓,a↓)→+uu、(ā↑,ā↓)→+ud、(a↓,a↑)→+du、
    # 误配为零）——虚时/实时（顶点位移）/混合（raw）三轮廓逐元素
    for (xc, yc) in ((IFD.Δτ(bC, N=5, scheme=IFD.QuAPI()), QuAPI.Δτ(bC, N=5)),
                     (IFD.Δt(bC, N=4, t=1.2, scheme=IFD.QuAPI()), QuAPI.Δt(bC, N=4, t=1.2)),
                     (IFD.Δm(bC, Nτ=4, Nt=3, t=0.8, scheme=IFD.QuAPI()), QuAPI.Δm(bC, Nτ=4, Nt=3, t=0.8)))
        brs = IFD.branches(xc)
        mkc = (b, j, band, c) -> IFD.GrassmannContourIndex(j, b, band, c)
        mkq = (b, j, band, c) -> QuAPI.GrassmannContourIndex(j, b, band, c)
        for b1 in brs, j1 in jran(xc.uu, b1), band1 in 1:2, c1 in (false, true),
            b2 in brs, j2 in jran(xc.uu, b2), band2 in 1:2, c2 in (false, true)
            @test xc[mkc(b1, j1, band1, c1), mkc(b2, j2, band2, c2)] ==
                  yc[mkq(b1, j1, band1, c1), mkq(b2, j2, band2, c2)]
        end
    end
    # 玻色（无 band，直读）
    for (xc, yc) in ((IFD.Δτ(bB, N=5), QuAPI.Δτ(bB, N=5)),
                     (IFD.Δt(bB, N=4, t=1.2), QuAPI.Δt(bB, N=4, t=1.2)))
        brs = IFD.branches(xc)
        mkc = (b, j) -> IFD.BosonicContourIndex(j, b)
        mkq = (b, j) -> QuAPI.BosonicContourIndex(j, b)
        for b1 in brs, j1 in jran(xc, b1), b2 in brs, j2 in jran(xc, b2)
            @test xc[mkc(b1, j1), mkc(b2, j2)] == yc[mkq(b1, j1), mkq(b2, j2)]
        end
    end
end

println("parity_vs_quapi 完成")
