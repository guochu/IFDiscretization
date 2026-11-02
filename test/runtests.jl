# =============================================================================
# IFDiscretization 测试套件
#
# 单元测试 = QuAPI 原有测试的移植（不再支持的部分除外）：
#   - api/      correlation-function API 测试（去 *_deprecated 对照块、
#               infinitetime 测试集留在 QuAPI）
#   - quapi/    冻结核数值测试（原样移植；dense 相关仅存在于注释）
#   - exact/    exact API（Toulouse 严格核）自洽测试（← exactif/test；
#               其余 toulouse_*/mx_* 为无断言的标度/诊断脚本，随研究线留 QuAPI）
#   - osee.jl   GaussianStates 扩展的 corr→系数态 OSEE 测试（purify/解析锚；
#               稠密张量锚段留在 QuAPI/test/osee.jl）。默认环境（@v1.10）
#               dev 了 GaussianStates 时经 `julia --project=. test/runtests.jl`
#               生效；Pkg.test 隔离环境自动跳过
#   - 末尾三个 testset 为 IFDiscretization 特有：scheme 分派（rev 4 bath-first）、
#               链内自洽（semicircular 版）、基础设施
# 不移植（2026-11-01 裁决）：
#   - test/dense/*（Dense 不支持）；
#   - test/quapi/exactfermionic.jl、test/quapi/bcsimag.jl、test/quapi/bcsmixed.jl
#    （三者的收缩 harness 基于 QuAPI.Dense 稠密 IF 机器——GrassmannIF/Zvalue/
#     GrassmannADT/BosonicADT 等，随 Dense 留在 QuAPI/test/quapi 继续跑；
#     对应内核的数值由存档回归逐字节钉住：firstorder+secondorder 的
#     fermions/bcs CSV 即出自 fermionic_Δ* 与 bcs_Δ*）；
#   - archive 的 imagtime_dense.csv（Dense 变体基准，第 19 个 CSV 同上）；
#   - test/quapi/archive_regression.jl 的 imagtime_dense 测试集。
# 与 QuAPI 的逐位对拍在 debug/parity_vs_quapi.jl（独立环境，非 Pkg.test 一部分）。
# =============================================================================
using Test
using LinearAlgebra
using ImpurityModelBase
using IFDiscretization
import IFDiscretization as IFD

# isize/rsize 未导出（同 QuAPI 约定），测试显式取用
using IFDiscretization: isize, rsize

# GaussianStates 扩展测试的可用性探测（见文件头注）
const HAVE_GAUSSIANSTATES = try
    @eval using GaussianStates
    @eval using GaussianStates: FermionicGaussianState, BosonicGaussianState,
                               entropy, covariance, nmodes, ispure
    true
catch
    false
end

@testset "IFDiscretization" begin

### correlation-function API tests (Δτ/Δt/Δm, branch conventions, alignment)

include("api/runtests.jl")

### GaussianStates extension: IF coefficient-state OSEE（无 GaussianStates 时跳过）

HAVE_GAUSSIANSTATES && include("osee.jl")

### imaginary-time QuAPI
#（bcsimag.jl / bcsmixed.jl 的 Toulouse ED 比较整段经 QuAPI.Dense Grassmann
#  机器构造参照——dense 不支持，随 Dense 留在 QuAPI/test，不移植）

include("quapi/imag.jl")

### real-time QuAPI

include("quapi/real.jl")
include("quapi/bcsreal.jl")

### mixed-time (Kadanoff contour) QuAPI

include("quapi/mixed.jl")

### exact API (Toulouse 严格核，← exactif/test)

include("exact/api_consistency.jl")
include("exact/api_mixed_check.jl")
include("exact/exact_bosonic_check.jl")
include("exact/exact_bcs_check.jl")
include("exact/trotter_orders.jl")

### 存档回归：一阶/二阶装配基准 vs 存档 CSV 逐字节（数值移植的最终防线；
###   存档数据自 QuAPI/test/quapi/archive_* 原样拷入；耗时较长）

include("quapi/archive_regression.jl")

### scheme 分派与基础设施（IFDiscretization 特有）

@testset "统一入口：scheme 分派（rev 4 bath-first）" begin
    β = 2.0
    fF = semicircular()
    fB = bounded(ω -> exp(-(ω - 1.5)^2 / (2 * 0.4^2)) / sqrt(2π * 0.4^2), 0.5, 3.0)
    bF = fermionicbath(fF, β=β, μ=0.3)
    bB = bosonicbath(fB, β=β, μ=0.2)
    bC = BCSBath(fF, β=β, μ=0.0, Δ=0.4)
    # 默认 scheme = QuAPI()
    @test IFD.Δτ(bB, N=6) == IFD.Δτ(bB, N=6, scheme=IFD.QuAPI())
    @test IFD.Δt(bF, N=6, t=1.0) == IFD.Δt(bF, N=6, t=1.0, scheme=IFD.QuAPI())
    @test IFD.Δm(bC, Nτ=4, Nt=3, t=0.6).uu == IFD.Δm(bC, Nτ=4, Nt=3, t=0.6, scheme=IFD.QuAPI()).uu  # BCS 容器无 ==，比正常块
    # ExactIF 默认构造器 = ExactIF(EigenSolver(64))
    @test IFD.Δτ(bF, N=5, scheme=IFD.ExactIF()) ==
          IFD.Δτ(bF, N=5, scheme=IFD.ExactIF(IFD.EigenSolver(64)))
    # 统计分派：同一 scheme 下三种 bath 各走各的 exact 实现（与直接调用逐位一致）
    e = IFD.ExactIF(IFD.EigenSolver(32))
    @test IFD.Δτ(bB, N=5, scheme=e) ==
          IFD.exact_bosonic_Δτ(fB, β=β, N=5, μ=0.2, gfsolver=IFD.EigenSolver(32))
    @test IFD.Δτ(bF, N=5, scheme=e) ==
          IFD.exact_fermionic_Δτ(fF, β=β, N=5, μ=0.3, gfsolver=IFD.EigenSolver(32))
    xB = IFD.Δτ(bC, N=5, scheme=e)  # BCS 容器无 ==，逐块比（uu/dd 正常块 + ud/du 反常块）
    yB = IFD.exact_bcs_Δτ(fF, β=β, N=5, μ=0.0, Δ=0.4, gfsolver=IFD.EigenSolver(32))
    @test xB.uu == yB.uu && xB.dd == yB.dd && xB.ud == yB.ud && xB.du == yB.du
    @test IFD.Δt(bB, N=4, t=0.8, scheme=e) ==
          IFD.exact_bosonic_Δt(fB, β=β, N=4, t=0.8, μ=0.2, gfsolver=IFD.EigenSolver(32))
    @test IFD.Δm(bF, Nτ=3, Nt=3, t=0.6, scheme=e) ==
          IFD.exact_fermionic_Δm(fF, β=β, Nτ=3, Nt=3, t=0.6, μ=0.3, gfsolver=IFD.EigenSolver(32))
    # 核技术与统计正交：QuAPI() 与 ExactIF() 结果不同（同为 ImagCorrelationFunction）
    @test IFD.Δτ(bF, N=5) != IFD.Δτ(bF, N=5, scheme=e)
end

@testset "exact API：链内自洽（semicircular 版，μ=0 恒等段同原 api_mixed_check）" begin
    β = 2.0
    fF = semicircular()
    M = 48
    # μ=0（Δm↔Δt 交叉块恒等与 Gτ₊ 反对称仅 μ=0 成立，同原 api_mixed_check 参数）
    Δm = IFD.exact_fermionic_Δm(fF, β=β, Nτ=5, t=0.6, Nt=4, μ=0.0, gfsolver=IFD.EigenSolver(M))
    Δt_, _ = IFD.exact_fermionic_Δt(fF, β=β, N=4, t=0.6, μ=0.0, gfsolver=IFD.EigenSolver(M))
    Δτ = IFD.exact_fermionic_Δτ(fF, β=β, N=5, μ=0.0, gfsolver=IFD.EigenSolver(M))
    @test maximum(abs.(Δm.G₊₋ - Δt_.G₊₋)) < 1e-13
    @test maximum(abs.(Δm.G₋₊ - Δt_.G₋₊)) < 1e-13
    @test Δm.Gττ.ηⱼₖ == Δτ.data.ηⱼₖ
    @test Δm.Gττ.ηₖⱼ == Δτ.data.ηₖⱼ
    @test maximum(abs.(Δm.Gτ₊ .+ transpose(Δm.G₊τ))) < 1e-10   # μ=0 反对称
    @test branch(Δm; b1=:-, b2=:τ) == .-Δm.G₊τ
    @test branch(Δm; b1=:τ, b2=:-) == .-Δm.Gτ₊
end

@testset "基础设施" begin
    β = 2.0
    fF = semicircular()
    bF = fermionicbath(fF, β=β, μ=0.3)
    # tol 配置（defaults.jl）
    old = IFD.get_quapi_tol()
    @test IFD.set_quapi_tol!(1e-8) == 1e-8
    @test IFD.get_quapi_tol() == 1e-8
    IFD.set_quapi_tol!(old)
    # Δiw_to_Δτ（utility.jl）：Matsubara → τ 装配可跑且实值
    Δiw = [1.0 / (im * ω) for ω in IFD.ifrequencies(β, 8)]
    d = IFD.Δiw_to_Δτ(Δiw, β=β, N=6)
    @test size(d) == (6, 6)
    # gauss_legendre（utility.jl）：Golub–Welsch 已知性质
    xs, ws = IFD.gauss_legendre(5)
    βv = [0.5 / sqrt(1 - 1 / (2 * k)^2) for k in 1:4]
    F = eigen(SymTridiagonal(zeros(5), βv))
    @test xs ≈ F.values atol=1e-14
    @test ws ≈ 2 .* (F.vectors[1, :] .^ 2) atol=1e-14
    # contourindex：腿指标访问（Grassmann a-ā 取 Δ，ā-a 取 −Δ；Toeplitz 非对称）
    corr = IFD.Δτ(bF, N=4)
    i = IFD.GrassmannContourIndex(2; branch=:τ, conj=false)
    j = IFD.GrassmannContourIndex(3; branch=:τ, conj=true)
    @test corr[i, j] == branch(corr)[2, 3]
    @test corr[j, i] == -branch(corr)[3, 2]
    # AnalyticSolver 桩
    @test_throws ErrorException IFD._nbath(IFD.AnalyticSolver())
    # ExactIF 类型层级
    @test IFD.QuAPI() isa IFD.FrozenKernel <: IFD.DiscretizationScheme
    @test IFD.ExactIF() isa IFD.DiscretizationScheme
    @test !(IFD.ExactIF() isa IFD.FrozenKernel)
end

end
