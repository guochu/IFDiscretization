# =============================================================================
# IFDiscretization 测试套件
#
# 单元测试 = QuAPI 原有测试的移植（不再支持的部分除外）：
#   - api/      correlation-function API 测试（语义/约定 + 小规模物理正确性：
#               自旋-玻色 ED 对拍 spinboson_*、Grassmann-Pfaffian 装配语义
#               bcs_grassmann；infinitetime 测试集留在 QuAPI）
#   - quapi/    冻结核驱动的快速时间演化阶数断言：自由玻色/费米/BCS Toulouse
#               模型上，朴素 face-value 装配 ⇒ 一阶、K/IF 拆分（Weyl-K）⇒ 二阶
#               （9 核全覆盖；目录结构镜像 performance/，求解器为快照拷贝）
#               + 存档逐字节回归
#   - exact/    严格核金标准（实轴）：fermionic_realtime.jl（FormB 契约
#               装配 == 直接 Trotter 逐位）+ bcs_realtime.jl（FormB Nambu
#               翻倍：Δ=0 逐位退化 + ν 二倍锚 + raw 参照自检；ED 端到端
#               证书随 BCS 消费面立项——开口在测试头注声明，不静默）
#               虚轴/混合轴与玻色/BCS 虚混轴严格核退役（README todolist）；
#               退役路由的分派行为（抛「未实现」）由 scheme 分派 testset 钉住
#   - osee.jl   GaussianStates 扩展的 corr→系数态 OSEE 测试（purify/解析锚；
#               稠密张量锚段留在 QuAPI/test/osee.jl）。默认环境（@v1.10）
#               dev 了 GaussianStates 时经 `julia --project=. test/runtests.jl`
#               生效；Pkg.test 隔离环境自动跳过
#   - 末尾两个 testset 为 IFDiscretization 特有：scheme 分派（rev 4 bath-first）、
#               基础设施
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
# 测试的 using/import 集中于本文件：leaf 测试文件（api/ quapi/ exact/）不再
# 自带 using，统一经本文件在 include 树中提供（begin 块不引入新作用域）。
# 例外：osee.jl 的 `using ._IGSExt` 为扩展模块绑定；quapi 快照 common.jl 的
# using 随 performance/ 母本镜像逐字保留（含 Printf）。
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

### frozen-kernel QuAPI：自由模型上的快速时间演化，收敛阶断言
#（目录结构镜像 performance/，求解器为 performance 树的快照拷贝（不跨树
#  include，基准树独立演化）；自旋-玻色 ED 对拍与 Grassmann-Pfaffian 装配
#  语义测试已移 test/api；bcsimag/bcsmixed 的 dense harness 随 QuAPI.Dense
#  留守，bcs_Δτ/Δm 的数值由阶数断言 + 存档回归覆盖）

include("quapi/assert_order.jl")
include("quapi/firstorder/runtests.jl")
include("quapi/secondorder/runtests.jl")

### 严格核金标准：实轴费米（exact_fermionic_Δt 装配 == 同阶直接 Trotter，逐位）
### + BCS 实时（exact_bcs_Δt FormB Nambu 翻倍：Δ=0 逐位退化 + ν 二倍锚 +
###   raw 参照自检；ED 端到端证书随消费面立项）

include("exact/fermionic_realtime.jl")
include("exact/bcs_realtime.jl")

### 存档回归：一阶/二阶装配基准 vs 存档 CSV 逐字节（数值移植的最终防线；
###   存档数据自 QuAPI/test/quapi/archive_* 原样拷入；耗时较长）

include("quapi/archive_regression.jl")

### scheme 分派与基础设施（IFDiscretization 特有）

@testset "统一入口：scheme 分派（rev 4 bath-first；2026-11-25 轴面收敛）" begin
    β = 2.0
    fF = semicircular()
    bF = fermionicbath(fF, β=β, μ=0.3)
    fB = bounded(ω -> exp(-(ω - 1.5)^2 / (2 * 0.4^2)) / sqrt(2π * 0.4^2), 0.5, 3.0)
    bB = bosonicbath(fB, β=β, μ=0.2)
    bC = BCSBath(fF, β=β, μ=0.0, Δ=0.4)
    # 默认 scheme = QuAPI()
    @test IFD.Δτ(bB, N=6) == IFD.Δτ(bB, N=6, scheme=IFD.QuAPI())
    @test IFD.Δt(bF, N=6, t=1.0) == IFD.Δt(bF, N=6, t=1.0, scheme=IFD.QuAPI())
    @test IFD.Δm(bC, Nτ=4, Nt=3, t=0.6).uu == IFD.Δm(bC, Nτ=4, Nt=3, t=0.6, scheme=IFD.QuAPI()).uu  # BCS 容器无 ==，比正常块
    # ExactEqualDistantKernel 默认构造器 = ExactEqualDistantKernel(EDToulouseSolver(64))
    @test IFD.ExactEqualDistantKernel() == IFD.ExactEqualDistantKernel(IFD.EDToulouseSolver(64))
    # 存活严格分派：费米实时（与 direct exact_fermionic_Δt 逐位一致——同一函数同一 gfsolver）
    e = IFD.ExactEqualDistantKernel(IFD.EDToulouseSolver(32))
    xt = IFD.Δt(bF, N=4, t=0.8, scheme=e)
    yt = IFD.exact_fermionic_Δt(fF, β=β, N=4, t=0.8, μ=0.3, gfsolver=IFD.EDToulouseSolver(32))
    @test xt == yt[1]                            # 严格容器 == = 逐块相等
    # 存活严格分派：BCS 实时（2026-11-25 恢复，FormB Nambu 翻倍；金标准见 test/exact/bcs_realtime.jl）
    bC = BCSBath(fF, β=β, μ=0.0, Δ=0.4)
    xtc = IFD.Δt(bC, N=4, t=0.8, scheme=e)
    ytc = IFD.exact_bcs_Δt(fF, β=β, N=4, t=0.8, Δ=0.4, gfsolver=IFD.EDToulouseSolver(32))
    @test xtc.uu == ytc[1].uu && xtc.dd == ytc[1].dd && xtc.ud == ytc[1].ud && xtc.du == ytc[1].du
    # 退役严格分派（2026-11-25 轴面收敛）：统一入口路由抛「未实现」；
    # 恢复入口见 README todolist
    @test_throws ErrorException IFD.Δτ(bF, N=5, scheme=e)   # exact_fermionic_Δτ
    @test_throws ErrorException IFD.Δm(bF, Nτ=3, Nt=3, t=0.6, scheme=e)  # exact_fermionic_Δm
    @test_throws ErrorException IFD.Δτ(bC, N=5, scheme=e)   # exact_bcs_Δτ
    @test_throws ErrorException IFD.Δm(bC, Nτ=3, t=0.6, Nt=3, scheme=e)  # exact_bcs_Δm
    @test_throws ErrorException IFD.Δτ(bB, N=5, scheme=e)   # exact_bosonic_Δτ
    @test_throws ErrorException IFD.Δt(bB, N=4, t=0.8, scheme=e)         # exact_bosonic_Δt
    @test_throws ErrorException IFD.Δm(bB, Nτ=3, Nt=3, t=0.6, scheme=e)  # exact_bosonic_Δm
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
    # AnaToulouseSolver 桩
    @test_throws ErrorException IFD._nbath(IFD.AnaToulouseSolver())
    # ExactEqualDistantKernel 类型层级
    @test IFD.QuAPI() isa IFD.FrozenKernel <: IFD.DiscretizationScheme
    @test IFD.ExactEqualDistantKernel() isa IFD.DiscretizationScheme
    @test !(IFD.ExactEqualDistantKernel() isa IFD.FrozenKernel)
end

end
