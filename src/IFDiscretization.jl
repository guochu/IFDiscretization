module IFDiscretization

# Our convention is to absorb the negative sign in the exponent of the IF into the hybridization matrix!!!
#
# IFDiscretization —— 离散影响泛函（IF）装配层：以 DiscretizationScheme 虚类封装
# 「离散 IF 怎么算」。统计（玻色/费米/BCS）由 bath 对象类型承担（Δτ/Δt/Δm 的
# 第一个输入必须是 bath），核技术由 scheme 关键字承担：
#   Δτ(bath; N=…, scheme=QuAPI())                    # 一阶 Trotter 冻结核
# 严格核（ExactKernel 族）分派面 = 实轴（费米 Δt + BCS Δt）；虚轴/混合轴
# 严格核与玻色严格核（exact_fermionic_Δτ/Δm、exact_bosonic_Δτ/Δt/Δm、
# exact_bcs_Δτ/Δm）已于 2026-11-25 退役——统一入口对其路由抛「未实现」，
# 恢复入口见 README todolist；实轴严格消费 = exact_fermionic_Δt /
# exact_bcs_Δt 直连调用。
#
# 本包由 QuAPI 忠实移植而来（kernel 装配逐字未动）；接口调整见
# plan/migration-plan.md。dense / deprecated / exactfermionic(ExactGrassmannIF)
# / infinitetime 不在移植范围（留在 QuAPI 或按裁决删除）。

export CorrelationMatrix
export AbstractCorrelationFunction, branch, branches
# imaginary time
export ImagCorrelationFunction, Δτ, fermionic_Δτ, bosonic_Δτ, Δiw_to_Δτ
# real time
export AbstractRealCorrelationFunction, RealCorrelationFunction, Δt, fermionic_Δt, bosonic_Δt
# mixed time (L-shaped Kadanoff-Byam contour)
export AbstractMixedCorrelationFunction, MixedCorrelationFunction, Δm, fermionic_Δm, bosonic_Δm

# bcs
export BCSCorrelationFunction, bcs_Δτ, bcs_Δt, bcs_Δm

# contour indices
export AbstractContourIndex, GrassmannContourIndex, BosonicContourIndex

# 严格核（exact API）= **纯耦合项严格核**族（统一契约见 real.jl
# exact_fermionic_Δt docstring：核内只含浴+耦合、无杂质信息；消费收缩 =
# 同调度直接 Trotter 链的迹，任意 ε_d 机器精确）。带踢算符忠实核构造 =
# 测试 helper（test/exact/_kicks_helper.jl、_bcs_kicks_helper.jl），不在包内。
# 2026-11-25 轴面收敛：仅存实轴——费米 exact_fermionic_Δt + BCS
# exact_bcs_Δt（同日 FormB Nambu 翻倍恢复）；虚轴/混合轴与玻色严格核
#（exact_fermionic_Δτ/Δm、exact_bosonic_Δτ/Δt/Δm、exact_bcs_Δτ/Δm）
# 退役——scheme 类型层保留，统一入口路由抛「未实现」，恢复入口见
# README todolist。
export ExactKernel, ExactEqualDistantKernel, exact_fermionic_Δt, exact_bcs_Δt

# 离散方案与格林函数求解器
export DiscretizationScheme, FrozenKernel, QuAPI
export ToulouseSolver, EDToulouseSolver, AnaToulouseSolver

# 严格核装配级载体（ExactKernel 族输出；配对 Δ + 腿基，branch 切片无损）
export ExactRealCorrelationFunction

# numerical tolerance
export get_quapi_tol, set_quapi_tol!

# Gauss–Legendre 节点（自 exactif/volterra.jl 收编）
export gauss_legendre

using Base: @boundscheck
using ImpurityModelBase
using LinearAlgebra

# branch/branches 由 ImpurityModelBase 定义并 export（Green 函数侧）；
# IFDiscretization 对同一函数在相关函数（IF）侧重载，避免双包同名函数歧义。
import ImpurityModelBase: branch, branches

include("defaults.jl")
include("utility.jl")
include("schemes.jl")
include("corrstorages/corrstorages.jl")
include("contourindex.jl")
include("exactcorrstorages/exactcorrstorages.jl")
include("exact/formb.jl")       # FormB 消元原语（费米/BCS 实轴严格核共享，内部符号）

# 三统计 × {quapi 冻结核, exact 严格核} + 统一入口（bath-first 分派，见各 *.jl）
include("bosons/bosons.jl")
include("fermions/fermions.jl")
include("bcs/bcs.jl")          # 依赖 fermions（bcs_Δ* 复用 fermionic_Δ* + disperse）

end
