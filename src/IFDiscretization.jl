module IFDiscretization

# Our convention is to absorb the negative sign in the exponent of the IF into the hybridization matrix!!!
#
# IFDiscretization —— 离散影响泛函（IF）装配层：以 DiscretizationScheme 虚类封装
# 「离散 IF 怎么算」。统计（玻色/费米/BCS）由 bath 对象类型承担（Δτ/Δt/Δm 的
# 第一个输入必须是 bath），核技术由 scheme 关键字承担：
#   Δτ(bath; N=…, scheme=QuAPI())                    # 一阶 Trotter 冻结核
#   Δτ(bath; N=…, scheme=ExactIF(EigenSolver(64)))   # 严格核（GL-M + 本征求解器）
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

# 严格核（exact API，自 QuAPI/exactif/test 收编；exact_fermionic_* 接替
# 已删除的 ExactGrassmannIF 路线，返回标准 CorrelationFunction）
export ExactIF, exact_fermionic_Δτ, exact_fermionic_Δt, exact_fermionic_Δm
export exact_bosonic_Δτ, exact_bosonic_Δt, exact_bosonic_Δm
export exact_bcs_Δτ, exact_bcs_Δt, exact_bcs_Δm

# 离散方案与格林函数求解器
export DiscretizationScheme, FrozenKernel, QuAPI
export ToulouseGreenFunctionSolver, EigenSolver, AnalyticSolver

# corr → 配对（杂交）矩阵层（GaussianStates 扩展的底座；稠密张量机器留 QuAPI.Dense）
export basis, contour_ordering
export GrassmannHybridizationMatrix, BosonicHybridizationMatrix, BosonicSplitDelta

# numerical tolerance
export get_quapi_tol, set_quapi_tol!

# Gauss–Legendre 节点（自 exactif/volterra.jl 收编）
export gauss_legendre

using Base: @boundscheck
using ImpurityModelBase
using LinearAlgebra

include("defaults.jl")
include("utility.jl")
include("schemes.jl")
include("corrstorages/corrstorages.jl")
include("contourindex.jl")
include("ifmatrix/ifmatrix.jl")

# 三统计 × {quapi 冻结核, exact 严格核} + 统一入口（bath-first 分派，见各 *.jl）
include("bosons/bosons.jl")
include("fermions/fermions.jl")
include("bcs/bcs.jl")          # 依赖 fermions（bcs_Δ* 复用 fermionic_Δ* + disperse）

end
