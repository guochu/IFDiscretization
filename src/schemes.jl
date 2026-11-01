# schemes.jl —— 离散 IF 算法的类型层级（本包的核心接口层）
#
#   DiscretizationScheme
#   ├── FrozenKernel
#   │   └── QuAPI()                    # 一阶 Trotter 冻结核
#   └── ExactIF(gfsolver)              # 严格核（exact API 实装）
#       └── gfsolver::ToulouseGreenFunctionSolver
#           ├── EigenSolver(nbath)     # 本征路线（exactif 现行）
#           └── AnalyticSolver()       # 解析路线（扩展位）

"""
    abstract type DiscretizationScheme end

离散 IF 算法的抽象基类（虚类）。统一入口 `Δτ / Δt / Δm` 以其末位关键字参数
`scheme` 分派核技术；统计（玻色/费米/BCS）由 bath 对象类型分派。
"""
abstract type DiscretizationScheme end

"""冻结核技术：顶点幅冻结为 Duhamel 低阶项的离散方案。"""
abstract type FrozenKernel <: DiscretizationScheme end

"""
    QuAPI()

一阶 Trotter 冻结核方案（原 QuAPI 包 `bosonic_Δτ/Δt/Δm`、`fermionic_*`、
`bcs_*` 的装配；名字纪念来源包）。
"""
struct QuAPI <: FrozenKernel end

# 后续扩展位：二阶装配（Strang/Weyl-K）等
# struct QuAPI2 <: FrozenKernel end

"""
    abstract type ToulouseGreenFunctionSolver end

Toulouse 型严格核（单杂质能级 + 数守恒浴，U = 0）的格林函数求解器抽象。
exactif 机器的语义边界：杂质模 + 数守恒浴的二次型链。
"""
abstract type ToulouseGreenFunctionSolver end

"""
    EigenSolver(nbath)

本征求解器：把浴 + 杂质单粒子 h₀ 组装为 (nbath+1)×(nbath+1) 矩阵后
`eigen(Hermitian(h0))`，传播子用本征相位闭式（exactif/test/exactif_api.jl
的 `_bath_eig` 路线）。

- `nbath`：浴模式数（连续谱 = Gauss–Legendre–M 求积节点数 M，
  exactif 现默认 M = 64；离散谱 = 谱线数目）。
"""
struct EigenSolver <: ToulouseGreenFunctionSolver
    nbath::Int
end

"""
    AnalyticSolver()

解析求解器（扩展位，待实装）：接 ImpurityModelBase 中 Toulouse 模型的
解析解，作为 EigenSolver 的零求积误差参照/替代。
"""
struct AnalyticSolver <: ToulouseGreenFunctionSolver end

# 其他求解器扩展位：矩阵指数直解（exp(-iΔt·h)）、Krylov / Chebyshev 等。

"求解器 → GL-M 求积节点数（浴模式数）。"
_nbath(s::EigenSolver) = s.nbath
_nbath(::AnalyticSolver) =
    error("AnalyticSolver 待实装：接 ImpurityModelBase Toulouse 解析解")

"""
    ExactIF(gfsolver)
    ExactIF()  # = ExactIF(EigenSolver(64))，exactif 现行默认

严格离散影响泛函核方案，格林函数由内插求解器 `gfsolver` 提供。
exact API 实装自 QuAPI/exactif/test 收编（kernel 装配逐字未动）。
"""
struct ExactIF{G <: ToulouseGreenFunctionSolver} <: DiscretizationScheme
    gfsolver::G
end
ExactIF() = ExactIF(EigenSolver(64))
