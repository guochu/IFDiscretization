# schemes.jl —— 离散 IF 算法的类型层级（本包的核心接口层）
#
#   DiscretizationScheme
#   ├── FrozenKernel
#   │   └── QuAPI()                    # 一阶 Trotter 冻结核
#   └── ExactKernel                    # 严格核抽象族
#       └── ExactEqualDistantKernel(solver)   # 等距离散严格核
#           └── solver::ToulouseSolver
#               ├── EDToulouseSolver(nbath)   # 本征路线（现行）
#               └── AnaToulouseSolver()       # 解析路线（扩展位）
#
# 分派面（2026-11-25 轴面收敛 + BCS 实时严格核）：严格核 = 实轴费米
# `exact_fermionic_Δt` + 实轴 BCS `exact_bcs_Δt`（粒子字典 FormB 链）；
# 虚轴/混合轴核（`exact_fermionic_Δτ/Δm`）与玻色严格核
# （`exact_bosonic_Δτ/Δt/Δm`）、BCS 虚/混轴（`exact_bcs_Δτ/Δm`）已退役——
# 统一入口对其 scheme 路由抛「未实现」错误（类型层保留为将来恢复/扩展的
# 挂载点，恢复入口见 README todolist）。

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
    abstract type ToulouseSolver end

Toulouse 型严格核（单杂质能级 + 数守恒浴，U = 0）的格林函数求解器抽象。
严格核机器的语义边界：杂质模 + 数守恒浴的二次型链。
"""
abstract type ToulouseSolver end

"""
    EDToulouseSolver(nbath)

本征求解器：把浴 + 杂质单粒子 h₀ 组装为 (nbath+1)×(nbath+1) 矩阵后
`eigen(Hermitian(h0))`，传播子用本征相位闭式（exactif/test/exactif_api.jl
的 `_bath_eig` 路线）。

- `nbath`：浴模式数（连续谱 = Gauss–Legendre–M 求积节点数 M，
  现行默认 M = 64；离散谱 = 谱线数目）。
"""
struct EDToulouseSolver <: ToulouseSolver
    nbath::Int
end

"""
    AnaToulouseSolver()

解析求解器（扩展位，待实装）：接 ImpurityModelBase 中 Toulouse 模型的
解析解，作为 EDToulouseSolver 的零求积误差参照/替代。
"""
struct AnaToulouseSolver <: ToulouseSolver end

# 其他求解器扩展位：矩阵指数直解（exp(-iΔt·h)）、Krylov / Chebyshev 等。

"求解器 → GL-M 求积节点数（浴模式数）。"
_nbath(s::EDToulouseSolver) = s.nbath
_nbath(::AnaToulouseSolver) =
    error("AnaToulouseSolver 待实装：接 ImpurityModelBase Toulouse 解析解")

"""
    abstract type ExactKernel <: DiscretizationScheme end

严格离散影响泛函核方案抽象族（exact API：`exact_*_Δτ/Δt/Δm` 装配级严格核）。
族内按离散化调度区分：等距格点（现行 ExactEqualDistantKernel）、
非均匀/逐 cell 调度（扩展位）等。
"""
abstract type ExactKernel <: DiscretizationScheme end

"""
    ExactEqualDistantKernel(solver)
    ExactEqualDistantKernel()  # = ExactEqualDistantKernel(EDToulouseSolver(64))，现行默认

等距离散的严格影响泛函核方案，格林函数由内插求解器
`solver` 提供。2026-11-25 轴面收敛后实轴可分派（费米 `Δt` + BCS `Δt`）；
虚轴/混合轴与玻色路由抛「未实现」（恢复入口见
README todolist）。
"""
struct ExactEqualDistantKernel{S <: ToulouseSolver} <: ExactKernel
    solver::S
end
ExactEqualDistantKernel() = ExactEqualDistantKernel(EDToulouseSolver(64))
