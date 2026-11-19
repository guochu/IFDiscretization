# API

本页列出 `IFDiscretization` 的导出 API。函数签名与说明直接取自源码
docstring；轮廓几何与槽位约定的展开讨论见
[Contours and Containers](@ref "Contours and Containers")，严格核装配契约见
[ExactKernel Contract](@ref "ExactKernel Contract")。

## 统一入口（bath-first 两层分派）

```@docs
Δτ
Δt
Δm
```

## 方案与求解器

```@docs
DiscretizationScheme
FrozenKernel
QuAPI
ExactKernel
ExactEqualDistantKernel
ToulouseSolver
EDToulouseSolver
AnaToulouseSolver
```

## 核心容器与接口

```@docs
CorrelationMatrix
AbstractCorrelationFunction
branch
branches
AbstractContourIndex
GrassmannContourIndex
BosonicContourIndex
```

## 点值容器

```@docs
ImagCorrelationFunction
RealCorrelationFunction
MixedCorrelationFunction
AbstractRealCorrelationFunction
AbstractMixedCorrelationFunction
```

## 冻结核（QuAPI 路线）

```@docs
fermionic_Δτ
bosonic_Δτ
Δiw_to_Δτ
fermionic_Δt
bosonic_Δt
fermionic_Δm
bosonic_Δm
```

## BCS 冻结核

```@docs
BCSCorrelationFunction
bcs_Δτ
bcs_Δt
bcs_Δm
```

## 严格核（ExactKernel 族）

费米三轮廓为元组返回 `(容器, normdelta)`；玻色/BCS 为点值/严格容器
（玻色未升级、BCS 无 normdelta——见 [ExactKernel Contract](@ref "ExactKernel Contract")）。

```@docs
exact_fermionic_Δτ
exact_fermionic_Δt
exact_fermionic_Δm
exact_bosonic_Δτ
exact_bosonic_Δt
exact_bosonic_Δm
exact_bcs_Δτ
exact_bcs_Δt
exact_bcs_Δm
ExactRealCorrelationFunction
ExactMixedCorrelationFunction
IFDiscretization.isize
IFDiscretization.rsize
```

## 工具

```@docs
get_quapi_tol
set_quapi_tol!
gauss_legendre
```

## 内部入口（未导出）

逐 cell 调度 + kicks 门相位、BCS Nambu 数据与稠密参照路径。使用时显式
限定：

```julia
using IFDiscretization
IFDiscretization.exact_fermionic_Δt_cells(f; β, cells, kicks)   # 逐 cell + 门相位
IFDiscretization.bcs_nambu_data(f, Δ; β, gfsolver)              # BdG 本征数据
IFDiscretization._exrt_strict_kernel_dense(...)                 # 实时稠密参照（等价性受测）
IFDiscretization._exrt_mixed_kernel_dense(...)                  # 混合稠密参照（等价性受测）
```

```@docs
IFDiscretization.exact_fermionic_Δt_cells
IFDiscretization.bcs_nambu_data
IFDiscretization.bcs_Gtau
IFDiscretization.bcs_Gtau_neg
```
