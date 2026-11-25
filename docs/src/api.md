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

## 严格核（ExactKernel 族；实轴：费米 Δt + BCS Δt）

虚轴/混合轴/玻色严格核与 BCS 虚/混轴严格核已退役（统一入口路由抛「未实现」；
恢复入口见包 README todolist）——机器契约见 [ExactKernel Contract](@ref "ExactKernel Contract")。

```@docs
exact_fermionic_Δt
exact_bcs_Δt
ExactRealCorrelationFunction
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

逐 cell 调度 + kicks 门相位、实时稠密参照路径。使用时显式限定：

```julia
using IFDiscretization
IFDiscretization.exact_fermionic_Δt_cells(f; β, cells, kicks)   # 逐 cell + 门相位
IFDiscretization._exrt_strict_kernel_dense(...)                 # 实时稠密参照（等价性受测）
```

```@docs
IFDiscretization.exact_fermionic_Δt_cells
```
