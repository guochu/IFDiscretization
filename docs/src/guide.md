# Guide

本页给出从谱函数到离散杂化矩阵的典型工作流。所有输入类型由
ImpurityModelBase 提供；约定（负号吸收、槽位语义）见
[Home](@ref "IFDiscretization") 与 [Contours and Containers](@ref "Contours and Containers")。

## 谱函数与 bath

谱函数是支撑上的强度密度 ``A(\varepsilon)\ge 0``
（`AbstractBoundedFunction`）。常用预定义谱：

```julia
using IFDiscretization, ImpurityModelBase

f = semicircular(t = 1.0)                       # Bethe 半圆谱，支撑 [-t, t]
g = bounded(ω -> exp(-(ω - 1.5)^2 / 0.32), 0.5, 3.0)   # 任意支撑上的高斯
δ = DiracDelta(ω = 0.5; α = 0.3)                # 单 δ 峰（α = 谱强度 = V²）
spec = DiscreteSpectrum([1.0, 2.0], [0.25, 0.09])      # 离散模（εₙ, λₙ²）
```

配上温度 / 化学势 / 配对隙打包成 bath（统一入口的第一输入）：

```julia
bF = fermionicbath(f; β = 2.0, μ = 0.3)   # 费米
bB = bosonicbath(g; β = 2.0, μ = 0.2)     # 玻色
bC = BCSBath(f; β = 2.0, μ = 0.0, Δ = 0.4)   # BCS 超导（Δ 可复数；严格核只支持 μ=0）
```

## 统一入口：三种轮廓

```julia
corr = Δτ(bF; N = 32)                       # 虚时反周期环，δτ = β/N
corr = Δt(bF; N = 16, t = 0.8)              # 实时闭 Keldysh 轮廓，δt = t/N
corr = Δm(bF; Nτ = 8, Nt = 8, t = 0.8)      # 混合 Kadanoff–Baym L 形回路
```

返回点值容器；分支块按统一约定读取：

```julia
Gττ = branch(corr; b1 = :τ, b2 = :τ)        # 虚时环 N×N
G₊₋ = branch(corr; b1 = :+, b2 = :-)        # Keldysh 交叉块 (N+1)×(N+1)
G₊τ = branch(corr; b1 = :+, b2 = :τ)        # 混合交叉块 (Nt+1)×Nτ
```

Matsubara 频域数据可直变换到虚时网格：

```julia
Δiw = [1.0 / (im * ω) for ω in ifrequencies(2.0, 64)]
corr = Δiw_to_Δτ(Δiw; β = 2.0, N = 32)
```

## 方案切换：冻结核 ⇄ 严格核（实轴费米）

`scheme` 关键字选择核技术；bath 统计决定走哪个实现（两层分派）。
**2026-11-25 轴面收敛**：严格核分派面 = 实轴费米 `Δt` 一条；其余轮廓/统计的
Exact 路由抛「未实现」（恢复入口见包 README todolist）：

```julia
using IFDiscretization

corr = Δτ(bF; N = 32, scheme = QuAPI())     # 冻结核（默认；现行统一入口唯一严格可用的对照面）
e = ExactEqualDistantKernel()               # = ExactEqualDistantKernel(EDToulouseSolver(64))

# 费米实时：统一入口只返回严格容器；normdelta 经直接调用获取
kc = Δt(bF; N = 16, t = 0.8, scheme = e)    # ExactRealCorrelationFunction（四块严格容器）
G₊₊ = branch(kc; b1 = :+, b2 = :+)          # 严格容器四块无损切片
k, normdelta = exact_fermionic_Δt(f; β = 2.0, N = 16, t = 0.8)   # 直连（元组返回）
```

节点数（求积精度）由求解器携带：`ExactEqualDistantKernel(EDToulouseSolver(16))`
用 GL-16 谱离散；`EDToulouseSolver(64)` 为缺省。解析求解器
`AnaToulouseSolver()` 是扩展位（未实装，构造后抛 `ErrorException`）。

## 直接调用严格核（绕过统一入口）

需要 normdelta / kicks 门相位 / 逐 cell 调度时直接调用
`exact_fermionic_Δt`：

```julia
Δrt, normdelta = exact_fermionic_Δt(f; β = 2.0, N = 16, t = 0.8)
```

纯浴对象（无杂质信息）：杂质 kick 相位经 `kicks` 接口析出，逐 cell 非均匀
调度与幺正门相位走**内部入口**（未导出，限定名消费）：

```julia
kern, ndlt = IFD.exact_fermionic_Δt_cells(f; β = 2.0, cells = fill(0.05, 16),
                                          kicks = [exp(-im * g * ε_d) for g in gates])
```

装配侧的 Z 证书（`Z = normdelta · 轮廓收缩`）与门序列语义见
[ExactKernel Contract](@ref "ExactKernel Contract")。

## BCS 超导 bath

冻结核走 Bogoliubov 相干因子 `(u², uv, vu, v²)·f` 加权 + 色散
``\varepsilon \mapsto E(\varepsilon)=\sqrt{\varepsilon^2+|\Delta|^2}`` 复用
费米核（BCS 严格核 `exact_bcs_Δτ/Δt/Δm` 已于 2026-11-25 退役，恢复入口见
包 README todolist）：

```julia
corr = bcs_Δτ(f; β = 2.0, N = 32, Δ = 0.4, μ = 0.0)      # 冻结核（也可 Δτ(bC; ...)）

corr.uu    # 正常块 ā↑ Δ↑↑ a↑  (= cn)
corr.ud    # 反常块 ā↑ Δ↑↓ ā↓  (= cc)
corr.du    # 反常块 a↓ Δ↓↑ a↑  (= nn)
corr.dd    # 正常 ↓↓（洞侧 Nambu(2,2) partner，非 ā↓a↓ 配对块——见 conventions）
```

块语义与 raw Nambu 收缩槽位见 [Contours and Containers](@ref "Contours and Containers")。

## GaussianStates 扩展

弱依赖 `GaussianStates` 激活：`FermionicGaussianState(corr)` /
`BosonicGaussianState(corr)` 直接由杂化函数构造 IF 系数态（operator-space
纠缠熵入口）。扩展加载失败会被测试门静默降级；OSEE 割约定与稠密交叉验证
见 `test/osee.jl` 头注与 QuAPI 文档的 IF OSEE 页。

## 注意事项

1. **BCS 只支持 μ = 0**：严格核对 μ ≠ 0 抛 `ArgumentError`（Nambu 约定）。
2. **玻色零频奇异性**：μ = 0 且谱支撑触及 ε = 0 时玻色求积抛 `DomainError`。
3. **对角元读法**：[`CorrelationMatrix`](@ref) 对角 = 两排序之和
   `ηⱼₖ[1] + ηₖⱼ[1]`；冻结核把接触约定在 `ηⱼₖ[1]`（`ηₖⱼ[1] = 0`）。
4. **δτ/δt 旋钮**：统一入口与直接调用的 `δτ`（虚时/混合）缺省 `β/Nτ`，
   显式传入时透传到网格/细胞演化。
5. **严格容器的 sector**：严格核的分支块相互独立，点值容器的 sector 推导
   不适用——严格容器 `branch` 无损直取，请勿把严格核的块塞回点值容器。
