# IFDiscretization.jl

离散影响泛函（IF）装配层：以 `DiscretizationScheme` 虚类封装「离散 IF 怎么
算」。统计（玻色 / 费米 / BCS）由 bath 对象类型承担（`Δτ / Δt / Δm` 的第一个
输入必须是 bath），核技术由 scheme 关键字承担。由 QuAPI 忠实移植而来。

约定：配对矩阵的指数负号吸收进 hybridization 本身（"Our convention is to
absorb the negative sign in the exponent of the IF"）。

## 统一入口

```julia
using IFDiscretization, ImpurityModelBase

bF = fermionicbath(semicircular(), β = 2.0, μ = 0.3)

Δτ(bF; N = 32)                                  # 虚时（反周期环），δτ = β/N
Δt(bF; N = 16, t = 0.8)                         # 实时闭 Keldysh 轮廓，δt = t/N
Δm(bF; Nτ = 8, Nt = 8, t = 0.8)                 # 混合 Kadanoff–Baym L 轮廓
```

`Δτ/Δt/Δm` 双层分派：第一层 = bath 统计类型（`AbstractFermionicNormalBath` /
玻色 / `BCSCorrelationFunction` 的 bath），第二层 = `scheme::DiscretizationScheme`。

## 离散方案（src/schemes.jl）

```
DiscretizationScheme
├── FrozenKernel
│   └── QuAPI()                            # 一阶 Trotter 冻结核（默认）
└── ExactKernel                            # 严格核抽象族
    └── ExactEqualDistantKernel(solver)    # 等距离散严格核（现行）
        └── solver::ToulouseSolver
            ├── EDToulouseSolver(nbath)    # 本征路线（默认 EDToulouseSolver(64)）
            └── AnaToulouseSolver()        # 解析路线（扩展位，未实装）
```

```julia
Δτ(bF; N = 32, scheme = ExactEqualDistantKernel())          # 严格核
Δτ(bF; N = 32, scheme = ExactEqualDistantKernel(EDToulouseSolver(16)))
```

## 输出容器

| 轮廓 | 冻结核（`QuAPI()`） | 严格核（`ExactEqualDistantKernel()`） |
|---|---|---|
| `Δτ` | `ImagCorrelationFunction` | `(ImagCorrelationFunction, normdelta)` |
| `Δt` | `RealCorrelationFunction` | `(StrictRealCorrelationFunction, normdelta)` |
| `Δm` | `MixedCorrelationFunction` | `(StrictMixedCorrelationFunction, normdelta)` |

- 点值容器（src/corrstorages/）：Toeplitz/分支块的解析关联函数存储，
  `branch(x; b1, b2)` 按统一约定读分支块。
- 严格容器（src/exactcorrstorages/）：装配级**配对矩阵**的稠密分支块字段——
  `StrictRealCorrelationFunction` 四块（G₊₊/G₊₋/G₋₊/G₋₋）、
  `StrictMixedCorrelationFunction` 九块（ττ/τ₊/τ₋/₊τ/₋τ + 四 Keldysh 块）；
  `G[b1,b2][i,j]` = 系数 ā^{b1}_i·a^{b2}_j，节点序 = 冻结核 contour_ordering
  约定。`branch` 切片直取真块（无损——严格 Δ 的块相互独立，点值容器的
  sector 推导不适用，数值定谳见金标准头注）。

## 严格核契约（费米三轮廓，已完结）

严格性机制 = **bath+耦合 cell 的带踢 Schur**：单粒子轮廓矩阵经 Schur 消浴
得配对 Δ；杂质 kick 相位（幺正门 `x = e^{−iα·ε_d}`）经 `kicks` 接口进入
gap 复合矩阵（+ 方向 `D(x)·W` 杂质输出行 ×x，− 方向伴随 `W†D†` 杂质输入列
×x̄）。全因子幺正 ⟹ Z 证书机器精确。

| 入口 | 轮廓 | Z 证书（normdelta 定义） | 金标准（test/exact/） |
|---|---|---|---|
| `exact_fermionic_Δτ` | 反周期环（纯浴，kicks 线性析出 `Δ_IF(x) = x·Δ_IF(1)`） | `Z = det(BB)·Pf(装配)` | fermionic_imagtime.jl（54） |
| `exact_fermionic_Δt` / `_cells` | 闭 Keldysh 开线 | **Z ≡ 1 机器（任意 ε_d/调度）**；`n̄_vac = Z·(1 − A⁻¹[ā⁻_N, a⁺_N])` | fermionic_realtime.jl（58） |
| `exact_fermionic_Δm` | 单环 τ→+→− | Keldysh 段算符精确相消 ⟹ `Z = τ 环踢乘积迹`；`normdelta = (−1)^{n(K+1)}·det(BB)/Z_bath` | fermionic_mixedtime.jl（37） |

- 补零 pair 网格调度（每 gap 一 Trotter 因子）下：order 2 = Strang 门链
  （n̄ 斜率 2.0）、order 1（n̄ 斜率 1.05）——两阶真实分离。
- `exact_fermionic_Δt_cells` 支持逐 cell 调度（0/负 cell）与 `kicksτ` /
  `kickst` 门相位；混合版同（`kicksτ` 长 Nτ 含结合 cell）。
- BB 消元走块结构求解：实时 = 双分支双对角 + 两角块（`det(BB) = det(I + ΦΘΨ)`），
  混合 = 单环（`det(BB) = (−1)^{Kn}·det(I − Π)`），复杂度 O(N·K²·Limp + K³)；
  稠密参照 `_exrt_strict_kernel_dense` / `_exrt_mixed_kernel_dense` 保留
  （机器等价受测）。
- 虚时核的严格性：`Δ_IF(x) = x·Δ_IF(1)` 恒等式（对角 H_imp 的环上 Lie ≡
  Strang，踢相位纯规范），order 无关是结构性的。

## GaussianStates 扩展（ext/）

弱依赖 `GaussianStates` 激活：`FermionicGaussianState(corr)` /
`BosonicGaussianState(corr)` 由 hybridization function 直接构造 IF 系数态
（OSEE 熵割约定见扩展头注）。ifmatrix 稠密装配层已从包核心删除，其机器
（顶点位移装配等）由本扩展自携；`test/osee.jl` 经
`Base.get_extension(IFDiscretization, :IFDiscretizationGaussianStatesExt)`
消费。注意：扩展加载失败会被测试门静默降级——改扩展后须显式跑 osee。

## 测试

```
test/api/        关联函数 API（语义/约定 + spinboson ED 对拍 + Grassmann-Pfaffian 装配语义）
test/osee.jl     GaussianStates 扩展（corr → 系数态 OSEE；无 GaussianStates 跳过）
test/quapi/      冻结核自由模型阶数断言 + 存档逐字节回归（18+1 CSV × orders 1/2）
test/exact/      ExactEqualDistantKernel 金标准（费米三轮廓；见上表）
```

全量：`julia --project=. test/runtests.jl`（当前 9903 绿；勿与其它 julia
进程并行——内存竞态）。

## 包结构

```
src/
├── IFDiscretization.jl        # 导出 + include 树
├── schemes.jl                 # DiscretizationScheme 类型层级
├── corrstorages/              # 点值容器（纯存储，无装配逻辑）
├── exactcorrstorages/         # 严格核配对块容器
├── contourindex.jl            # Grassmann/BosonicContourIndex + branch 访问器
├── bosons/  fermions/  bcs/   # 三统计 × {quapi/ 冻结核, exact/ 严格核} + 统一入口
├── defaults.jl  utility.jl
ext/
└── IFDiscretizationGaussianStatesExt.jl
```

## TODO / 已知状态

- [ ] **玻色严格核（exact_bosonic_Δτ/Δt/Δm）未修复**：现为每模时间域闭式的
      **点值**实现（`CorrelationMatrix` Toeplitz 存储，直接返回点值容器），
      未升级到费米已完结的装配级契约——缺：`(容器, normdelta)` 元组返回、
      kicks 门相位接口、装配级配对块存储（玻色 pairing 结构需单独设计，非
      费米四块的直接推广）、块结构求解、金标准（test/exact/ 无玻色文件；
      现有锚 = GL-M 谱求积收敛，无 Fock ED/直接链机器锚）。注意玻色无
      Nambu 结构、严格性机制（每模位移型闭式 vs 费米 Schur）不同，不能
      照搬费米机器。
- [ ] **BCS 严格核（exact_bcs_Δτ/Δt/Δm）未修复**：现为 Nambu/BdG 时间域
      闭式的**点值**实现（raw 2×2 imp 块装入 `BCSCorrelationFunction`），
      同样缺元组 normdelta、kicks、装配级块容器（BCS 有反常 āā/aa 块，
      容器需在 StrictReal 基础上扩展 band/Nambu 结构）、块结构求解与
      金标准（现有锚 = Fock ED raw 逐位，可直接复用为金标准起点）。
- [ ] 非均匀/逐 cell 调度的 ExactKernel 族扩展位（`cells`/`kicks` 入口已备，
      缺独立 scheme 类型与调度表 API）。
- [ ] 连续谱大规模基准（performance/ 树的 GL-M 基准未覆盖严格核三轮廓）。
- [ ] 外部消费适配：GTEMPO 侧 `quapi_delta` 类适配器读旧 `hmat.Δ/basis`
      接口，需改读严格容器字段。
