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
│   └── QuAPI()                            # 一阶 Trotter 冻结核（默认；现行统一入口唯一方案）
└── ExactKernel                            # 严格核抽象族（类型层保留）
    └── ExactEqualDistantKernel(solver)    # 等距离散严格核
        └── solver::ToulouseSolver
            ├── EDToulouseSolver(nbath)    # 本征路线（默认 EDToulouseSolver(64)）
            └── AnaToulouseSolver()        # 解析路线（扩展位，未实装）
```

**分派面（2026-11-25 轴面收敛）**：严格核仅存实轴费米——
`Δt(bF; N, t, scheme = ExactEqualDistantKernel())` 可用，其余轮廓/统计的
Exact 路由抛「未实现」（退役面与恢复入口见文末 todolist）；实轴严格消费
推荐 `exact_fermionic_Δt` **直连调用**（返回 `(容器, normdelta)` 元组，
不经 scheme 分派）。

## 输出容器

| 轮廓 | 冻结核（`QuAPI()`） | 严格核（费米实轴） |
|---|---|---|
| `Δτ` | `ImagCorrelationFunction` | 退役 |
| `Δt` | `RealCorrelationFunction` | `ExactRealCorrelationFunction`（直连同返回 normdelta） |
| `Δm` | `MixedCorrelationFunction` | 退役 |

- 点值容器（src/corrstorages/）：Toeplitz/分支块的解析关联函数存储，
  `branch(x; b1, b2)` 按统一约定读分支块。
- 严格容器（src/exactcorrstorages/）：装配级**配对矩阵**的稠密分支块字段——
  `ExactRealCorrelationFunction` 四块（G₊₊/G₊₋/G₋₊/G₋₋）；
  `G[b1,b2][i,j]` = 系数 ā^{b1}_i·a^{b2}_j，节点序 = 冻结核 contour_ordering
  约定。`branch` 切片直取真块（无损——严格 Δ 的块相互独立，点值容器的
  sector 推导不适用，数值定谳见金标准头注）。

## 严格核契约（费米实轴，现行唯一严格面）

严格性机制 = **bath+耦合 cell 的 FormB 消元**：接口槽 + bath 配对的
总 Gaussian 经耦合演化 W 缝合、热边进 bath 块；Schur 消 bath 块 + 反对称化
得配对 Δ（数值 Pfaffian 出 normdelta）。杂质 kick 相位（幺正门
`x = e^{−iα·ε_d}`）不进核——**纯耦合项契约**（核内无任何杂质信息），
消费侧（K 链门 + 接口反测度 `counter_c = +1`）自行携带。

| 入口 | 轮廓 | Z 证书（normdelta 定义） | 金标准（test/exact/） |
|---|---|---|---|
| `exact_fermionic_Δt`（+ 内部入口 `_cells`，未导出） | 闭 Keldysh 开线 | **Z ≡ 1 机器（任意 ε_d/调度）**；`n̄_vac = Z·(1 − A⁻¹[ā⁻_N, a⁺_N])` | fermionic_realtime.jl（58） |

- 补零 pair 网格调度（每 gap 一 Trotter 因子）下：order 2 = Strang 门链
  （n̄ 斜率 2.0）、order 1（n̄ 斜率 1.05）——两阶真实分离。
- `IFD.exact_fermionic_Δt_cells`（未导出，逐 cell 调度 + `kicks` 门相位的
  内部入口）支持 0/负 cell 与任意相位向量。
- BB 消元走块结构求解：实时 = 双分支双对角 + 两角块
  （`det(BB) = det(I + ΦΘΨ)`），复杂度 O(N·K²·Limp + K³)；稠密参照
  `_exrt_strict_kernel_dense`（测试侧）保留（机器等价受测）。
- 虚轴（`exact_fermionic_Δτ`）与混合轴（`exact_fermionic_Δm`）严格核已退役
  （2026-11-25 轴面收敛）——契约与金标准见文末 todolist。

## BCS 严格核（已退役）

`exact_bcs_Δτ/Δt/Δm`（Nambu/BdG 对角化路线，2026-11-19 金标准完结）已于
2026-11-25 轴面收敛整体退役——统一入口的 Exact 路由抛「未实现」；机器
描述、槽位约定与金标准见文末 todolist 与 git 历史。BCS 的 QuAPI 冻结核
路线（`bcs_Δτ/Δt/Δm` + `Δτ/Δt/Δm(bath; scheme=QuAPI())`）**不受影响**。

## GaussianStates 扩展（ext/）

弱依赖 `GaussianStates` 激活：`FermionicGaussianState(corr)` /
`BosonicGaussianState(corr)` 由 hybridization function 直接构造 IF 系数态
（OSEE 熵割约定见扩展头注）。ifmatrix 稠密装配层已从包核心删除，其机器
（稠密装配等，全轮廓 raw 直构）由本扩展自携；`test/osee.jl` 经
`Base.get_extension(IFDiscretization, :IFDiscretizationGaussianStatesExt)`
消费。注意：扩展加载失败会被测试门静默降级——改扩展后须显式跑 osee。

## 测试

```
test/api/        关联函数 API（语义/约定 + spinboson ED 对拍 + Grassmann-Pfaffian 装配语义）
test/osee.jl     GaussianStates 扩展（corr → 系数态 OSEE；无 GaussianStates 跳过）
test/quapi/      冻结核自由模型阶数断言 + 存档逐字节回归（18+1 CSV × orders 1/2）
test/exact/      实轴费米严格核金标准（fermionic_realtime.jl；退役路由的分派
                 行为（抛「未实现」）由 runtests.jl 的 scheme 分派 testset 钉住）
```

全量：`julia --project=. test/runtests.jl`（当前 9834 绿；勿与其它 julia
进程并行——内存竞态）。

文档（Documenter，镜像 QuAPI 体系）：`julia --project=docs docs/make.jl`
构建到 docs/build/（KaTeX 本地自托管，file:// 可读）；
`julia --project=docs docs/liveserver.jl` 实时预览。

## 包结构

```
src/
├── IFDiscretization.jl        # 导出 + include 树
├── schemes.jl                 # DiscretizationScheme 类型层级
├── corrstorages/              # 点值容器（纯存储，无装配逻辑）
├── exactcorrstorages/         # 严格核配对块容器
├── contourindex.jl            # Grassmann/BosonicContourIndex + branch 访问器
├── bosons/  fermions/  bcs/   # 三统计 quapi/ 冻结核 + 费米 exact/ 严格核（实轴）+ 统一入口
├── defaults.jl  utility.jl
ext/
└── IFDiscretizationGaussianStatesExt.jl
docs/
├── make.jl  Project.toml  liveserver.jl   # Documenter 构建（KaTeX 自托管）
└── src/                                    # 文档源（Home/Guide/API/约定/严格核契约）
```

## TODO / 已知状态

**严格核退役面（2026-11-25 轴面收敛：严格计算只支持实轴）**——下列函数已删除，
统一入口（`Δτ/Δt/Δm`）对相应 `ExactEqualDistantKernel` 路由抛「未实现」；
scheme 类型层（`ExactKernel`/`ExactEqualDistantKernel`）保留为恢复挂载点。
恢复入口 = git 历史（本节条目随退役提交锁定 commit）。

- [ ] **虚时严格核 `exact_fermionic_Δτ`**（谱闭式 Schur O(K³+N·K+N²)；kick 线性
      析出 `Δ_IF(x) = x·Δ_IF(1)` 契约；金标准 = test/exact/fermionic_imagtime.jl
      @ git 历史）。恢复前置 = 严格核虚轴消费机器立项（GTEMPO ⑩ 着装管线 /
      HS 管线见 GTEMPO git）。
- [ ] **混合轮廓严格核 `exact_fermionic_Δm`**（Kadanoff–Baym L 轮廓 FormB；紧凑
      (ā,a) 链序装配契约 = 测度 −1 + 单位链 + 两条闭合边；⑪ broken 锚诊断史见
      GTEMPO plan/exactif_port_plan.md §7.2.11 @ git）。恢复前置 = 混合严格
      K 链机器立项。
- [ ] **玻色严格核族 `exact_bosonic_Δτ/Δt/Δm`**（每模时间域闭式的**点值**实现，
      `CorrelationMatrix` Toeplitz 存储；未升级装配级契约——缺 `(容器, normdelta)`
      元组返回、kicks 门相位接口、装配级配对块存储（玻色 pairing 结构需单独
      设计，非费米四块的直接推广）、块结构求解、金标准；玻色无 Nambu 结构、
      严格性机制（每模位移型闭式 vs 费米 Schur）不同，不能照搬费米机器）。
      恢复前置 = 玻色装配级契约设计 + 消费面立项。
- [ ] **BCS 严格核族 `exact_bcs_Δτ/Δt/Δm`**（Nambu/BdG 对角化三轮廓时间域闭式，
      机器精度；槽位约定 = raw 收缩 ⟨T_C Φ_p(b1,i)Φ_q†(b2,j)⟩；16 维 Fock ED
      定义级金标准 = test/exact/bcs_{imagtime,realtime,mixedtime}.jl @ git
      历史；normdelta/kicks 接口留白待立项）。恢复前置 = normdelta 闭式 +
      BCS 严格消费面立项。

**其余已知状态**

- [ ] 非均匀/逐 cell 调度的 ExactKernel 族扩展位（`cells`/`kicks` 入口已备，
      缺独立 scheme 类型与调度表 API）。
- [ ] 连续谱大规模基准（performance/ 树的 GL-M 基准未覆盖严格核实轴路线）。
- [ ] 外部消费适配：GTEMPO 侧 `quapi_delta` 类适配器读旧 `hmat.Δ/basis`
      接口，需改读严格容器字段。
