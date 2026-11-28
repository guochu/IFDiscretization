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

**分派面（2026-11-25 轴面收敛 + 同日 BCS 实时恢复）**：严格核仅存实轴——
费米 `Δt(bF; N, t, scheme = ExactEqualDistantKernel())` 与 BCS
`Δt(bC; N, t, scheme = ExactEqualDistantKernel())` 可用，其余轮廓/统计的
Exact 路由抛「未实现」（退役面与恢复入口见文末 todolist）；实轴严格消费
推荐 `exact_fermionic_Δt` / `exact_bcs_Δt` **直连调用**（返回 `(容器, normdelta)`
元组，不经 scheme 分派）。

## 输出容器

| 轮廓 | 冻结核（`QuAPI()`） | 严格核（实轴） |
|---|---|---|
| `Δτ` | `ImagCorrelationFunction` | 退役 |
| `Δt` | `RealCorrelationFunction` | 费米 `ExactRealCorrelationFunction`、BCS `BCSCorrelationFunction{ExactRealCorrelationFunction}`（直连同返回 normdelta） |
| `Δm` | `MixedCorrelationFunction` | 退役 |

- 点值容器（src/corrstorages/）：Toeplitz/分支块的解析关联函数存储，
  `branch(x; b1, b2)` 按统一约定读分支块。
- 严格容器（src/exactcorrstorages/）：装配级**配对矩阵**的稠密分支块字段——
  `ExactRealCorrelationFunction` 四块（G₊₊/G₊₋/G₋₊/G₋₋）；
  `G[b1,b2][i,j]` = 系数 ā^{b1}_i·a^{b2}_j，节点序 = 冻结核 contour_ordering
  约定。`branch` 切片直取真块（无损——严格 Δ 的块相互独立，点值容器的
  sector 推导不适用，数值定谳见金标准头注）。

## 严格核契约（费米实轴；BCS 实轴契约见下节）

严格性机制 = **bath+耦合 cell 的 FormB 消元**：接口槽 + bath 配对的
总 Gaussian 经耦合演化 W 缝合、热边进 bath 块；Schur 消 bath 块 + 反对称化
得配对 Δ（数值 Pfaffian 出 normdelta）。杂质 kick 相位（幺正门
`x = e^{−iα·ε_d}`）不进核——**纯耦合项契约**（核内无任何杂质信息），
消费侧（K 链门）自行携带。**裸核契约（2026-11-28）**：FormB Schur 装配
吸收的 −1 接口分辨率单位（(+,+)/(-,-) 同支等时对角接触位，幂零像
`exp(−x̄x)`）构造期就地扣除 ⟹ 核 = 物理裸 IF 双线性型（可加）；旧消费侧
接口反测度 `counter_c = +1`（compact 核的对消因子）退役——消费者**不得**
再乘（证据 GTEMPO debug/kernel_measure_subtraction/）。

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

## BCS 严格核（实时；粒子字典；虚/混轴退役）

`exact_bcs_Δt`（粒子字典 FormB 链，2026-11-26 换代；前身 = 2026-11-25 的 Nambu
翻倍链，经用户裁决退役 @ git 82a2368）——**消费契约 = 冻结核字面放置语义**
（= GTEMPO AnomalousPairing 四键消费）：两味 d↑/d↓ 皆物理粒子、ρ₀ 真空，
`uu`→(ā↑,a↑)、`dd`→(ā↓,a↓)、`ud`→(ā↑,ā↓)（对产生）、`du`→(a↓,a↑)（对湮灭）；
下游与冻结核同一 PartialIF 装配路径，零特例。

构造（全 Gaussian 闭式——对 Fock 精确提取机器精度）：① cell = 物理变量的相干态
重叠 ⟨−γ̄'|e^{−iδt·H_cell}|γ⟩ = c₀·exp(γ̄'uγ + ½γ̄'vγ̄' + ½γwγ)（实时轴上为
Gaussian；闭式取 Bogoliubov 矩阵 `𝔘 = exp(+iδt·[[h,Δ],[−Δ̄,−hᵀ]])` 的块：
u = conj(𝔘₂₂⁻¹)、v = (𝔘₂₂⁻¹𝔘₂₁)†、w = (𝔘₁₂𝔘₂₂⁻¹)†）；② 热边 = 配对浴 Gibbs
态 double overlap ⟨−κ̄|e^{−βH_B}|κ⟩ 的二次部分（逐对 4×4 闭式）；③ 浴块沿轮廓
= 块三对角 + 热边角块，逐块 LU + Woodbury 的 O(N·K³) Schur 求解
（K=64、N=20 约 2 s）；④ `G = antisym(E + C·B⁻¹·Cᵀ)`，核 = `−antisym(G⁻¹)`
逐块读出（conj ↔ eta、plain ↔ bar——与费米核同一规则）。normdelta = ν·Pf(G)，
ν = |c₀|^{2N}·a_edge·|Pf(B)|/Z_B（离散链浴配分函数 / 精确浴配分函数）。

**证书**（`test/exact/bcs_realtime.jl`）：冻结核字面放置语义下世界线消费的
Z≡1、n̄↑、n̄↓、⟨d↓d↑⟩、⟨d†↑d†↓⟩ == 多模 Fock ED 门序列链，机器精度
（一般谱 × 实/复 Δ × orders 1/2 × N；T8）；闭式 cell/热边 == Fock 精确提取
（T0）；结构化装配 == **独立** Fock 稠密参照（T0d）。Δ=0 退化：`uu` == `dd` ==
费米核（谱 ε）逐位、`ud = du ≡ 0`、normdelta = 费米 normdelta²；V→0 极限
核平凡、ν→1。`exact_bcs_Δτ/Δm` 仍退役（2026-11-25 轴面收敛）——统一入口的
Exact 路由抛「未实现」。BCS 的 QuAPI 冻结核路线（`bcs_Δτ/Δt/Δm` +
`Δτ/Δt/Δm(bath; scheme=QuAPI())`）**不受影响**。

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
test/exact/      实轴严格核金标准：fermionic_realtime.jl（费米 FormB 契约 +
                 uniform 核世界线消费证书）与 bcs_realtime.jl（BCS 粒子字典核：
                 冻结核字面放置语义的 ED 端到端证书 + 闭式/结构化 ↔ 独立 Fock
                 稠密参照 + Δ=0 退化 + ν 锚 + Pfaffian 回归）；
                 _worldline.jl = GTEMPO 严格核消费语义的世界线求值器，
                 _bcs_ed_chain.jl / _fermion_ed_chain.jl = Fock ED 门序列链参照，
                 _bcs_physical_ref.jl = BCS 严格核的独立 Fock 稠密参照。
                 退役路由的分派行为（抛「未实现」）由 runtests.jl 的 scheme
                 分派 testset 钉住
```

全量：`julia --project=. test/runtests.jl`（当前 12185 绿，单线程约 9–11 分钟——
勿加 `-t`、`BLAS.set_num_threads(1)`、勿与其它 julia 进程并行）。

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
- [ ] **BCS 严格核族 `exact_bcs_Δτ/Δt/Δm`**：实时轴 Δt = 粒子字典 FormB 链
      `exact_bcs_Δt`（2026-11-26 换代；消费契约 = 冻结核字面放置语义 = GTEMPO
      AnomalousPairing 四键消费；前身 = 2026-11-25 Nambu 翻倍链（↓ 味 = 空穴
      模式 X，只可按 X 契约消费），经用户裁决退役 @ git 82a2368；计划 =
      plan/bcs-realtime-exact-kernel-plan.md rev 4）。证书：冻结核字面放置语义下
      Z≡1 / n̄↑ / n̄↓ / 反常关联 == Fock ED 机器精度（一般谱 + 复 Δ +
      orders 1/2）；闭式 cell/热边与结构化 Schur 对独立 Fock 稠密参照机器精度。
      raw 闭式 = 测试参照 helper `_bcs_raw_reference.jl`、16 维 ED 工厂
      `_bcs_ed_utils.jl`。**下游**：GTEMPO 侧 BCS 严格核 blockification +
      `exactif_bcs_pipeline`（2026-11-26 同批接线）。Δτ/Δm 仍退役，恢复前置
      不变（虚轴消费机器立项）。

**其余已知状态**

- [ ] 非均匀/逐 cell 调度的 ExactKernel 族扩展位（`cells`/`kicks` 入口已备，
      缺独立 scheme 类型与调度表 API）。
- [ ] 连续谱大规模基准（performance/ 树的 GL-M 基准未覆盖严格核实轴路线）。
- [ ] 外部消费适配：GTEMPO 侧 `quapi_delta` 类适配器读旧 `hmat.Δ/basis`
      接口，需改读严格容器字段。
- [ ] **`exact_fermionic_Δt` 装配升级：src 稠密 Schur → 块结构求解迁移**
      （2026-11-27 定谳）：src 导出入口现走 FormB 稠密消元
      （`fermions/exact/real.jl` 的 `effective = E + C·(B\Cᵀ)`，O((2KN)³) 时间 /
      O((KN)²) 内存——K=64、N=64 约 4.5 GB A 阵 + 分钟级 LU）；块结构求解
      **已存在且受测**：test/exact/_kicks_helper.jl `_exrt_schur_structured`
      （`exact_fermionic_Δt_cells` 的 BB·X = BI 仿射前推——双分支双对角 +
      热边/闭合边角块、det(BB) 闭式、O(N·K²·Limp + K³)；与稠密参照
      `_exrt_strict_kernel_dense` 机器等价受测），BCS 侧同类原语
      `_formb_chain_schur` 已在 src/exact/formb.jl（头注「原语维度无关」）。
      迁移 = 结构化装配（无 kicks 部分）提升进 src（fermions/exact/），
      `exact_fermionic_Δt` 等距 cells 径改调用（kicks 机器留测试侧不动）；
      验收 = 新径对现行 src 稠密径 kernel 逐块 + normdelta 机器精度、
      金标准阶梯全绿。附带订正（2026-11-27 已统一，结论与初记相反）：
      det(BB) 正确式 = **加号**「det(I + ΦΘΨ)」——契约节与 helper 代码
      （`detBB = det(I + Φ*Θ*Ψ)`，_kicks_helper.jl :186）一致，且被金标准
      「结构化 Schur vs 稠密参照（机器等价）」的 normdelta 1e-9 对拍锚定
      （fermionic_realtime.jl）；减号只在 helper 的 docstring/行内注释
      （笔误，与其自身代码矛盾），已改 :130/:179。
- [ ] **`exact_fermionic_Δt`/`exact_bcs_Δt` 的 normdelta 独立接口（kernel-only /
      normdelta-only 拆分）**（2026-11-27 分析定谳，立项前不动）：现状 =
      `(容器, normdelta)` 元组，normdelta 与 kernel 共享 Schur 中间量 G
      （Δ = −antisym(G⁻¹)、normdelta = ν·Pf(G)）——搭车边际成本可忽略
      （Pf O(L³) + ν O(N·K³) ≈ Schur 主导成本的 10⁻⁴），拆分本身**不省计算**；
      多浴/双 lead 加法亦不需要（GTEMPO `exactif_pipeline` 已按 ∏nd 消费、
      组合核 normdelta = 乘积免费；见 GTEMPO `Base.:+(::Exact, ::Exact)`
      因子化裁决门 src/correlationfunction.jl +
      plan/exactif_port_plan.md Phase B @ git，已删除 2026-11-27）。
      **【2026-11-28 裸核契约闭环，取代 11-27 实施门证伪记录】**FormB compact
      核的对角接触 −1 接口分辨率单位改为**构造期就地扣除**（裸核 = 物理裸
      IF 双线性型；同支等时对角 +1），消费端 counter_c=+1 反测度退役；
      `Base.:+(::Exact,::Exact)` 已实施（blocks 相加 + normdelta 相乘）。
      同带双浴加法 vs 并集谱单构造的残差 = 胞内 [V_A,V_B] 换位 O(δt²)
      同时/顺序耦合约定差（11-27 遗留残差 1.3e-7/1.4e-6 定谳；幂律探针 =
      GTEMPO/debug/kernel_measure_subtraction/）。历史证伪档案 =
      GTEMPO/debug/kernel_addition_plan/。同浴联合谱单构造（并集谱直连本
      入口）仍可用（同时耦合约定）。若为接口纯度立项：拆口在 IFD
      内部（两导出入口共享 FormB 原语；GTEMPO 侧只包 `weylcn` 同款便利
      函数，禁复刻 legpos/模表/ν 约定——两处真源）。normdelta-only 两路：
      ① 推荐 = 浴侧闭式 Π(1−n̄ₖ)·det(I + Φ·diag(θ)·Ψ)（即
      `_exrt_schur_structured` 的 detBB 部分——O(N·K³)、无 Pfaffian、
      无接口 Schur，仅消费浴模 + 调度；依赖 = 上一条迁移先行；两套 Z 闭式
      稠密 ν·Pf(G) vs 结构化 Π(1−n̄)·detBB 的机器等价须显式对拍钉死）；
      ② 备选 = 恒等式路 normdelta = ν/Pf(Δ̃)（Δ̃ = 容器块全反称补全 =
      −G⁻¹；Pf(G) = 1/Pf(Δ̃) 精确无符号歧义；前提 = normal 浴 G⁻¹ 同味块
      严格零、补全规则 = 缺向 −transpose）。裁决均 = 离散谱 + 小 N 对拍
      现行稠密径机器精度；大 N 硬化 = log-Pf（Parlett–Reid 累积 log|主元| +
      符号）。BCS 同构拆法：ν = |c₀|^{2N}·a_edge·|Pf(B)|/Z_B 已是浴侧
      闭式（chain schur 中间量）。
