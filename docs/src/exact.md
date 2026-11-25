# ExactKernel Contract

`ExactKernel` 族（现行 `ExactEqualDistantKernel`）的严格核是**纯浴对象**：
签名与冻结核一致，不含任何杂质信息（无 ε_d、无劈分阶数）——消费侧自行
携带杂质门相位（K 链门 + 接口反测度，见 GTEMPO `exactif_pipeline`）。

**分派面（2026-11-25 轴面收敛 + 同日 BCS 实时恢复）**：严格计算支持
**实轴**——费米 `exact_fermionic_Δt` 与 BCS `exact_bcs_Δt`（FormB Nambu
翻倍）。虚轴（`exact_fermionic_Δτ`）、混合轴（`exact_fermionic_Δm`）、
玻色族（`exact_bosonic_Δτ/Δt/Δm`）与 BCS 虚/混轴（`exact_bcs_Δτ/Δm`）
已退役——统一入口对相应 scheme 路由抛「未实现」；scheme 类型层保留为
恢复挂载点，机器描述与恢复前置见包 README todolist（金标准测试与实现见
git 历史）。

## 费米实轴：FormB 消元（现行唯一严格面）

入口：`exact_fermionic_Δt(f; β, N, t, μ, gfsolver) -> (ExactRealCorrelationFunction, normdelta)`
（均匀网格便捷封装；逐 cell 非均匀调度 + `kicks` 幺正门相位走内部入口
`IFD.exact_fermionic_Δt_cells(f; β, cells, μ, gfsolver, kicks)`，`cells[k]`
可取 0/负）。

- **构造（形态 B / FormB）**：接口槽（c = 1..N × 支 × (bar, eta)）+ bath
  配对（trace_final + forward(N) + backward(N)，每对 (γ̄,γ) 测度 −1）的
  总 Gaussian 经耦合演化 ``W = e^{-i\cdot\mathrm{cell}\cdot h}`` 缝合、热边
  ``e^{-\beta\varepsilon}`` 进 bath 块；Schur 消 bath 块
  ``E + C\cdot(B\backslash C^{\mathsf T})``、反对称化 ``G``、读矩阵
  ``\Delta = -\mathrm{antisym}(G^{-1})``。
- **normdelta = ``\nu\cdot\mathrm{Pf}(G)``**（``\nu`` = 浴链配分比
  ``\det(\mathbb{1}+BRB^{\dagger})/\det(\mathbb{1}+R)``，Pfaffian 走
  Parlett–Reid O(n³)）；装配契约
  ``Z = \mathrm{normdelta}\cdot\text{轮廓收缩}``（比值观测量严格相消）。
- **消费侧严格性**：消费收缩 = 同调度**直接 Trotter 离散链的迹，任意
  ε_d 机器精确**（消费参考 = GTEMPO `exactif_pipeline`）；带踢「算符忠实」
  核的底层 W/Schur 数学以测试 helper（`test/exact/_kicks_helper.jl`）
  持有 Z ≡ 1 锚。
- **真空占据读出**：``\bar n_{\mathrm{vac}} = Z\cdot(1-A^{-1}[\bar a^-_N,\ a^+_N])``。
- **BB 消元走块结构求解**（复杂度 ``O(N\cdot K^2\cdot L_{\mathrm{imp}} + K^3)``）：
  实时 = 双分支双对角 + 两角块
  ``\det(\mathrm{BB}) = \det(\mathbb{1}_K + \Phi\,\Theta\,\Psi)``；稠密参照
  `_exrt_strict_kernel_dense`（测试侧）保留（机器等价性受测，勿用于大规模）。
- 腿基 = 轮廓点 1..N+1 × ± × (a, ā)；点 1（``\rho_0`` 端）与点 N+1（开放端
  恒等窗）置零，Δ 支撑 = 接口点 2..N。

## BCS 实轴：FormB Nambu 翻倍（2026-11-25 恢复）

入口：`exact_bcs_Δt(f; β, N, t, Δ, μ, gfsolver) -> (BCSCorrelationFunction{ExactRealCorrelationFunction}, normdelta)`
（μ 只支持 0，照搬退役守卫；统一入口 `Δt(bath; scheme=ExactEqualDistantKernel(s))` 分派取核）。

- **构造**：费米 FormB 的 Nambu 翻倍——单粒子 ``\mathcal{H} =
  \begin{bmatrix} h_+ & B \\ B^\dagger & -h_+^{\mathsf T} \end{bmatrix}``
  （``B = \Delta\cdot\mathrm{diag}(0,1..1)``，杂质槽不配对）；接口槽每支点
  4 个（``\bar a_\uparrow, a_\uparrow, a_\downarrow, \bar a_\downarrow``）；
  链对测度两扇区同 −1；热边 ``e^{-\beta\mathcal{H}_B}`` 置 (fwd outs ×
  bwd ins)。Schur/Pfaffian/``G^{-1}`` 读出机器与费米共享（`src/exact/formb.jl`）。
- **读出**：(out,in) 型对（uu/ud/du）partner 交叉式 ``-\mathrm{GA}[\mathrm{partner(row)},\mathrm{partner(col)}]``；
  dd（消费指数 ``\bar a_\downarrow`` 在前）直读式。
- **normdelta** ``= \nu_{\mathrm{BCS}}\cdot\mathrm{Pf}(G)``，
  ``\nu_{\mathrm{BCS}} = \det(\mathbb{1}+\mathfrak{B}\mathcal{R}\mathfrak{B}^{\dagger})/\det(\mathbb{1}+\mathcal{R})``。
- **锚定面（test/exact/bcs_realtime.jl）**：Δ=0 退化逐位（镜像简并谱）+
  ``\nu_{\mathrm{BCS}}(\Delta{=}0) = \nu_{\mathrm{ferm}}^2`` 二倍锚 + 退役
  raw 闭式参照自检（== 16 维 Fock ED）+ 分派。
- **已知开口**：①一般离散谱（非对称 ε/不等权）的热边需换 BCS 玻尔兹曼核
  Gaussian（当前 ``e^{-\beta\mathcal{H}_B}`` 携 Nambu 雅可比记账）；
  ②ED 端到端证书（Z≡1 + n̄）随 GTEMPO exact-BCS 消费面立项。二者均不静默，
  见 src 头注与 README todolist。

## 金标准（test/exact/）

严格性的机器级证据：装配输出对**直接 Trotter 定义级锚定**（逐位/机器精度）。

| 文件 | 断言层次 |
|---|---|
| `fermionic_realtime.jl` | Z ≡ 1 证书（任意 εd/调度）＋ 门序列 ED 锚 ＋ 大 N ＋ 阶数标度 ＋ FormB 认证参照（`_formb_fixture.jl`，GTEMPO 阶梯 ≤2.5e-15 同源）＋ 结构化↔稠密 |
| `bcs_realtime.jl` | Δ=0 退化逐位（镜像简并谱）＋ ν 二倍锚 ＋ Δ-连续/复 Δ 结构 ＋ 守卫 ＋ raw 参照自检（== 16 维 Fock ED）＋ 分派（ED 端到端证书随消费面立项，头注声明） |

退役轮廓/统计的金标准（fermionic_imagtime 123 项、fermionic_mixedtime 37 项、
bcs_imagtime 17 项、bcs_mixedtime 25 项）随实现入 git
历史——恢复时按 README todolist 的恢复前置立项。

## 开放项

- 非均匀/逐 cell 调度的独立 scheme 类型与调度表 API（入口已备）。
- GL-M 连续谱大规模基准（严格核实轴路线）。
- 退役族恢复（虚轴/混合轴/玻色/BCS 虚混轴，见 README todolist 恢复前置）。
- BCS 实时开口（见上「已知开口」）：一般谱热边玻尔兹曼核 + ED 端到端证书
  （随 GTEMPO exact-BCS 消费面立项）。
