# ExactKernel Contract

`ExactKernel` 族（现行 `ExactEqualDistantKernel`）的严格核是**纯浴对象**：
签名与冻结核一致，不含任何杂质信息（无 ε_d、无劈分阶数）——消费侧自行
携带杂质门相位（K 链门 + 接口反测度，见 GTEMPO `exactif_pipeline`）。

**分派面（2026-11-25 轴面收敛；BCS 实时严格核 2026-11-26 换代为粒子字典
FormB 链）**：严格计算支持
**实轴**——费米 `exact_fermionic_Δt` 与 BCS `exact_bcs_Δt`（粒子字典
FormB 链）。虚轴（`exact_fermionic_Δτ`）、混合轴（`exact_fermionic_Δm`）、
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
- 腿基 = 轮廓点 1..N+1 × ± × (a, ā)；仅点 1（``\rho_0`` 端）置零，Δ 支撑 =
  点 2..N+1（点 p+1 ↔ 接口槽 p）；消费者的开放端恒等窗（格点 N+2）在容器
  之外，由消费侧零填充。

## BCS 实轴：粒子字典 FormB 链（2026-11-26 换代）

入口：`exact_bcs_Δt(f; β, N, t, Δ, μ, gfsolver) -> (BCSCorrelationFunction{ExactRealCorrelationFunction}, normdelta)`
（μ 只支持 0；统一入口 `Δt(bath; scheme=ExactEqualDistantKernel(s))` 分派取核）。
前身（2026-11-25 的 FormB Nambu 翻倍链，↓ 味 = 空穴模式 ``X = d^{\downarrow\dagger}``，
只可按 X 契约消费）经用户裁决退役——下游需要冻结核式直接消费。

- **消费契约 = 冻结核字面放置语义**（= GTEMPO AnomalousPairing 四键消费）：
  两味 ``d^\uparrow``/``d^\downarrow`` 皆物理粒子（ρ₀ 真空、两带门相位
  ``e^{-i\varepsilon_d t}``），跨支取负由消费侧 blockification 期收编——下游与
  冻结核同一 PartialIF 装配路径，零特例：

  | 容器字段 | 消费腿对 | 物理含义 |
  |---|---|---|
  | `uu` | ``(\bar a^\uparrow, a^\uparrow)`` | 正常（↑ 味） |
  | `dd` | ``(\bar a^\downarrow, a^\downarrow)`` | 正常（↓ 味） |
  | `ud` | ``(\bar a^\uparrow, \bar a^\downarrow)`` | 对产生 ``d^\uparrow{}^\dagger d^\downarrow{}^\dagger`` |
  | `du` | ``(a^\downarrow, a^\uparrow)`` | 对湮灭 ``d^\downarrow d^\uparrow`` |

- **构造**（全 Gaussian 闭式，对 Fock 精确提取机器精度）：
  1. cell = 物理变量的相干态重叠
     ``\langle -\bar\gamma'|e^{-i\delta t H_{\mathrm{cell}}}|\gamma\rangle = c_0\exp Q``，
     ``Q = \bar\gamma' u\gamma + \tfrac12\bar\gamma' v\bar\gamma' + \tfrac12\gamma w\gamma``
     （``H_{\mathrm{cell}}`` = 星形耦合 + 配对浴
     ``\sum_k[\varepsilon_k(n_{P_k}+n_{Q_k}) + \Delta(P_k^\dagger Q_k^\dagger + \mathrm{h.c.})]``）。
     闭式取 Bogoliubov 矩阵 ``\mathfrak U = \exp(+i\delta t\,[[h,\Delta],[-\bar\Delta,-h^{\mathsf T}]])``
     的块：``u = \overline{\mathfrak U_{22}^{-1}}``、``v = (\mathfrak U_{22}^{-1}\mathfrak U_{21})^\dagger``、
     ``w = (\mathfrak U_{12}\mathfrak U_{22}^{-1})^\dagger``、``|c_0|^2 = \det(\mathbb 1 + vv^\dagger)^{-1/2}``
     （`_formb_gauss_cell`；数守恒极限 ``u = e^{-i\delta t h}``、``c_0 = 1``）。
  2. 热边 = 配对浴 Gibbs 态 double overlap
     ``\langle -\bar\kappa|e^{-\beta H_B}|\kappa\rangle`` 的二次部分（逐对 4×4
     闭式，谱投影子形式数值稳定；无需辅助模式）。
  3. 浴块沿轮廓 = 块三对角（轮廓位置块：outs 2K + ins 2K）+ 热边角块：逐块部分
     主元 LU + Woodbury 的 ``O(N\cdot K^3)`` Schur 求解（`_formb_chain_schur`）。
  4. ``G = \mathrm{antisym}(E + C\,B^{-1}C^{\mathsf T})``，核 = ``-\mathrm{antisym}(G^{-1})``
     逐块读出（conj ↔ eta、plain ↔ bar——与费米核同一规则）。
- **normdelta** ``= \nu\cdot\mathrm{Pf}(G)``，
  ``\nu = |c_0|^{2N}\cdot a_{\mathrm{edge}}\cdot|\mathrm{Pf}(B)|/Z_B``
  （离散链浴配分函数 / 精确浴配分函数：``a_{\mathrm{edge}} = \prod_k\langle 00|e^{-\beta H_k}|00\rangle``、
  ``Z_B = \prod_k \mathrm{Tr}\, e^{-\beta H_k}``）；V→0 极限核平凡、``\nu \to 1``。
- **Δ=0 退化（全部谱）**：BCS 浴 = 两个独立费米浴（``P_k``、``Q_k`` 同在 ``+\varepsilon_k``）：
  `uu` == `dd` == 费米核（谱 ε）逐位，``ud = du \equiv 0``，normdelta = 费米 normdelta²。
- **ED 端到端证书**：冻结核字面放置语义的世界线消费
  （`test/exact/_worldline.jl`，GTEMPO 严格核消费语义的紧凑翻译）的
  ``Z \equiv 1``、``\bar n_\uparrow``、``\bar n_\downarrow``、
  ``\langle d^\downarrow d^\uparrow\rangle``、``\langle d_\uparrow^\dagger d_\downarrow^\dagger\rangle``
  == 多模 Fock ED 门序列链（``4\cdot 4^K`` 维）机器精度：一般谱 × Δ ∈ {0, 实, 复}
  × orders 1/2 × N × ``\varepsilon_d``。

## 金标准（test/exact/）

严格性的机器级证据：装配输出对**直接 Trotter 定义级锚定**（逐位/机器精度）。

| 文件 | 断言层次 |
|---|---|
| `fermionic_realtime.jl` | Z ≡ 1 证书（任意 εd/调度）＋ 门序列 ED 锚 ＋ 大 N ＋ 阶数标度 ＋ FormB 认证参照（`_formb_fixture.jl`，GTEMPO 阶梯 ≤2.5e-15 同源）＋ 结构化↔稠密 ＋ **uniform 核世界线消费证书**（Z≡1 + n̄ == Fock ED 门序列链；非对称谱 × orders 1/2 × 真空/占据初态，`_worldline.jl` + `_fermion_ed_chain.jl`） |
| `bcs_realtime.jl` | **ED 端到端证书（冻结核字面放置语义）**：Z≡1 + n̄↑ + n̄↓ + 反常关联 == 多模 Fock ED（T8）＋ 闭式 cell/热边 == Fock 精确提取（T0）＋ 块三对角 Schur == 稠密（T0c）＋ **结构化装配 == 独立 Fock 稠密参照**（T0d，`_bcs_physical_ref.jl`：Fock cell + Fock 热边 + 稠密 ``B\backslash C^{\mathsf T}``）＋ Δ=0 退化逐位（T1）＋ ν 锚（T2）＋ Δ-连续/复 Δ 结构 ＋ 守卫 ＋ raw 参照自检（== 16 维 Fock ED）＋ 分派 ＋ Pfaffian 主元交换回归（T7） |

退役轮廓/统计的金标准（fermionic_imagtime 123 项、fermionic_mixedtime 37 项、
bcs_imagtime 17 项、bcs_mixedtime 25 项）随实现入 git
历史——恢复时按 README todolist 的恢复前置立项。

## 开放项

- 非均匀/逐 cell 调度的独立 scheme 类型与调度表 API（入口已备）。
- GL-M 连续谱大规模基准（严格核实轴路线）。
- 退役族恢复（虚轴/混合轴/玻色/BCS 虚混轴，见 README todolist 恢复前置）。
