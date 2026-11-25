# ExactKernel Contract

`ExactKernel` 族（现行 `ExactEqualDistantKernel`）的严格核是**纯浴对象**：
签名与冻结核一致，不含任何杂质信息（无 ε_d、无劈分阶数）——消费侧自行
携带杂质门相位（K 链门 + 接口反测度，见 GTEMPO `exactif_pipeline`）。

**分派面（2026-11-25 轴面收敛）**：严格计算只支持**实轴费米**
（`exact_fermionic_Δt`）。虚轴（`exact_fermionic_Δτ`）、混合轴
（`exact_fermionic_Δm`）、玻色族（`exact_bosonic_Δτ/Δt/Δm`）与 BCS 族
（`exact_bcs_Δτ/Δt/Δm`）已退役——统一入口对相应 scheme 路由抛「未实现」；
scheme 类型层保留为恢复挂载点，机器描述与恢复前置见包 README todolist
（金标准测试与实现见 git 历史）。

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

## 金标准（test/exact/）

严格性的机器级证据：装配输出对**直接 Trotter 定义级锚定**（逐位/机器精度）。

| 文件 | 断言层次 |
|---|---|
| `fermionic_realtime.jl` | Z ≡ 1 证书（任意 εd/调度）＋ 门序列 ED 锚 ＋ 大 N ＋ 阶数标度 ＋ FormB 认证参照（`_formb_fixture.jl`，GTEMPO 阶梯 ≤2.5e-15 同源）＋ 结构化↔稠密 |

退役轮廓/统计的金标准（fermionic_imagtime 123 项、fermionic_mixedtime 37 项、
bcs_imagtime 17 项、bcs_realtime 11 项、bcs_mixedtime 25 项）随实现入 git
历史——恢复时按 README todolist 的恢复前置立项。

## 开放项

- 非均匀/逐 cell 调度的独立 scheme 类型与调度表 API（入口已备）。
- GL-M 连续谱大规模基准（严格核实轴路线）。
- 退役族恢复（虚轴/混合轴/玻色/BCS，见 README todolist 恢复前置）。
