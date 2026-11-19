# ExactKernel Contract

`ExactKernel` 族（现行 `ExactEqualDistantKernel`）的严格核是**纯浴对象**：
签名与冻结核一致，不含任何杂质信息（无 ε_d、无劈分阶数）——杂质 kick 相位
经恒等式或 `kicks` 接口析出到装配侧。本页给出三统计严格核的严格性机制、
装配级契约（Z 证书）与金标准测试清单。

两种严格性机制：

| 统计 | 机制 | 状态 |
|---|---|---|
| 费米 | **bath+耦合 cell 的带踢 Schur**（单粒子轮廓矩阵 Schur 消浴得配对 Δ） | 三轮廓完结 |
| BCS | **Nambu/BdG 对角化闭式**（时间域机器精度，与分割无关） | 三轮廓完结（normdelta/kicks 待立项） |
| 玻色 | 每模时间域闭式（点值） | **未升级**（无 Nambu/Schur 结构，机制不同） |

## 费米：带踢 Schur 三轮廓

严格性机制：单粒子轮廓矩阵经 Schur 消浴得配对 Δ；杂质 kick 相位（幺正门
``x = e^{-i\alpha\cdot\varepsilon_d}``）经 `kicks` 接口进入 gap 复合矩阵——
+ 方向 `D(x)·W` 杂质输出行 ``\times x``、− 方向伴随 ``W^{\dagger}D^{\dagger}``
杂质输入列 ``\times\bar x``。全因子幺正 ⟹ Z 证书机器精确。

### 虚时：反周期环

`exact_fermionic_Δτ(f; β, N, μ, δτ, gfsolver) -> (ImagCorrelationFunction, normdelta)`

- 构造 = 反周期环单粒子矩阵 ``M = -\mathbb{1} + W\otimes S``（``S`` 反周期
  移位）的杂质通道 Schur 补，减去 ``x=1`` 骨架得 IF-only Δ。
- **kick 线性析出**：``\Delta_{\mathrm{IF}}(x) = x\cdot\Delta_{\mathrm{IF}}(1)``
  （对角 ``H_{\mathrm{imp}}`` 的环上 Lie ≡ Strang，踢相位纯规范）⟹ 核与
  order、``\varepsilon_d`` 无关是结构性的。
- **normdelta = det(BB)**（纯浴环浴块行列式）；装配契约
  ``Z = \mathrm{normdelta}\cdot\mathrm{Pf}(B)``，
  ``B = \mathrm{skeleton}(x) + x\cdot\Delta_{\mathrm{IF}}``。

### 实时：闭 Keldysh 开线

`exact_fermionic_Δt(f; β, N, t, μ, gfsolver) -> (ExactRealCorrelationFunction, normdelta)`

- 均匀网格便捷封装（`cells = fill(t/N, N)`，`kicks = 1`）。
- **normdelta = ``\prod_k \bar n_k \cdot \det(\mathrm{BB})``**；装配契约
  ``Z = \mathrm{normdelta}\cdot\det(\text{装配}) \equiv 1`` **机器精确**
  （任意 ``\varepsilon_d``、任意调度、任意阶数——最锐利的严格性证书）。
- 真空占据读出：``\bar n_{\mathrm{vac}} = Z\cdot(1-A^{-1}[\bar a^-_N,\ a^+_N])``。
- **kicks 为幺正门相位时观测量 = 门序列 ``\Gamma(D(x)\cdot W)`` 的精确收缩**：
  补零 pair 网格调度下 order 2 = Strang 门链（``\bar n`` 斜率 2.0）、
  order 1 斜率 1.05——两阶真实分离（踢锚定在 gap 上，τ 环式循环等价
  不成立）。

### 混合：单环 τ→+→−

`exact_fermionic_Δm(f; β, Nτ, t, Nt, μ, δτ, gfsolver; kicksτ, kickst) -> (ExactMixedCorrelationFunction, normdelta)`

- 轮廓**单环** ``\tau_1\to\dots\to\tau_{N\tau}\to(+,1)\to\dots\to(-,1)\to\tau_1``；
  ``kicks\tau`` 长 ``N\tau``（轮廓序：``N\tau-1`` 内部 + 结合 cell）、
  `kickst` 长 ``Nt``（− cells 自动取共轭）。
- **Keldysh 段算符层面精确求和相消**（``P_+P_- = \mathbb{1}``，含任意幺正
  踢）⟹ ``Z = \tau`` 环踢乘积的精确迹（与离散化无关的机器恒等式）。
- **normdelta = ``(-1)^{n(K+1)}\cdot\det(\mathrm{BB})/Z_{\mathrm{bath}}``**
  （``n = N\tau + 2(Nt+1)`` 轮廓节点数；Berezin 折叠 × impurity 折叠，
  K=1/K=2 × 奇偶 n 四点机器钉定）。

### 逐 cell 调度（内部入口）

`IFD.exact_fermionic_Δt_cells(f; β, cells, μ, gfsolver, kicks)` 支持逐 cell
调度（`cells[k]` 可取 0/负）与每 gap 幺正门相位；混合版同（`kicksτ`/
`kickst`）。参考调度表 `pair_grid(order, N, δt)`（金标准测试文件内定义）：
order 2 = Strang（半步踢两端），order 1 全步踢。

### 块结构求解

BB 消元走块结构求解，复杂度 ``O(N\cdot K^2\cdot L_{\mathrm{imp}} + K^3)``：

- 实时 = 双分支双对角 + 两角块：
  ``\det(\mathrm{BB}) = \det(\mathbb{1}_K + \Phi\,\Theta\,\Psi)``；
- 混合 = 单环：
  ``\det(\mathrm{BB}) = (-1)^{Kn}\det(\mathbb{1}_K - \Pi)``，
  ``\Pi = B[n]\cdots B[1]``（环方向有序积）。

稠密参照 `_exrt_strict_kernel_dense` / `_exrt_mixed_kernel_dense` 保留
（机器等价性受测，勿用于大规模）。

## BCS：Nambu/BdG 闭式

对角化"杂质-热库系数矩阵" = BdG 矩阵

```math
\mathcal H = \begin{pmatrix} h_+ & B \\ B^\dagger & -h_+^{\mathsf T} \end{pmatrix},
\qquad B = \Delta\cdot\mathrm{diag}(0,1,\dots,1)
```

（``h_+`` = 星形杂质(ε=0)+bath；杂质槽不配对；Δ 实/复均可）。热协方差

```math
M_0 = V\,\mathrm{diag}\Big(\tfrac{1}{1+e^{-\beta\lambda}}\Big)V^{\dagger} = \mathbb{1} - f(\mathcal H),
```

逐元 ``= \langle\Phi\Phi^{\dagger}\rangle``（``\langle\Phi^{\dagger}\Phi\rangle = f(\mathcal H)``
逐元成立）。三轮廓闭式（机器精度，与任何分割无关）：

- **虚时** ``\mathcal G_{pq}(\tau) = [e^{-\tau\mathcal H}M_0]_{\mathrm{imp}}``；
  KMS 反周期延拓 ``\mathcal G(-\tau) = -\mathcal G(\beta-\tau)``；环对角 =
  ``-\mathcal G(\beta^-)``（δ 源约定）。
- **实时** raw 轮廓收缩 ``C(v,w)=\langle T_{\mathcal C}\Phi_p(t_v)\Phi_q^{\dagger}(t_w)\rangle``：
  greater（pos ≥）``[U_v M_0 U_w^{\dagger}]``、lesser（pos <）
  ``-[U_w(\mathbb{1}-M_0)U_v^{\mathsf T}]``（反对易翻号）。
- **混合 τ 交叉**：``G_{\tau\pm}`` 走 lesser 分支（τ 早于一切 Keldysh 点）；
  ``G_{\pm\tau}`` 走 greater 闭式
  ``[W(\pm t)\cdot e^{+\mathcal H\tau}\cdot M_0]``（KMS 上半平面延拓；
  ``e^{\lambda\tau}/(1+e^{-\beta\lambda}) = e^{-\lambda(\beta-\tau)}/(1+e^{\beta\lambda})``
  ⟹ τ ∈ [0, β] 有界）。

槽位约定（row = ``\Phi_p`` 侧、col = ``\Phi_q^{\dagger}`` 侧）与四 Nambu 块
语义见 [Contours and Containers](@ref "Contours and Containers")。**μ = 0 convention**（μ ≠ 0 抛
`ArgumentError`）；ε_d 不入核（调用方 kick 簿记）。

**Δ = 0 退化**：`uu` = 单味费米 GF 闭式（h₀ 本征）；`dd` 非对角 =
``-uu^{\mathsf T}``、对角差正则反对易接触（``-1``）；`ud/du` 归零。

BCS 元组 normdelta / kicks 门相位接口**未设**——装配契约（反常 āā/aa 块的
Pfaffian 装配 + Z 证书）待 GTEMPO BCS 后端消费时立项。

## 金标准（test/exact/）

严格性的机器级证据：装配/闭式输出对 **Fock ED / 直接 Trotter 定义级锚定**
（逐位），加物理退化与跨核一致性。

| 文件 | 项数 | 断言层次 |
|---|---|---|
| `fermionic_imagtime.jl` | 54 | T2 ED 定义锚（2 模）＋ T1 装配==直接 Trotter（GL-32，N≤64）＋ T3 二阶收敛＋谱地板 |
| `fermionic_realtime.jl` | 58 | Z ≡ 1 证书（任意 εd/调度）＋ T2 门序列 ED 锚 ＋ T1 大 N ＋ T3 阶数标度 ＋ 结构化↔稠密 |
| `fermionic_mixedtime.jl` | 37 | Z = τ 环踢乘积迹（K=1/K=2 × 奇偶 n）＋ kicks 非均匀复相位 ＋ 腿基契约 |
| `bcs_imagtime.jl` | 17 | 16 维 Fock ED 环逐位（实/复 Δ）＋ Δ=0 退化 ＋ V→0 ＋ μ≠0 ＋ dispatch |
| `bcs_realtime.jl` | 11 | 四块 × 四分支对全网格 ED 逐位 ＋ 等时恒等 ＋ V→0 ＋ dispatch |
| `bcs_mixedtime.jl` | 25 | 九块 ED L 轮廓逐位 ＋ Δm↔Δτ/Δt 逐位一致 ＋ δτ 透传 dispatch |

## 开放项

- BCS：元组 normdelta、kicks 接口、装配级契约（待 GTEMPO BCS 后端）。
- 玻色严格核升级（机制不同，不能照搬费米机器）。
- 非均匀/逐 cell 调度的独立 scheme 类型与调度表 API（入口已备）。
- GL-M 连续谱大规模基准（严格核三轮廓）。
