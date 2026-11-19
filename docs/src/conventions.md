# Contours and Containers

本页汇总包内全部存储约定：三种时间轮廓的几何与节点序、Toeplitz 向量读法、
`branch` 分支块访问、点值容器与严格容器的槽位语义，以及 BCS Nambu 块的
有向 Wick 形式。

## 三种时间轮廓

| 轮廓 | 入口 | 几何 | 网格 |
|---|---|---|---|
| 虚时（Matsubara） | `Δτ(bath; N)` | 反周期环 | N 节点 ``\tau_m=(m-1)\delta\tau``，``\delta\tau=\beta/N`` |
| 实时（闭 Keldysh） | `Δt(bath; N, t)` | 正支 0→t + 反支 t→0 | 每支 N+1 节点 ``t_a=a\,\delta t``，``\delta t=t/N`` |
| 混合（Kadanoff L） | `Δm(bath; Nτ, Nt, t)` | τ 段接闭 Keldysh | τ: Nτ 节点；Keldysh 每支 Nt+1 节点 |

## contour_ordering：节点统一序

`contour_ordering(x) -> Vector{Tuple{Symbol,Int}}` 给出轮廓节点的统一序
（`ext` 扩展的 IF 张量轴序同此）：

```text
IM : (:τ,1) (:τ,2) … (:τ,Nτ)                    虚时升序
RT : (+,1) (−,1) (+,2) (−,2) …                  时间升序，每节点 + 先 − 后
MX : (:τ,1) … (:τ,Nτ)  ++  (+,1) (−,1) (+,2) …  虚时升序接实时同款交错
```

费米多 band（BCS）时，腿基 `basis(x)` 在每个节点内按 band 1,2,… 展开，
每个 band 内 `a`、`ā` 相邻（`ā` 紧跟 `a`）；单 flavor 腿数 `2n`、BCS `4n`
（n = 轮廓节点数）。

## CorrelationMatrix：Toeplitz 向量读法

虚时（及冻结核实时/混合的 Toeplitz 生成向量）只存两条向量：

```math
\Delta_{ij} = \begin{cases}
\eta_{jk}[\,i-j+1\,], & i>j \\
\eta_{jk}[1]+\eta_{kj}[1], & i=j \\
\eta_{kj}[\,j-i+1\,], & i<j
\end{cases}
```

**对角元是两种排序之和**——只对对角耦合成立。冻结核把接触（δ 源）约定
在 `ηⱼₖ[1]`（`ηₖⱼ[1] = 0`），对角由 `ηⱼₖ[1]` 独担；严格虚时核（费米/BCS
`exact_*_Δτ`）同款。

## branch：分支块访问

`branch(x; b1, b2)` 按 `(b1, b2) ∈ (:τ,:+),:-` 组合返回分支块矩阵：

- **虚时**：只有 `(:τ,:τ)`——即 Toeplitz 环矩阵的展开。
- **实时点值容器**：存储两块交叉块 `G₊₋`（lesser 型，``\bar n`` 权）与
  `G₋₊`（greater 型，``1-\bar n`` 权）；**对角块由 sector 规则推导**
  （三角形从伙伴块重构 + 对角约定）。推导对解析关联函数精确，对
  **raw Schur 收缩不成立**（见下节严格容器）。
- **混合点值容器**：存 `G₊₋ / G₋₊ / G₊τ / Gτ₊` 四块 + 虚时环；其余块推导。

`GrassmannContourIndex` / `BosonicContourIndex` 提供单条配对访问：
`corr[i, j]`（`i, j` 为腿指标）——Grassmann `ā–a` 方向取 Δ、`a–ā` 取
``-\Delta``。

## 严格容器（ExactReal / ExactMixed）

`ExactRealCorrelationFunction`（四块 G₊₊/G₊₋/G₋₊/G₋₋）与
`ExactMixedCorrelationFunction`（九块 ττ/τ₊/τ₋/₊τ/₋τ + 四 Keldysh）是
**装配级配对矩阵**的稠密分支块存储：

```math
\mathsf{G}[b_1,b_2][i,j] \;=\; \text{系数}\ \bar a^{b_1}_i \cdot a^{b_2}_j
```

（row = ``\bar a`` 侧、col = ``a`` 侧；节点序同 `contour_ordering`）。严格
核的块相互独立（bath Schur 产物），**相互间无 sector 关系**——`branch`
直取真块（无损），点值容器的对角推导不适用。

接口：`branch(x; b1, b2)` 无损切片、`==` / `isapprox` 逐字段、混合容器以
`isize`（τ 节点数）/`rsize`（Keldysh 节点数）承载两段规模。

## BCSCorrelationFunction：四 Nambu 块

BCS（超导）浴的 IF 指数是**有向 Wick 形式**（注意 ↓↓ 正常项 ā↓ 在前）：

```math
\bar a^\uparrow \Delta^{\uparrow\uparrow} a^\uparrow
+ \bar a^\uparrow \Delta^{\uparrow\downarrow} \bar a^\downarrow
+ a^\downarrow \Delta^{\downarrow\uparrow} a^\uparrow
+ \bar a^\downarrow \Delta^{\downarrow\downarrow} a^\downarrow
```

内部四块按配对约定存储：

| 字段 | 位置序 | 语义 | Bogoliubov 通道 |
|---|---|---|---|
| `uu` | cn | 正常 ↑↑（``\bar a^\uparrow a^\uparrow`` 系数） | ``u^2`` 减 ``v^2`` 洞修正 |
| `ud` | cc | 反常 cc（``\bar a^\uparrow \bar a^\downarrow``） | ``uv + (uv)^T`` |
| `du` | nn | 反常 nn（``a^\downarrow a^\uparrow``） | ``vu + (vu)^T`` |
| `dd` | nc | **洞侧 Nambu(2,2) partner**（≠ ``\bar a^\downarrow a^\downarrow`` 配对块） | ``v^2 - (u^2)^T`` |

- `x[c1, c2]`（`c1, c2 ∈ (:↑, :↓)` 或 `Bool`）按味选块；`x.uu` 等别名经
  `getproperty`。
- **Δ = 0 退化**：`uu` 退化为单味费米 GF；`dd` 非对角 = ``-uu^{\mathsf T}``
  （对角差正则反对易接触）；`ud/du` 归零。下游单味装配应使用 `uu`
  （`corr.dd` **不**退化为单味结果——它是洞侧 partner）。
- 严格 BCS 核存 **raw Nambu 收缩**（无提取符号）：

```math
\mathsf{G}^{(p,q)}[b_1,b_2][i,j] =
\langle T_{\mathcal C}\, \Phi_p(b_1,i)\,\Phi_q^\dagger(b_2,j)\rangle,
\qquad \Phi = (d^\uparrow,\ d^{\downarrow\dagger})
```

  row = ``\Phi_p`` 侧时间/支、col = ``\Phi_q^\dagger`` 侧；等时点走
  greater/``0^+`` 侧。四块语义：`(1,1)=uu`、`(2,2)=dd`、`(1,2)=ud`、
  `(2,1)=du`。逐位锚定见 `test/exact/bcs_*`（Fock ED 金标准）。

## 方案层级与输出对照

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

| 轮廓 | 冻结核（`QuAPI()`） | 严格核（`ExactEqualDistantKernel()`） |
|---|---|---|
| `Δτ` | 点值容器 | 费米 `(ImagCorrelationFunction, normdelta)`；玻色/BCS 点值容器 |
| `Δt` | 点值容器 | 费米 `ExactRealCorrelationFunction`；玻色点值；BCS `BCSCorrelationFunction{ExactReal}` |
| `Δm` | 点值容器 | 费米 `ExactMixedCorrelationFunction`；玻色点值；BCS `BCSCorrelationFunction{ExactMixed}` |

费米严格核元组返回的第二分量 `normdelta` 是装配级 Z 证书
（[ExactKernel Contract](@ref "ExactKernel Contract")）；统一入口对 Δτ 透传元组、Δt/Δm 只返回
容器（normdelta 经直接调用 `exact_*` 获取）。
