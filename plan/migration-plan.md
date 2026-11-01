# IFDiscretization 移植计划 —— QuAPI 包重构

> **实施状态（2026-11-01，rev 7 阶数严格对比 + 存档回归移植）**：阶段 0–4 完成。
> 正式测试套件（test/）= QuAPI 单元测试的移植（**不依赖 QuAPI**）；
> 与 QuAPI 的逐位对拍在 `debug/parity_vs_quapi.jl`（独立环境，23 项全绿）。
> QuAPI 未做任何改动（阶段 5 薄壳化待做）。
>
> rev 7 补齐：① `test/exact/trotter_orders.jl`——exact IF 的 1/2/4/6 阶
> Trotter 严格对比（exact 核 + order-k kick 装配 vs **同阶直接 Trotter 严格
> 演化**（JW Fock 多体轮廓迹），全节点逐位 <1e-12，×2 网格 + 非空性检查；
> ← mx_mb_anchor.jl 推广到 4/6 阶）。② 一阶/二阶装配存档回归移植——
> `performance/{common,firstorder,secondorder}` 脚本树（`using QuAPI` →
> `using IFDiscretization`；*_dense.jl 不迁）+ `test/quapi/archive_*` 存档
> CSV（19 个中 18 个，imagtime_dense 随 Dense 留 QuAPI）+ `archive_regression.jl`
> 逐字节回归（数值移植的最终防线）。
>
> rev 6 补齐：GaussianStates 扩展随迁——`src/ifmatrix/`（← QuAPI.Dense 的
> corr→配对矩阵层：basis/contour_ordering/GrassmannHybridizationMatrix（含
> 顶点位移装配）/BosonicHybridizationMatrix/BosonicSplitDelta；稠密**张量**
> 机器 GrassmannADT/IF/PT/Zvalue/margin 仍留守 QuAPI.Dense）+
> `ext/IFDiscretizationGaussianStatesExt.jl`（原样移植）+ test/osee.jl
> （ext 核心测试集 55 项；稠密张量锚段与 BCS SVD 锚留 QuAPI/test/osee.jl，
> 过渡期继续验证 BCS 路径）。GaussianStates 已 dev 入默认环境（@v1.10，
> 与家族其余包一致）；osee 测试有 `HAVE_GAUSSIANSTATES` 守卫：脚本直跑
> `julia --project=. test/runtests.jl` 生效，Pkg.test 隔离环境自动跳过。
>
> 实测备注：两包同名类型跨模块 `==` 回落 `===`（对拍用逐字段比较）；
> exact_bcs_Δτ 无 δτ 旋钮（内定 δτ=β/N）；exact_fermionic_Δt 保留元组返回
> （第二元素 = 四块 NamedTuple，api_consistency 在用）；quapi 测试中
> 经 QuAPI.Dense Grassmann 机器构造参照的三处（imag.jl 的 Toulouse
> Grassmann 张量测试集、bcsimag.jl 全文件、bcsmixed.jl 全文件）属 dense
> 交叉验证，不移植（留在 QuAPI/test）。
>
> 日期：2026-11-01（rev 4。修订史：rev 2 求解器改名/exactfermionic 删除/utility 重排；
> rev 3 Ng–Reichman 研究线不并入，只收编 exactif/test 的三套 exact API；
> rev 4 ExactKernel→**ExactIF**，统一入口 Δτ/Δt/Δm **第一个输入必须是 bath 对象**）
> 状态：阶段 0–4 已实施；阶段 5（QuAPI 薄壳化 + 下游回归）与阶段 6 未实施
> 范围：将 QuAPI 的离散 IF（influence functional）装配层重构为以
> `DiscretizationScheme` 抽象为核心的独立包 `IFDiscretization`；
> QuAPI 的 `Dense` 子模块与 `deprecated` 实现**不移植**；
> `src/exactfermionic.jl`（ExactGrassmannIF）**直接删除**，由 exactif 的 exact API 接替。

---

## 0. 核心思想与类型层级

现 QuAPI 中「离散 IF 怎么算」散落在三处且互相不可替换：
1. 冻结核（Trotter 冻结顶点幅）装配：`src/{imagtime,realtime,mixedtime}/{fermionic,bosonic}.jl`、`src/bcs/*.jl`；
2. 严格核（exact API）：`exactif/test/exactif_api.jl`（费米）、`exactif/test/exact_bosonic_api.jl`（玻色）、`exactif/test/exact_bcs_api.jl`（BCS）——研究目录，不在 QuAPI 模块内；
3. Grassmann 严格 IF：`src/exactfermionic.jl`（`ExactGrassmannIF`，依赖 `Dense`）——**本计划直接删除**（§0.2）。

重构后用一个虚类（abstract type）封装「离散 IF 算法」，调用方只面对
统一入口 `Δτ / Δt / Δm` 与末位的 `scheme::DiscretizationScheme` 参数：

```julia
# ---------- 方案（schemes.jl） ----------
"""离散 IF 算法的抽象基类（虚类）。"""
abstract type DiscretizationScheme end

"""冻结核技术：顶点幅冻结为 Duhamel 低阶项的离散方案。"""
abstract type FrozenKernel <: DiscretizationScheme end

"""一阶 Trotter 冻结核（现 bosonic_Δτ/Δt/Δm、fermionic_*、bcs_* 的装配）。"""
struct QuAPI <: FrozenKernel end

# 后续扩展位（不在本期）：二阶装配（Strang/Weyl-K）、QuAPI2 等
# struct QuAPI2 <: FrozenKernel end

# ---------- 格林函数求解器（ExactIF 的内插部件） ----------
"""
Toulouse 型严格核（单杂质能级 + 浴，U = 0）的格林函数求解器抽象。
exactif 机器的语义边界：杂质模 + 数守恒浴的二次型链。
"""
abstract type ToulouseGreenFunctionSolver end

"""
    EigenSolver(nbath)

本征求解器：把浴 + 杂质单粒子 h₀ 组装为 (nbath+1)×(nbath+1) 矩阵后
`eigen(Hermitian(h0))`，传播子用本征相位闭式（exactif/test/exactif_api.jl
的 `_bath_eig` 路线，即当前实际使用的求解器）。

- `nbath`：浴模式数（连续谱 = Gauss–Legendre–M 求积节点数 M，
  exactif 现默认 M = 64；离散谱 = 谱线数目）。
"""
struct EigenSolver <: ToulouseGreenFunctionSolver
    nbath::Int
end

"""
    AnalyticSolver()

解析求解器（扩展位，本期不实现）：接 ImpurityModelBase 中 Toulouse 模型的
解析解，作为 EigenSolver 的零求积误差参照/替代。
"""
struct AnalyticSolver <: ToulouseGreenFunctionSolver end

# 其他求解器扩展位：矩阵指数直解（exp(-iΔt·h)）、Krylov / Chebyshev 等。

"""严格离散影响泛函核方案，格林函数由内插求解器 `gfsolver` 提供。"""
struct ExactIF{G <: ToulouseGreenFunctionSolver} <: DiscretizationScheme
    gfsolver::G
end
ExactIF() = ExactIF(EigenSolver(64))   # 默认 = exactif 现行配置
```

层级关系（「QuAPI < FrozenKernel < DiscretizationScheme」）：

```
DiscretizationScheme
├── FrozenKernel
│   ├── QuAPI()            # 一阶冻结核（本期迁移现实现）
│   └── (QuAPI2 等，留扩展位)
└── ExactIF(gfsolver)      # 严格核，gfsolver::ToulouseGreenFunctionSolver
    ├── ExactIF(EigenSolver(nbath))   # 本征路线（现行默认 nbath = 64）
    └── ExactIF(AnalyticSolver())     # 解析路线（扩展位，接 ImpurityModelBase Toulouse 解析解）
```

### 0.1 exactif 的收编范围（rev 3 裁决）：只收 exact API，研究线留原地

`QuAPI/exactif/` 拆成两半，命运不同：

| 部分 | 内容 | 处置 |
|---|---|---|
| **exact API**（收编） | `test/exactif_api.jl`（exact_fermionic_Δτ/Δt/Δm + exact_fidelity）、`test/exact_bosonic_api.jl`、`test/exact_bcs_api.jl` | 迁入 IFDiscretization `*/exact/`，成为 `ExactIF` 方案的实装 |
| **Ng–Reichman 研究线**（不并入） | `kernel.jl`（struct ExactKernel，G 四块 + Θ）、`spectral.jl`、`volterra.jl`、`green.jl`、`brute.jl`、`contract.jl`、`runtests.jl`（L0–L8）、`test/mx_mb_anchor.jl` | **留在 `QuAPI/exactif/` 原地**，作为研究/验证目录，永不进模块 |

- **命名说明（rev 4）**：IFDiscretization 的方案类型名为 `ExactIF`
  （自 rev 1–3 的 `ExactKernel` 改名，语义 =「严格影响泛函方案」）；
  exactif 研究线的 `struct ExactKernel` 原名保留、不进模块，与研究线外
  的 `ExactIF` 互不相干，即使将来研究线反悔要并入也不撞名。
- **exact API 对研究线的唯一代码级依赖（已逐符号核实）**：
  `gauss_legendre(n)`（Golub–Welsch 求积节点，定义在 `volterra.jl:107`）。
  exactif_api.jl / exact_bosonic_api.jl / exact_bcs_api.jl 对 brute.jl 的
  多体机器（single_particle_h、if_path_values、mb_generator 等）**零引用**。
  处理：把 `gauss_legendre` 拷入 IFDiscretization `src/utility.jl` 并导出，
  三套 exact API 共用；exactif 内的 volterra.jl 原件不动（两份独立，见 §7 风险）。
- exact API 三个文件的 include 链（`exact_bosonic_api/exact_bcs_api → exactif_api
  → ../brute.jl + ../volterra.jl`）迁移时打断：改为 `using IFDiscretization`。

### 0.2 `src/exactfermionic.jl` 直接删除（rev 2 裁决，维持）

`ExactGrassmannIF` 路线（Schur 边缘化 + Grassmann 配对矩阵 + prefactor，
依赖 `Dense.GrassmannHybridizationMatrix`）不再维护：

- **删除** `QuAPI/src/exactfermionic.jl`（363 行）与 `QuAPI/test/quapi/exactfermionic.jl`
  （12 失败在途的测试，2026-10-31 起本已屏蔽）；
- **接替者**：exact API 的 `exact_fermionic_Δτ / Δt / Δm`——三轮廓齐全
  （`exact_fermionic_Δm` 在 exactif_api.jl:159 已有实现），返回标准
  CorrelationFunction，正好是 `ExactIF(EigenSolver(M))` 方案；
- **QuAPI export 同步收缩**：`QuAPI.jl:23` 的
  `ExactGrassmannIF, exact_fermionic_Δτ, exact_fermionic_Δt` 三个名字删除
  （`exact_fermionic_*` 由 IFDiscretization 的 exact API 重新承担，
  `ExactGrassmannIF` 无继承者）；
- **下游影响 = 零**（已核实）：QuAPI 之外唯一提及在
  `QuAPI/benchmark/bosons/realtime_exactkernel.jl:21`（注释里描述 Schur 路线，
  并入后顺手更新该注释）。

---

## 1. 目标文件结构（完整映射表）

新包为 `/home/guochu/Documents/Missile/IFDiscretization`（QuAPI 的兄弟目录），
依赖 `ImpurityModelBase`（`AbstractBoundedFunction`、`spectrumshift`、
`boseeinstein`、`quadgkwrapper`、`*Bath` 类型、`DiracDelta`/`DiscreteSpectrum`）
与 `LinearAlgebra`。

```
IFDiscretization/
├── Project.toml                  # name=IFDiscretization；deps: ImpurityModelBase, LinearAlgebra
├── plan/
│   └── migration-plan.md         # 本文件
├── src/
│   ├── IFDiscretization.jl       # 模块入口 + exports + include 序
│   ├── schemes.jl                # §0 全部类型：DiscretizationScheme/FrozenKernel/QuAPI/
│   │                             #   ToulouseGreenFunctionSolver/EigenSolver/AnalyticSolver/ExactIF
│   ├── defaults.jl               # ← QuAPI/src/utility.jl（QuAPI_tol/get/set_quapi_tol!）
│   ├── utility.jl                # Δiw_to_Δτ（自 QuAPI/src/imagtime/imagtime.jl 拆出）
│   │                             # + gauss_legendre（自 exactif/volterra.jl:107 拷入并导出，§0.1）
│   ├── contourindex.jl           # ← QuAPI/src/contourindex.jl（AbstractContourIndex/
│   │                             #   GrassmannContourIndex/BosonicContourIndex；TEMPO/GTEMPO 依赖）
│   │
│   ├── corrstorages/             # —— 纯存储容器（无装配逻辑）——
│   │   ├── corrstorages.jl       #   入口：include 各文件 + abstractcorr
│   │   ├── abstractcorr.jl       # ← QuAPI/src/abstractcorr.jl（AbstractCorrelationFunction + branch 接口）
│   │   ├── corrmat.jl            # ← QuAPI/src/corrmat.jl（CorrelationMatrix Toeplitz 存储）
│   │   ├── imagcorr.jl           # ← QuAPI/src/imagtime/imagcorr.jl（ImagCorrelationFunction）
│   │   ├── realcorr.jl           # ← QuAPI/src/realtime/realcorr.jl（RealCorrelationFunction）
│   │   ├── mixedcorr.jl          # ← QuAPI/src/mixedtime/mixedcorr.jl（MixedCorrelationFunction）
│   │   └── bcscorr.jl            # ← QuAPI/src/bcs/bcscorr.jl（BCSCorrelationFunction）
│   │                             #   （决策点 D2：bcscorr 是存储容器，随迁 corrstorages；
│   │                             #    若坚持只放用户点名的 4 个文件，则它退到 bcs/ 根，二选一）
│   │
│   ├── bosons/
│   │   ├── bosons.jl             # 统一入口 Δτ/Δt/Δm：bath 对象入参 + scheme 末位关键字（见 §3）
│   │   ├── quapi/                # 冻结核装配（现实现原样迁移）
│   │   │   ├── quapi.jl          #   入口，include 序：util → realtime → imagtime → mixedtime
│   │   │   ├── util.jl           #   _bosonic_gⱼₖ_i / _bosonic_gⱼₖ_r / _bosonic_lⱼₖ 等共享 g 内核
│   │   │   │                     #   （现分散在 src/realtime/bosonic.jl、src/imagtime/bosonic.jl、
│   │   │   │                     #    src/mixedtime/bosonic.jl 尾部，含 *_disperse 包装；集中成一个文件）
│   │   │   ├── imagtime.jl       # ← src/imagtime/bosonic.jl 的 bosonic_Δτ（4 个方法 + 文档 + μ=0 奇异性注记）
│   │   │   ├── realtime.jl       # ← src/realtime/bosonic.jl 的 bosonic_Δt（含 conj 派生次块的统一约定）
│   │   │   └── mixedtime.jl      # ← src/mixedtime/bosonic.jl 的 bosonic_Δm（ζₖⱼ ×(−im) 约定）
│   │   └── exact/                # ← QuAPI/exactif/test/exact_bosonic_api.jl
│   │       ├── exact.jl          #   入口
│   │       └── bosonic.jl        #   _boson_modes + exact_bosonic_Δτ / Δt / Δm
│   │
│   ├── fermions/
│   │   ├── fermions.jl           # 统一入口 Δτ/Δt/Δm
│   │   ├── quapi/
│   │   │   ├── quapi.jl
│   │   │   ├── util.jl           #   _fermionic_gⱼₖ_i/_r、_fermionic_lⱼₖ 等共享 g 内核
│   │   │   │                     #   （注意：src/imagtime/util.jl 的 _fⱼₖ_i 系列是 deprecated 专用
│   │   │   │                     #    不迁；safe_mult 被 bosonic 装配用 → 归 bosons/quapi/util.jl）
│   │   │   ├── imagtime.jl       # ← src/imagtime/fermionic.jl 的 fermionic_Δτ
│   │   │   ├── realtime.jl       # ← src/realtime/fermionic.jl 的 fermionic_Δt
│   │   │   └── mixedtime.jl      # ← src/mixedtime/fermionic.jl 的 fermionic_Δm
│   │   └── exact/                # ← QuAPI/exactif/test/exactif_api.jl
│   │       ├── exact.jl          #   入口
│   │       ├── eigensolve.jl     #   EigenSolver 的实现：_quad_nodes + _bath_eig + W/Wd/Wim 传播子
│   │       └── fermionic.jl      #   exact_fermionic_Δτ / Δt / Δm（接替 src/exactfermionic.jl）
│   │
│   ├── bcs/
│   │   ├── bcs.jl                # 统一入口 Δτ/Δt/Δm（bcs 四块包装）
│   │   ├── quapi/
│   │   │   ├── quapi.jl
│   │   │   ├── imagtime.jl       # ← src/bcs/imagtime.jl 的 bcs_Δτ（复用 fermionic_Δτ + disperse）
│   │   │   ├── realtime.jl       # ← src/bcs/realtime.jl 的 bcs_Δt
│   │   │   └── mixedtime.jl      # ← src/bcs/mixedtime.jl 的 bcs_Δm
│   │   └── exact/                # ← QuAPI/exactif/test/exact_bcs_api.jl
│   │       ├── exact.jl
│   │       └── bcs.jl            #   bcs_nambu_data + exact_bcs_Δτ / Δt / Δm（Nambu 闭式）
│   │
│   ├── infinitetime/             # （决策点 D3：一期迁入 or 留 QuAPI，见 §6）
│   │   ├── imagtime.jl           # ← src/infinitetime/imagtime.jl（infinite_Δτ + InfiniteImagCorrelationCache）
│   │   └── realtime.jl           # ← src/infinitetime/realtime.jl（infinite_Δt + InfiniteRealCorrelationCache）
│   │
│   └── …（扩展位：QuAPI2 冻结核、AnalyticSolver 实装等）
│
├── ext/
│   └── IFDiscretizationGaussianStatesExt.jl
│                                 # ← QuAPI/ext/QuAPIGaussianStatesExt.jl（weakdeps GaussianStates，
│                                 #    IF 系数态构造；osee 测试的 Dense 锚见 §5 决策点 D4）
│
└── test/
    ├── runtests.jl               # 新写（见 §5）
    ├── api/                      # ← QuAPI/test/api/（删 *_deprecated 对照块）
    ├── quapi/                    # ← QuAPI/test/quapi/（imag/real/mixed/bcsimag/bcsreal/bcsmixed；
    │                             #    archive_regression 延后，见 §5）
    ├── exact/                    # ← exactif/test/ 中与 exact API 绑定的测试（§5 逐文件核实）
    │   ├── exact_bosonic_check.jl
    │   ├── exact_bcs_check.jl          #   需拷 mb_generator 锚（~15 行，test/_anchor_utils.jl）
    │   ├── api_consistency.jl
    │   ├── api_mixed_check.jl
    │   ├── mx_chk.jl
    │   ├── mx_mb_anchor_boson.jl
    │   ├── toulouse_greens.jl
    │   ├── toulouse_greens_imagtime.jl
    │   ├── toulouse_greens_mixedtime.jl
    │   ├── toulouse_bosonic_greens.jl
    │   ├── toulouse_bcs_greens_mixedtime.jl
    │   └── _anchor_utils.jl            #   测试本地锚：mb_generator（自 brute.jl 拷，注明出处）
    └── osee.jl                   # ← QuAPI/test/osee.jl（视 D4 决策）
```

### 不迁移清单（留在 QuAPI 或直接删除）

| 源 | 处置 |
|---|---|
| `src/dense/`（Dense 子模块：common/fermion/boson，1042 行） | **不迁移**；QuAPI 内保留到下游不再使用后删除 |
| `src/deprecated/{imagtime,realtime,mixedtime}/*.jl`（522 行） | **不迁移**；随交叉验证测试一起删除 |
| `src/exactfermionic.jl`（ExactGrassmannIF，依赖 Dense） | **直接删除**（rev 2 裁决，§0.2）；由 exact API 接替 |
| `test/quapi/exactfermionic.jl` + `runtests.jl:33-38` 屏蔽块 | 随 `src/exactfermionic.jl` 一起删除 |
| `exactif/{kernel,spectral,volterra,green,brute,contract}.jl` + `exactif/runtests.jl`（L0–L8） + `exactif/test/mx_mb_anchor.jl` | **Ng–Reichman 研究线，留 QuAPI/exactif 原地**（rev 3 裁决，§0.1），不并入、不进模块 |
| `test/dense/`（3 个文件） | 删除（测试 dense 的，dense 没了） |
| `performance/`、`bugrepots/`、`docs/` | 留 QuAPI；archive_regression 一期从 QuAPI test 移除（见 §5） |
| `benchmark/` | 留 QuAPI；`benchmark/bosons/realtime_exactkernel.jl:21` 注释提及 Schur 路线，删除 exactfermionic 后顺手更新 |

---

## 2. 迁移中的代码级注意点（逐文件）

1. **mixedtime 依赖 realtime/imagtime 的内部 helper**：
   `src/mixedtime/{bosonic,fermionic}.jl` 直接调用 `_bosonic_fⱼₖ_r_disperse`、
   `_bosonic_fⱼₖ_i_disperse`（定义在 realtime/imagtime 的 bosonic.jl）。
   迁移时把全部 `_*_disperse` / `_g*` 内核收进各统计目录 `quapi/util.jl`，
   `quapi.jl` 的 include 顺序改为 util 最先，消除跨文件隐式依赖。

2. **`src/imagtime/util.jl` 的拆解**：`_fⱼₖ_i` 系列（无前缀）只被
   deprecated 使用 → 不迁；`safe_mult` 被 bosonic 装配用 → 迁入
   `bosons/quapi/util.jl`（fermions 若也用则提根到 `src/utility.jl`）。

3. **utility/defaults 重排（rev 2）**：`QuAPI_tol/get_quapi_tol/set_quapi_tol!`
   → `src/defaults.jl`；`Δiw_to_Δτ` 自 `imagtime/imagtime.jl` 拆出 →
   `src/utility.jl`（依赖 ImpurityModelBase 的 `ifrequencies`）。
   `gauss_legendre`（rev 3）同放 `src/utility.jl` 并导出。

4. **exact API 的 include 链打断（rev 3）**：原链
   `exact_bosonic_api/exact_bcs_api → exactif_api → ../brute.jl + ../volterra.jl`
   改为 `using IFDiscretization`；三个 API 文件内对 QuAPI 的
   `using QuAPI / using QuAPI: spectrumshift` 改为包内直接引用；
   `_quad_nodes`/`_bath_eig` 归 `fermions/exact/eigensolve.jl`（EigenSolver 实装），
   `_boson_modes` 归 `bosons/exact/bosonic.jl`，`bcs_nambu_data` 归
   `bcs/exact/bcs.jl`——三者都从 `EigenSolver.nbath` 取节点数 M。

5. **BCS 复用费米冻结核**：`bcs/quapi/*.jl` 调用 `fermionic_Δτ(f, disperse, …)`
   取 `:τ`/Keldysh 主块再包装四块 → `bcs` 目录 include 依赖 `fermions`，
   模块 include 序必须 `fermions` 在 `bcs` 之前（现状也是这个顺序）。

6. **类型参数约定不能破坏**（GTEMPO `correlationfunction.jl:20` 依赖）：
   `ImagCorrelationFunction{T}` 的 `T` 是 `CorrelationMatrix` 元素类型；
   `Real/MixedCorrelationFunction` 为 ComplexF64；
   `BCSCorrelationFunction{M}` 泛化包裹。迁移是纯移动，不改任何 struct 定义。

7. **exact 核的 μ 语义差异要写进文档**（从 exactif 注释原样搬运）：
   玻色 `exact_bosonic_Δτ` 走 spectrumshift（要求 ω_k>0），
   `exact_bosonic_Δt/Δm` 走热因子 boseeinstein(β,μ,ω)（要求 ω_k>μ）；
   谱带不得触及 ε=0。

8. **exact 核的求解器化（EigenSolver）**：exactif_api.jl 里 `_quad_nodes`
   （GL-M 谱离散）+ `_bath_eig`（eigen(Hermitian(h0))）+ 传播子闭式 =
   `EigenSolver` 的全部内容。改造为：`exact/fermionic.jl` 只做轮廓收缩装配，
   谱→格林函数统一走 `solve_gf(s::EigenSolver, f, β, μ, ts)` 这类接口；
   `nbath` 即 M（当前默认 64）。玻色 `_boson_modes` 与 BCS Nambu 闭式同法接
   `EigenSolver.nbath`。`AnalyticSolver` 本期只定义类型 + 分派桩
   `error("AnalyticSolver 待实装：接 ImpurityModelBase Toulouse 解析解")`，
   不写实现。迁移前后数值必须逐位一致（同一 M、同一节点排布）。
   `ExactIF` 分派接线见 §3。

9. **QuadGK 依赖**：exactif_api.jl 的 `exact_fidelity` 用 `QuadGK`（核装配
   本身不用）→ `exact_fidelity` 属测试/解析参照，放 test/ 不进 src；
   Project.toml 只需 `ImpurityModelBase + LinearAlgebra`
   （quadgkwrapper 来自 ImpurityModelBase）；test 对象加 QuadGK。

---

## 3. 统一入口 API（rev 4：第一个输入必须是 bath 对象）

**统计信息必须由类型承担**：裸谱函数 `f::AbstractBoundedFunction` 分不出
费米/玻色/BCS，所以统一入口 `Δτ / Δt / Δm` 的第一个输入**只收 bath 对象**
（`AbstractBosonicNormalBath` / `AbstractFermionicNormalBath` / `AbstractBCSBath`），
Julia 按第一层分派选统计；`scheme::DiscretizationScheme` 作为最后一个
关键字参数做第二层分派选核技术（冻结核 or 严格核）：

```julia
# ---------------- bosons/bosons.jl（玻色侧；fermions/bcs 同构） ----------------
# 虚时：第一层 = bath 类型（统计），第二层 = scheme（核技术）
Δτ(bath::AbstractBosonicNormalBath; N::Int, δτ::Real=bath.β/N,
   scheme::DiscretizationScheme=QuAPI()) = _bosons_Δτ(scheme, bath, N, δτ)

_bosons_Δτ(::QuAPI, bath, N, δτ) =
    bosonic_Δτ(bath.spectrum; β=bath.β, N=N, μ=bath.μ, δτ=δτ)
_bosons_Δτ(s::ExactIF, bath, N, δτ) =
    exact_bosonic_Δτ(bath.spectrum; β=bath.β, N=N, μ=bath.μ, δτ=δτ; s.gfsolver)

# 实时 / 混合同构：
Δt(bath::AbstractBosonicNormalBath; N, t, scheme=QuAPI()) = _bosons_Δt(scheme, bath, N, t)
Δm(bath::AbstractBosonicNormalBath; Nτ, t, Nt, δτ=bath.β/Nτ, scheme=QuAPI()) = _bosons_Δm(scheme, bath, Nτ, t, Nt, δτ)

# fermions/fermions.jl：Δτ/Δt/Δm(::AbstractFermionicNormalBath; …) → fermionic_Δτ / exact_fermionic_Δτ
# bcs/bcs.jl：      Δτ/Δt/Δm(::AbstractBCSBath; …)            → bcs_Δτ      / exact_bcs_Δτ
```

设计约定：

1. **两层分派，零类型标记**：调用方写 `Δτ(bath; N=…, scheme=ExactIF())`
   即可——bath 类型回答「什么统计」，scheme 回答「什么核」，
   不需要 `bosonic_Δτ(bath; scheme=…)` 这类带统计前缀的统一入口。
2. **向后兼容是结构性的**：QuAPI 现状就有 `Δτ(bath; N)`、`Δt(bath; N, t)`、
   `Δm(bath; Nτ, t, Nt, δτ)` 三组 bath 方法（三统计共 9 个），统一入口
   就是给它们加一个默认 `QuAPI()` 的 `scheme` 关键字——既有调用
   语义不变、位置兼容。
3. **裸谱调用走具体函数**：只有谱没有 bath 对象的调用方（下游
   GTEMPOProjects 的 ~60 处 `fermionic_Δτ(bath.spectrum, …)` 等正是这种）
   继续用具体函数 `bosonic_Δτ(f; …)` / `fermionic_Δτ(f; …)` / `bcs_Δτ(f; …)`
   与 `exact_bosonic_Δτ(f; …)` / `exact_fermionic_Δτ(f; …)` / `exact_bcs_Δτ(f; …)`
   ——名字、签名全部保留，零改动。
4. **disperse 退出统一入口（rev 4 简化）**：色散变体只存在于具体函数层
   （`*_Δτ(f, disperse; …)`；BCS 冻结核内部的 `ε -> sqrt(ε²+|Δ|²)` 在
   `bcs/quapi` 内部拼接），bath-first 的统一层不再暴露 `disperse` 关键字。
5. 具体实现函数保持原名字、原签名导出；`exact_fermionic_*` 语义切换
   （exact API 新实现替代被删的 exactfermionic.jl 旧实现）属**显式行为变更**，
   在 IFDiscretization 文档字符串与 CHANGELOG 注明（§0.2）。
6. TEMPO/GTEMPO 的 `import QuAPI: branch` 不受影响（`branch` 属
   corrstorages 的接口，原样迁移）。
7. 返回类型不变：冻结核与 exact 核都返回标准
   `Imag/Real/MixedCorrelationFunction`（exact 核从 exact API 原样继承），
   调用方（含 branch 访问、GTEMPO scalartype）无感。
8. **Vacuum 浴不在统一入口内**：`infinite_Δτ(bath::Vacuum; …)` /
   `infinite_Δt` 属 infinitetime 模块（决策点 D3），签名不含 scheme，
   维持现状。

---

## 4. 实施阶段（建议 6 个 PR/commit，每步可回退）

### 阶段 0 —— 包骨架（无行为）
- 建目录树 + `Project.toml`（deps: ImpurityModelBase, LinearAlgebra；
  weakdeps: GaussianStates；extras/targets: Test, QuadGK）+ `src/IFDiscretization.jl`
  空壳 + `plan/`。
- 验收：`julia --project -e 'using IFDiscretization'` 通过。

### 阶段 1 —— 存储层与基础设施（纯移动）
- 迁 `defaults.jl`（tol）、`utility.jl`（Δiw_to_Δτ + gauss_legendre）、
  `contourindex.jl`、`corrstorages/` 六件。一行不改代码，只挪文件 + 写 include。
- exports：照抄 QuAPI 现有 export 清单中对应符号（+ `gauss_legendre`）。
- 验收：在 REPL 里对拍——对同一组谱构造 `CorrelationMatrix`、四种
  CorrelationFunction、`Δiw_to_Δτ` 结果，`==`/`isapprox` 与 QuAPI 版逐位一致
  （写临时脚本，不进 test）。

### 阶段 2 —— 冻结核（quapi/ 三目录）+ 统一入口
- 迁 bosons/fermions/bcs 的 `quapi/` 子目录（注意 §2.1/2.2/2.5）；
  写三个统计入口 `bosons.jl`/`fermions.jl`/`bcs.jl`
  （`Δτ/Δt/Δm(bath; …, scheme)` 两层分派，§3）。
- 此阶段 `ExactIF`/求解器类型先在 `schemes.jl` 定义（含 `AnalyticSolver`
  桩），exact 分派暂 `error("not yet migrated")`。
- 验收：单元测试 §5 的 api/ + quapi/ 部分通过（从 QuAPI test 拷来先跑）。

### 阶段 3 —— exact API 收编 + ExactIF 分派
- 迁 exactif/test 三套 API 进 `*/exact/`（include 链打断，§2.4）；
  `_bath_eig`/`_quad_nodes`/`_boson_modes`/Nambu 闭式重构为 `EigenSolver`
  求解器接口（§2.8）；接通 `_*(bath; scheme=ExactIF(…))` 分派（§3）。
- exports：`ExactIF`、`EigenSolver`、`AnalyticSolver`、
  `ToulouseGreenFunctionSolver`、`DiscretizationScheme`、`FrozenKernel`、
  `exact_*` 全部函数。
- 验收：test/exact/ 全绿；**对拍锚**——`Δτ(bath; scheme=ExactIF(EigenSolver(64)))`
  与 exactif 原版 `exact_fermionic_Δτ(bath.spectrum; M=64)` 逐位一致
  （三统计 × 三轮廓）。

### 阶段 4 —— 测试整改（§5 细目）
- 新 `test/runtests.jl`；删 dense/deprecated 相关；exactif 测试搬家适配。
- 验收：IFDiscretization 全量测试通过；记录与 QuAPI 旧版的对拍数字。

### 阶段 5 —— QuAPI 薄壳化 + 下游回归
- QuAPI.jl 改为：`using IFDiscretization` + `@reexport`（或逐符号
  `using … : …`），保留 `Dense`、`deprecated`、`infinitetime`（若 D3 选留）的
  原地定义；**删除 `src/exactfermionic.jl` 及其 export 行（§0.2）**、
  删除已迁走文件的重复定义，旧符号全部转 reexport。
- 下游全量测试（见 §6 清单）。
- 验收：TEMPO / GTEMPO / InfiniteTEMPO / InfiniteGTEMPO / GTEMPOProjects /
  GaussianStates ext 测试全绿，**零下游代码改动**。

### 阶段 6 ——（可选收尾）
- D3 落地（infinitetime 迁入）、ext 迁入（D4）、`AnalyticSolver` 实装
  （接 ImpurityModelBase Toulouse 解析解）、`test/quapi/archive_regression.jl`
  的 performance 脚本切到新 API 后恢复存档回归、QuAPI 退役评估。

---

## 5. 测试迁移明细

### 删除
| 项 | 理由 |
|---|---|
| `test/dense/accessor_vs_dense.jl`、`dense/imagtime_dense_vs_quapi.jl`、`dense/realtime_dense_vs_quapi.jl` | dense 不迁移 |
| `test/api/{imagtime,realtime,mixedtime}.jl` 中所有 `*_deprecated` 对照块（imagtime.jl:12,59；realtime.jl:66-70,138；mixedtime.jl:12,55） | deprecated 不迁移 |
| `runtests.jl:9`（`using QuAPI.Dense`）、`runtests.jl:12`（deprecated names）、`:50-51`（dense includes） | 随上 |
| `test/quapi/exactfermionic.jl` + `runtests.jl:33-38`（含 2026-10-31 屏蔽注释） | `src/exactfermionic.jl` 删除，被 exact API 接替（§0.2） |
| `_runner_x3.jl` 同样引用 | 该 runner 若保留须同步改 |

### 迁移并修改
| 项 | 处置 |
|---|---|
| `test/api/{imagtime,realtime,mixedtime,bcs}.jl`、`api/runtests.jl` | 去 deprecated 块后原样迁 `test/api/` |
| `test/quapi/{imag,real,mixed,bcsimag,bcsreal,bcsmixed}.jl` | 原样迁（其中对 Dense 的引用仅出现在注释，无需改代码；顶层 `using QuAPI.Dense` 删除） |
| **exactif/test → test/exact/**（rev 3，逐文件核实过研究线符号依赖） | |
| ├ `exact_bosonic_check.jl` | 0 依赖，`include(exact_bosonic_api.jl)` → `using IFDiscretization` |
| ├ `exact_bcs_check.jl` | 用 `mb_generator`（brute.jl）做 Fock ED 一级锚——把 `mb_generator`（~15 行 JW 生成元）拷入 `test/exact/_anchor_utils.jl`，注明出处 |
| ├ `api_consistency.jl`、`api_mixed_check.jl`、`mx_chk.jl` | 0 依赖，include 链 → `using IFDiscretization` |
| ├ `mx_mb_anchor_boson.jl` | 0 研究线依赖（只 include exactif_api 链），同上 |
| ├ `toulouse_greens{,_imagtime,_mixedtime}.jl`、`toulouse_bosonic_greens.jl`、`toulouse_bcs_greens_mixedtime.jl` | 只用 `gauss_legendre`（经 include 链）；include 链打断后 `gauss_legendre` 从 IFDiscretization 取（含文件顶部 `include(../brute.jl)`/`include(../volterra.jl)` 行删除） |
| `test/osee.jl` | **rev 6 已迁**（D4 更新）：ext 随 ifmatrix 层迁入；核心测试集（purify/解析锚，55 断言）进 test/osee.jl，稠密张量锚段留 QuAPI/test/osee.jl |
| `test/quapi/archive_regression.jl` | 一期**不迁**（它驱动 `performance/` 脚本产出 CSV 逐字节比对；performance 留 QuAPI）。阶段 6 把 performance 脚本切到 `Δt(f; scheme=QuAPI())` 后作为最终防线恢复，存档 CSV 不应有一字节变化 |

### 留守 QuAPI/exactif（研究线验证，不迁）
`exactif/runtests.jl`（L0–L8：Pfaffian、Volterra 收敛阶、谱代数 = 显式矩阵、
G 块结构、连续谱收敛、端到端 ρ_imp、结构定理、Wick/环路表示、连续热库端到端）
与 `exactif/test/mx_mb_anchor.jl`（混合装配的多体逐位锚，include brute.jl）。
它们锚的是 Ng–Reichman 研究线自身，与 exact API 迁移解耦。

### 新增
- `test/exact/scheme_dispatch.jl`：scheme 分派对拍（rev 4 bath-first）——
  同一 bath 上 `Δx(bath; scheme=QuAPI()) ≡ *_Δx(bath.spectrum; …)`、
  `Δx(bath; scheme=ExactIF(EigenSolver(M))) ≡ exact_*_Δx(bath.spectrum; M=M)`
  （三统计 bath 类型 × 三轮廓 × 逐位）；并断言三统计 bath 类型分派正确
  （boson bath 得 bosonic 路线、fermion bath 得 fermionic 路线、BCS bath 得 bcs 路线）。
- `test/api/scheme_dispatch.jl`：默认参数兼容——`Δτ(bath; N=n)` 与旧
  `Δτ(bath; N=n)` 结果一致（回归保护）。

---

## 6. 下游影响与兼容策略

| 下游 | 用法 | 影响 |
|---|---|---|
| TEMPO | `using ImpurityModelBase, QuAPI`；`import QuAPI: branch`；注释注明 BosonicContourIndex 来自 QuAPI | QuAPI reexport IFDiscretization 后零改动 |
| GTEMPO | `@reexport using QuAPI`；`import QuAPI: branch`；`correlationfunction.jl` 依赖 `ImagCorrelationFunction{T}` 类型参数约定 | 同上；§2.6 保证 |
| InfiniteTEMPO / InfiniteGTEMPO | `using QuAPI: InfiniteImagCorrelationCache, InfiniteRealCorrelationCache` | 决策点 D3：这两个 cache 一期留 QuAPI 则零改动；若迁入则 QuAPI reexport |
| GTEMPOProjects（6 文件，~60 处调用） | 直接调 `fermionic_Δτ/bosonic_Δτ/bcs_Δ*` 及 bath 便捷式 | 具体函数名保留 ⟹ 零改动 |
| GaussianStates | QuAPI 的 package ext（weakdeps） | D4：ext 一期留 QuAPI ⟹ 零改动 |
| SU2GTEMPO test | 引用 bosonic_Δ* | 同 GTEMPOProjects |
| （无）`ExactGrassmannIF` 用户 | QuAPI 之外零引用（已核实，仅 benchmark 脚本注释） | §0.2 删除无下游影响 |

**兼容总原则**：过渡期 QuAPI = `IFDiscretization` 的 reexport 壳 + 未迁移
遗留（Dense/deprecated/infinitetime/ext），下游零改动；
最终是否让下游直接 `using IFDiscretization` 由阶段 6 另行决定。

### 开放决策点汇总
- **D1** ~~`src/exactfermionic.jl` 去向~~ **已裁决（rev 2）：直接删除**，
  exact API 接替（§0.2）。
- **D2** `bcscorr.jl` 放 `corrstorages/`（推荐，存储容器语义）还是 `bcs/` 根。
- **D3** `infinitetime`（Infinite*CorrelationCache，InfiniteTEMPO/InfiniteGTEMPO 依赖）：
  一期留 QuAPI（推荐，改动面最小）或随迁 + reexport。
- **D4** ~~ext + osee~~ **已裁决（rev 6）：ext 随 ifmatrix 层迁入**；
  osee 的稠密张量锚段（BCS SVD、BosonicPT d→∞ 等）留 QuAPI/test/osee.jl。
- **D5** ~~exactfermionic 12 失败在途~~ **随 §0.2 删除失效**，无需归因。
- ~~exactif 研究线并入~~ **已裁决（rev 3）：不并入**，留 QuAPI/exactif 原地（§0.1）。

---

## 7. 风险清单

| 风险 | 缓解 |
|---|---|
| `gauss_legendre` 两份拷贝（IFDiscretization/utility.jl 与 exactif/volterra.jl）独立漂移 | 两份文件头互注出处与日期；exactif 原件本就冻结为研究代码 |
| exact_bcs_check 的 `mb_generator` 锚拷贝与 brute.jl 原件漂移 | 锚属测试夹具，行为由测试本身逐位锚定，漂移即测试失败，自暴露 |
| mixedtime 对 realtime/imagtime helper 的隐式跨文件依赖迁移时漏搬 | §2.1：helper 集中 `quapi/util.jl`；逐符号 grep 验证 |
| `src/imagtime/util.jl` 中 deprecated 专用 helper 被误迁 | §2.2：逐个核对调用方 |
| bcs→fermions 的 include 依赖倒置 | include 序 fermions 先于 bcs；测试覆盖 bcs_Δ* |
| exact API include 链打断后漏掉隐式依赖（`spectrumshift` 等） | §2.4 迁移时逐文件 grep `QuAPI\.`/裸符号；test/exact 全绿兜底 |
| exact 核 EigenSolver 化时 M/nbath 语义或节点排布变化导致数值漂移 | 阶段 3 逐位对拍锚（同 M 同排布） |
| `exact_fermionic_*` 语义切换（ExactGrassmannIF → 标准 CorrelationFunction）被下游无声依赖 | §0.2 已核实 QuAPI 之外零引用；CHANGELOG 注明 |
| QuAPI 薄壳化时 export 清单遗漏 → 下游 NameError | 阶段 5 前脚本比对两包 export 名单；下游全量测试 |
| **`QuAPI` struct 名与 QuAPI 模块名撞车**（模块 QuAPI 无法 `export QuAPI`，reexport 时报 conflicting name） | 阶段 5 薄壳改用 `using IFDiscretization: …` 逐符号导入并把 struct 留在 IFDiscretization 命名空间，下游以 `IFDiscretization.QuAPI()` 或别名（如 `const QuAPIScheme`）取用；实施阶段 5 时先定别名方案 |
| archive_regression 与 performance 脚本路径耦合 | 一期不迁、阶段 6 恢复（§5） |
| ImagCorrelationFunction{T} 参数约定被无意改动 | §2.6：struct 定义逐字不动 |
| exact API 文件在 exactif/test 内仍可能被并行会话改动 | 阶段 3 开工前对 exactif 打快照（或先冻结这三个文件），迁出以快照为基线 |
| 并行在途改动（contourindex 交互） | 开工前 rebase 对齐 |

---

## 8. 工作量粗估

| 阶段 | 内容 | 量级 |
|---|---|---|
| 0 | 骨架 | 0.5 天 |
| 1 | corrstorages + 基础设施 | 0.5 天 |
| 2 | quapi/ 三目录 + 统一入口 | 1–1.5 天（helper 重组是主要工作） |
| 3 | exact/ 三目录 + EigenSolver 化（无研究线并入，较 rev 2 减负） | 1–1.5 天 |
| 4 | 测试整改 | 1 天 |
| 5 | QuAPI 薄壳 + exactfermionic 删除 + 下游回归 | 0.5–1 天（下游测试跑得慢） |
| 6 | 可选收尾（AnalyticSolver 实装等） | 另议 |
