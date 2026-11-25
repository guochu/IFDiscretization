# IFDiscretization

**IFDiscretization** 是离散影响泛函（Influence Functional, IF）装配层：以
`DiscretizationScheme` 虚类封装「离散 IF 怎么算」。统计（玻色 / 费米 / BCS
超导）由 bath 对象类型承担（`Δτ / Δt / Δm` 的第一个输入必须是 bath），核
技术由 `scheme` 关键字承担——同一入口可切换 **QuAPI 冻结核**（一阶 Trotter
梯形装配）与 **ExactKernel 严格核**（等距离散 + 本征求解的机器精度闭式 /
Schur 构造）。本包由 QuAPI 忠实移植而来，与上游共享约定。

文档结构：

- **[Guide](guide.md)**：从谱函数与 bath 到三种轮廓杂化矩阵的典型工作流
  （冻结核 → 严格核 → BCS → 内部调度入口）。
- **[API](api.md)**：全部导出 API 的 docstring 汇总（容器、统一入口、
  冻结核、方案与求解器、严格核、工具）。
- **[Contours and Containers](conventions.md)**：三种时间轮廓的几何与
  节点序、`branch` 分支块约定、点值容器与严格容器的槽位语义（含 BCS
  Nambu 块的有向 Wick 形式）。
- **[ExactKernel Contract](exact.md)**：严格核的严格性机制与装配级契约——
  费米实轴 FormB 消元 + BCS 实时 FormB Nambu 翻倍（2026-11-25）、Z 证书
  （normdelta 定义）、金标准测试清单；退役族恢复入口见包 README todolist。

## 全局符号约定

影响泛函（IF）指数中的负号被吸收进杂化矩阵的定义（`src/IFDiscretization.jl`
模块头注），即离散路径权重为

```math
I = \exp\Big[\sum_{ij}\bar\psi_i\,\Delta_{ij}\,\psi_j\Big]
```

而非常见的 ``\exp[-\sum\bar\psi\Delta\psi]``。引用外部文献公式时注意相差
一个负号。配对矩阵的分支块槽位约定为

```math
\Delta^{b_1 b_2}[i, j] \;\widehat{=}\; \text{场}_1@(b_1,i)\ \times\ \text{场}_2@(b_2,j)
```

（row = 第一场的时间/支、col = 第二场），详见
[Contours and Containers](conventions.md)。

## 安装

依赖 [ImpurityModelBase](https://github.com/guochu/ImpurityModelBase)（谱
函数、bath 类型、ED 参考解）。本地开发布局要求几个包并列放置：

```
Missile/
├── IFDiscretization/
└── ImpurityModelBase/
```

`Project.toml` / `Manifest.toml` 中的本地依赖用**相对路径**记录
（如 `path = "../ImpurityModelBase"`），因此在任何机器上克隆后布局一致即
可直接使用，无需修改路径。

加载包：

```julia
using IFDiscretization, ImpurityModelBase
```

## 目录结构

```
src/
├── IFDiscretization.jl    # 模块入口与全部 export
├── schemes.jl             # DiscretizationScheme 类型层级
├── corrstorages/          # 点值容器（Toeplitz/分支块解析存储）
├── exactcorrstorages/     # 严格核装配级配对块容器（ExactReal/ExactMixed）
├── contourindex.jl        # Grassmann/BosonicContourIndex + branch 访问器
├── bosons/  fermions/  bcs/   # 三统计 × {quapi/ 冻结核, exact/ 严格核} + 统一入口
├── defaults.jl  utility.jl
ext/
└── IFDiscretizationGaussianStatesExt.jl   # corr → 高斯 IF 系数态（弱依赖 GaussianStates）
test/
├── api/                   # 语义/约定 + spin-boson ED 对拍 + BCS Grassmann 装配
├── quapi/                 # 冻结核自由模型阶数断言 + 存档逐字节回归
├── exact/                 # ExactEqualDistantKernel 金标准（费米 123+58+37，BCS 17+11+25）
└── osee.jl                # GaussianStates 扩展 OSEE
docs/
└── src/                   # 本文档源
```

## 运行测试

```bash
# 直接运行（推荐，绕过 Pkg.test 的沙箱环境解析；勿与其它 julia 进程并行）
julia --project=. test/runtests.jl

# 标准入口
julia --project=. -e 'using Pkg; Pkg.test()'
```

当前 9958 项全部通过，覆盖：三种统计 × 三轮廓的冻结核与严格核、方案分派、
BCS Fock ED 定义级金标准、费米带踢 Schur 装配契约（Z ≡ 1 机器证书）、
存档逐字节回归、GaussianStates 扩展 OSEE。

## 文档构建

```bash
julia --project=docs docs/make.jl          # 构建到 docs/build（KaTeX 自托管）
julia --project=docs docs/liveserver.jl    # 本地实时预览
```
