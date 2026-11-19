# ============================================================================
# IFDiscretizationGaussianStatesExt —— IFDiscretization × GaussianStates 扩展
#
# ← QuAPI/ext/QuAPIGaussianStatesExt.jl 原样移植（构造器未动）。
# 2026-11-19：ifmatrix 稠密装配层从包核心删除（包不再负责稠密装配），本扩展
# 作为该能力的唯一残留消费方（GaussianStates 桥接），在模块内自携
# correlation function → Grassmann/Bosonic 杂交矩阵装配 + 高斯态构造器；
# test/osee.jl 经本扩展消费（using 其模块命名空间）。顶点移位装配同日全量
# 移除：所有轮廓 raw（cell 值）直构。
#
# 由 hybridization function 直接构造影响泛函（IF）系数态的高斯态表示：
#
#   I[ā, a] = exp[Σ_pq ā_p Δ_pq a_q]     （Δ 取 IFDiscretization 约定：指数负号已吸收）
#
# → 系数态 = doubled 空间的纯配对高斯向量 |Ψ⟩ = exp(Σ_pq Δ_pq ā†_p a†_q)|0⟩，
#   按 GaussianStates v7「态只存实协方差」的约定表示：
#
#   fermionic：FermionicGaussianState(corr)  →  Γ = i⟨γγ⟩   （Majorana，4n×4n）
#   bosonic：  BosonicGaussianState(corr)    →  Σ = ½⟨{R,R}⟩ （(x,p)，4n×4n）
#
#   OSEE = entropy(s, modes)（GaussianStates v7 接口）：doubled 模式序
#   [a₁..aₙ; ā₁..āₙ]（模 1..n = ket contour 节点、n+1..2n = bra），
#   **site p 的割 = 模式对 [p, n+p]**，即 entropy(s, vcat(sites, n .+ sites))。
#
# 配对 → 协方差（一般路径，无 pairing 特殊分支；任意 band——BCS 反常块
# 含内；任意奇异度）：系数态的湮灭子 ξ_a = β_a − Σ_b W_ab β†_b
#（W = C − Cᵀ 一般反对称核）恒湮灭该态，湮灭子 Nambu 矩阵 V = [I, −W]
# 恒满行秩。Nambu 协方差 = V 行空间投影子（purify 内部同款的
# 湮灭子→投影子数学，见 F_utilities/FermionicGaussianStates 讲义 §5），
# 经 GaussianStates 的 `to_majorana` 变实 Majorana 协方差。零耦合
#（W = 0）良定义（投影子 = Nambu 恒等，真空态）。
# 锚：单 flavor 与 `purify`（Choi 路线，覆盖 ā–a 型配对）熵逐位互证；
# BCS 一般核以稠密 IF 张量的前缀割 SVD 熵为锚（QuAPI/test/osee.jl 侧，
# 与本扩展共用同一构造器）。
# 两个重载按统计性质区分（CorrelationFunction 类型本身不分玻色/费米，
# 由构造器名承担统计信息）：
#   fermionic：Δ[ā_p, a_q] 直接配对（复 Δ、任意 contour 均可）。
#   bosonic：只支持**实 Δ**（虚时 Matsubara contour；实时/混合 contour 的
#     复 Δ 对应复协方差，快速算法失效，报错）。直接配对振幅 exp(+Δ) 奇异值
#     > 1 ⟹ I−W² 不定、公式数值失败；取 M = exp(−Δ)（奇异值 = e^{−Re λ(Δ)}
#     < 1）——与 exp(+Δ) 只差每模的粒子-空穴反射，时间割的 Schmidt 谱相同、
#     熵不变（TEMPO/osee 报告 §5.2；几何分布 p_n ∝ t^{2n} 在 t ↔ 1/t 下不变）。
#
# Julia 1.10 下 extension 新名字不并入父包命名空间，故本扩展只给
# GaussianStates 的既有名字（两个 GaussianState 构造器）加方法。
# ============================================================================
module IFDiscretizationGaussianStatesExt

using IFDiscretization: AbstractCorrelationFunction, AbstractMixedCorrelationFunction,
                        AbstractRealCorrelationFunction, BCSCorrelationFunction,
                        ImagCorrelationFunction, MixedCorrelationFunction,
                        RealCorrelationFunction, CorrelationMatrix,
                        GrassmannContourIndex, BosonicContourIndex,
                        branch, branches, isize, rsize
using GaussianStates
using GaussianStates: FermionicGaussianState, BosonicGaussianState
using LinearAlgebra

# ------------------------------------------------------------------ #
# 稠密装配（← src/ifmatrix/ifmatrix.jl 原样迁移，包核心不再携带）
# ------------------------------------------------------------------ #

# 测试/桥接消费的稠密装配名（模块内自携，经 Base.get_extension + using 访问）
export GrassmannHybridizationMatrix, BosonicHybridizationMatrix, BosonicSplitDelta,
       basis, contour_ordering

# =============================================================================
# ifmatrix —— corr → 配对（杂交）矩阵层：GaussianStates 扩展的底座
#
# ← QuAPI/src/dense/{common,fermion,boson}.jl 中「correlation function →
#   HybridizationMatrix」的转换层原样移植（kernel 未动）。稠密 IF **张量**
# 机器（GrassmannADT/GrassmannIF/BosonicADT/BosonicPT/Zvalue/margin/乘法）
# 不在移植范围（QuAPI.Dense 留守）；本层只含：
#   · contour_ordering / basis —— 统一 contour 节点序与腿基
#   · GrassmannHybridizationMatrix（含顶点位移装配，2026-11-04 与 QuAPI.Dense
#     同构收口：实时单 flavor 与 BCS RT 保留位移，τ/mixed raw 读取）——
#     费米 corr → (腿基, C)
#   · BosonicHybridizationMatrix / BosonicSplitDelta —— 玻色 corr → Δ
#   · 玻色 OSEE 只消费 split=false 的普通矩阵；BosonicSplitDelta 供 PT
#     构造（QuAPI.Dense 的 BosonicIF 用），随类型一并保留。
# ============================================================================

# ------------------------------------------------------------------ #
# contour Δ（hybridization 矩阵）组装
#
# 节点行序：分支 [:+, :-, :τ] 逐分支、分支内时间升序；整条 contour 的 Δ
# 是分支块的块矩阵。元素类型由 correlation function 决定（虚时通常为实，
# 实/混合含虚部），这里不写死为复数。
# ------------------------------------------------------------------ #

# 按统一的 contour 节点序（contour_ordering）把分支块重排成扁平矩阵。
# blocks[(b1,b2)] 是 branch(x; b1=b1, b2=b2) 的矩阵（分支内指标 1..N_b）。
function _assemble_ordered(nodes, blocks::AbstractDict)
    T = promote_type(map(eltype, values(blocks))...)
    n = length(nodes)
    M = zeros(T, n, n)
    for (p, (b1, j1)) in enumerate(nodes), (q, (b2, j2)) in enumerate(nodes)
        M[p, q] = blocks[(b1, b2)][j1, j2]
    end
    return M
end

function _contour_delta(x::AbstractCorrelationFunction)
    brs = branches(x)
    blocks = Dict((b1, b2) => Matrix(branch(x; b1=b1, b2=b2))
                  for b1 in brs, b2 in brs)
    return _assemble_ordered(contour_ordering(x), blocks)
end

# Δ 的算符次序拆分 M1 + M2 = Δ：
# CorrelationMatrix 块区分两种等时算符次序——M1 取 ηⱼₖ（含等时 ηⱼₖ[1]），
# M2 取 ηₖⱼ（含等时 ηₖⱼ[1]）。普通矩阵块（实/混合 contour 的 ζ 交叉块）
# 无次序信息，按上三角 / 严格下三角拆分（等时项归入 M1）。
function _split_block(B::CorrelationMatrix)
    m = size(B, 1)
    T = eltype(B)
    A1 = zeros(T, m, m)
    A2 = zeros(T, m, m)
    for i in 1:m, j in 1:m
        if i > j
            A1[i, j] = B.ηⱼₖ[i-j+1]
        elseif i < j
            A2[i, j] = B.ηₖⱼ[j-i+1]
        else
            A1[i, i] = B.ηⱼₖ[1]
            A2[i, i] = B.ηₖⱼ[1]
        end
    end
    return A1, A2
end
_split_block(B::AbstractMatrix) = (Matrix(triu(B)), Matrix(tril(B, -1)))

function _contour_delta_split(x::AbstractCorrelationFunction)
    brs = branches(x)
    parts = Dict((b1, b2) => _split_block(branch(x; b1=b1, b2=b2))
                 for b1 in brs, b2 in brs)
    nodes = contour_ordering(x)
    M1 = _assemble_ordered(nodes, Dict(k => v[1] for (k, v) in parts))
    M2 = _assemble_ordered(nodes, Dict(k => v[2] for (k, v) in parts))
    return M1, M2
end

# ------------------------------------------------------------------ #
# 腿基
# ------------------------------------------------------------------ #

# 稠密 IF 的输入：普通 correlation function 或 BCS 包装（后者不是
# AbstractCorrelationFunction 的子类型）
const IFSource = Union{AbstractCorrelationFunction,BCSCorrelationFunction}

# flavor（band）个数：BCS bath 有 2 个
_bands(::AbstractCorrelationFunction) = 1
_bands(::BCSCorrelationFunction) = 2

"""
    contour_ordering(x::Union{AbstractCorrelationFunction,BCSCorrelationFunction})
        -> Vector{Tuple{Symbol,Int}}

contour 节点的**统一序**，返回 `(branch, j)` 列表。它同时定义
Grassmann/玻色 IF 张量的轴序与 `_contour_delta` 的行列序：

```text
IM : (:τ,1) (:τ,2) .. (:τ,Nτ)                    （虚时升序）
RT : (+,1) (-,1) (+,2) (-,2) ..                  （时间升序，+ 先 − 后）
MX : (:τ,1) .. (:τ,Nτ)  ++  (+,1) (-,1) ..       （虚时升序接实时升序）
```

费米多 band 时，`basis` 在每个节点内按 band 1,2,… 展开，每个 band 内
`a`、`ā` 相邻（`ā` 紧跟 `a`）。
"""
contour_ordering(x::ImagCorrelationFunction) =
    [(:τ, i) for i in 1:size(x, 1)]
contour_ordering(x::AbstractRealCorrelationFunction) =
    [(b, i) for i in 1:size(x, 1) for b in (:+, :-)]
function contour_ordering(x::AbstractMixedCorrelationFunction)
    kt, Nτ = rsize(x), isize(x)
    return vcat([(:τ, i) for i in 1:Nτ],
                [(b, i) for i in 1:kt for b in (:+, :-)])
end
contour_ordering(x::BCSCorrelationFunction) = contour_ordering(x.uu)

"""
    basis(x::Union{AbstractCorrelationFunction,BCSCorrelationFunction})
        -> Vector{GrassmannContourIndex}

稠密 Grassmann 影响泛函的腿基：节点按 [`contour_ordering`](@ref) 排列，
每个节点内逐 band（1, 2, …）展开，每个 band 内 `a`、`ā` 相邻（`ā` 紧跟 `a`）。
BCS bath 每节点 4 条腿（两个 band）。

单 flavor 的腿数为 `2n`（n 为 contour 节点数），BCS 为 `4n`。
"""
function basis(x::IFSource)
    nb = _bands(x)
    return GrassmannContourIndex[GrassmannContourIndex(j, b, band, c)
                                 for (b, j) in contour_ordering(x)
                                 for band in 1:nb for c in (false, true)]
end

# ------------------------------------------------------------------ #
# 费米：GrassmannHybridizationMatrix（含顶点位移装配）
# ------------------------------------------------------------------ #

"""
    GrassmannHybridizationMatrix{T<:Number}

费米稠密 IF 的杂交矩阵：`basis` 为链序腿列表（见 [`basis`](@ref)），
`Δ` 为这些腿上的双线性系数矩阵（`Δ[i,j]` 是 ξ_i ξ_j 的系数；普通
bath 给 `ā Δ a`，BCS 还含反常 `āā`、`aa` 块）。

用 `GrassmannHybridizationMatrix(corr)` 从 correlation function 计算。
GaussianStates 扩展（IF 系数态构造）消费本类型的 `Δ` 与 `basis`。
"""
struct GrassmannHybridizationMatrix{T<:Number}
    basis::Vector{GrassmannContourIndex}
    Δ::Matrix{T}
end

# ------------------------------------------------------------------ #
# 顶点移位装配已删除（2026-11-19）：全轮廓 raw（cell 值）直构——严格核
# （ExactKernel 族）契约即 raw 指数系数，两套口径并存曾致消费方按容器类型
# 静默分叉（GHM(ExactReal) 套移位损坏 0.039 量级）。
# ------------------------------------------------------------------ #

GrassmannHybridizationMatrix(x::IFSource) =
    GrassmannHybridizationMatrix(basis(x), _if_coefficients(x))

# IF 指数中的 Δ 项：((band₁, conj₁), (band₂, conj₂), Δ 块)
# 普通费米高斯 bath：ā Δ a
_delta_terms(x::RealCorrelationFunction) =
    [((1, true), (1, false), _contour_delta_realtime(x))]
# 混合轮廓与虚时轮廓：直接消费原块（历史约定，ED 校准）
_delta_terms(x::AbstractCorrelationFunction) =
    [((1, true), (1, false), _contour_delta(x))]
# 实时轮廓（RealCorrelationFunction）：Grassmann IF 指数使用 Keldysh 方位
# 未吸收的交叉块（历史存储约定；统一约定把 σ_uσ_v 放进了块内，指数侧需
# 翻回）——混合轮廓（AbstractMixed）的 dense 装配经 ED 校准直接消费原块。
function _contour_delta_realtime(x::RealCorrelationFunction)
    blocks = Dict((b1, b2) => begin
            m = Matrix(branch(x; b1=b1, b2=b2))
            (b1 == b2) || (m .*= -1)
            m
        end for b1 in (:+, :-), b2 in (:+, :-))
    return _assemble_ordered(contour_ordering(x), blocks)
end
# BCS：正常 ↑↑ (ā↑Δa↑)、正常 ↓↓ (ā↓Δa↓，自旋对称下与 ↑↑ 同块)、
# 反常 ↑↓ (ā↑Δā↓)、反常 ↓↑ (a↓Δa↑)
_delta_terms(x::BCSCorrelationFunction) =
    [((1, true), (1, false), _contour_delta(x.uu)),
     ((2, true), (2, false), _contour_delta(x.uu)),
     ((1, true), (2, true), _contour_delta(x.ud)),
     ((2, false), (1, false), _contour_delta(x.du))]

function _if_eltype(x::IFSource)
    return promote_type(map(t -> eltype(t[3]), _delta_terms(x))...)
end

# 双线性指数矩阵 C（腿序即 basis(x)），C[i, j] 为 ξ_i ξ_j 的系数。
# Δ 的行列序与 contour_ordering 一致，故节点 p ↔ Δ 的第 p 行/列。
function _if_coefficients(x::IFSource)
    legs = basis(x)
    T = _if_eltype(x)
    L = length(legs)
    idx = Dict(l => k for (k, l) in enumerate(legs))
    chain = contour_ordering(x)
    C = zeros(T, L, L)
    for (t1, t2, Δ) in _delta_terms(x)
        for (p, (b1, j1)) in enumerate(chain), (q, (b2, j2)) in enumerate(chain)
            c = Δ[p, q]
            iszero(c) && continue
            i = idx[GrassmannContourIndex(j1, b1, t1[1], t1[2])]
            j = idx[GrassmannContourIndex(j2, b2, t2[1], t2[2])]
            C[i, j] += c
        end
    end
    return C
end

# ------------------------------------------------------------------ #
# 玻色：BosonicHybridizationMatrix / BosonicSplitDelta
# ------------------------------------------------------------------ #

"""
    BosonicSplitDelta

contour 杂交矩阵 `Δ = M1 + M2` 的**等时算符次序拆分**（`CorrelationMatrix`
块用 `ηⱼₖ`/`ηₖⱼ` 区分等时两种次序），仅供 QuAPI.Dense 的 `BosonicIF` PT 构造。

它与 BdG 的 `(h, g)` 块（粒子-空穴 / 配对）含义无关，故用独立类型包装：
不会被误当作两个矩阵 splat 进二次型算子的 `(h, A)` 块。
"""
struct BosonicSplitDelta{T<:Number}
    M1::Matrix{T}
    M2::Matrix{T}
end

function BosonicSplitDelta(M1::AbstractMatrix, M2::AbstractMatrix)
    T = promote_type(eltype(M1), eltype(M2))
    return BosonicSplitDelta{T}(Matrix{T}(M1), Matrix{T}(M2))
end

"""
    BosonicHybridizationMatrix

玻色稠密 IF 的杂交矩阵：`basis` 为链序节点，`Δ` 为 contour 杂交矩阵
（节点行序 `[:+; :-; :τ]`）。

- `split=false`（默认）：`Δ = M1 + M2`（普通矩阵）；
- `split=true`：`Δ = ` [`BosonicSplitDelta`](@ref)`(M1, M2)` 为算符次序拆分。

玻色 OSEE（GaussianStates 扩展）只消费 `split=false` 的普通矩阵。
"""
struct BosonicHybridizationMatrix{T}
    basis::Vector{BosonicContourIndex}
    Δ::T
end

function BosonicHybridizationMatrix(x::AbstractCorrelationFunction; split::Bool=false)
    chain = contour_ordering(x)
    bs = BosonicContourIndex[BosonicContourIndex(j, br) for (br, j) in chain]
    Δ = split ? BosonicSplitDelta(_contour_delta_split(x)...) : _contour_delta(x)
    return BosonicHybridizationMatrix(bs, Δ)
end

# ------------------------------------------------------------------ #
# fermionic：IF 系数态（实 Majorana 协方差；单 flavor 与 BCS 多 band 通用）
# ------------------------------------------------------------------ #

"""
    FermionicGaussianState(x::AbstractCorrelationFunction) -> FermionicGaussianState
    FermionicGaussianState(h::GrassmannHybridizationMatrix) -> FermionicGaussianState

费米 hybridization function `x` 的 IF 系数态
`|Ψ⟩ = exp(Σ_ab C_ab β†_a β†_b)|0⟩`（Grassmann 有限和；任意 contour、
任意 band 数——含 BCS 反常块——恒良定义），以 v7「只存实协方差」的
`FermionicGaussianState` 表示：`Γ = i⟨γγ⟩` 实反对称，`nmodes(s) = L`
（L = 腿数：单 flavor 2n、BCS 4n）。

一般构造 = 湮灭子投影子（无 pairing 特殊分支）：W = C − Cᵀ（一般反对称
核），V = [I, −W] 湮灭子 Nambu 矩阵，Nambu 协方差 = V 行空间投影子，
`to_majorana` 变实 Γ；零耦合良定义。单 flavor 与 `purify`（Choi 路线）
熵逐位互证；BCS 一般核以稠密 IF 张量前缀割 SVD 熵为锚（QuAPI/test/osee.jl）。

OSEE 割约定：模式 = 腿重排为「全部 a 腿（contour 序），全部 ā 腿」；
单 flavor n 节点即 site p ↔ [p, n+p]（同旧约定）：

```julia
s = FermionicGaussianState(corr)
n = nmodes(s) ÷ 2                              # a 腿（ket 半区）模式数
entropy(s, vcat(sites, n .+ sites))            # site p ↔ 模式对 [p, n+p]
```
"""
function GaussianStates.FermionicGaussianState(h::GrassmannHybridizationMatrix)
    C = Matrix(h.Δ)
    L = size(C, 1)
    # 一般路径（无 pairing 特殊分支；任意 band——BCS 反常块含内；任意奇异度）：
    # 系数态 |Ψ⟩ = exp(Σ C_ab β†_a β†_b)|0⟩ 的湮灭子 ξ_a = β_a − Σ_b W_ab β†_b
    # （W = C − Cᵀ）恒湮灭该态 ⟹ 湮灭子 Nambu 矩阵 V = [I, −W]（L×2L，恒满行秩，
    # 零耦合 W = 0 亦然）。Nambu 协方差 = V 行空间投影子（purify 内部的同款
    # 湮灭子→投影子数学，F_utilities/FermionicGaussianStates 讲义 §5），再经
    # GaussianStates 的 to_majorana 变实 Majorana 协方差。
    W = C .- transpose(C)
    V = hcat(Matrix{ComplexF64}(I, L, L), -W)
    B = Matrix(qr(adjoint(V)).Q)                   # 2L×L，B†B = I，列空间 = V 行空间
    𝒞 = B * adjoint(B)                             # Nambu 投影子（2L×2L）
    # 模式约定：前 L÷2 = 全部 a 腿（contour 序），后 L÷2 = 全部 ā 腿
    # （单 flavor：site p ↔ [p, n+p]；BCS：L = 4n，2n 模/半区）
    alpleg = findall(l -> l.conj == false, h.basis)
    order = vcat(alpleg, setdiff(1:L, alpleg))     # 腿序 → 模式序
    majorder = vcat([[2p-1, 2p] for p in order]...)# 模式序 → Majorana 序
    Γ = GaussianStates.to_majorana(𝒞)
    Γ = Γ[majorder, majorder]
    return FermionicGaussianState(Γ)
end

GaussianStates.FermionicGaussianState(x::AbstractCorrelationFunction) =
    FermionicGaussianState(GrassmannHybridizationMatrix(x))
GaussianStates.FermionicGaussianState(x::BCSCorrelationFunction) =
    FermionicGaussianState(GrassmannHybridizationMatrix(x))

# ------------------------------------------------------------------ #
# bosonic：IF 系数态（实 (x,p) 协方差；d→∞ 高斯极限）
# ------------------------------------------------------------------ #

"""
    BosonicGaussianState(x::AbstractCorrelationFunction) -> BosonicGaussianState
    BosonicGaussianState(h::BosonicHybridizationMatrix) -> BosonicGaussianState

玻色 hybridization function `x` 的 IF 系数态（d→∞ 高斯极限，TEMPO/osee
报告 §5.2），配对振幅 `M = exp(−Δ)`、模式序 `[a₁..aₙ; ā₁..āₙ]` 的纯配对
向量，以 v7「只存实协方差」的 `BosonicGaussianState` 表示：
`Σ = ½⟨{R,R}⟩` 为 `4n×4n` 实对称正定（辛本征值全 ½，纯态）。

**只支持实 Δ**（虚时 Matsubara contour）；实时/混合 contour 的复 Δ 会使
系数态协方差复化、快速算法失效，调用报 `ArgumentError`。

`M = exp(−Δ)`（奇异值 `e^{−λ_k(Δ)} < 1`）：玻色系数向量 `exp(Σ Δ b†b)`
向量化的直接配对振幅 `exp(+Δ)` 奇异值 > 1，`I − W²` 不定、协方差公式数值
失败；`exp(−Δ)` 与之只差每模粒子–空穴反射，时间割 Schmidt 谱相同、
熵不变（TEMPO/osee 报告 §5.2；几何分布 p_n ∝ t^{2n} 在 t ↔ 1/t 下不变）。

`BosonicHybridizationMatrix` 重载仅接受 `split=false`（`Δ` 为普通矩阵）。
`split=true` 存的是等时算符次序拆分 `BosonicSplitDelta(M1, M2)`——它不是
配对振幅，不能拆成两个矩阵传入，故直接报错。
"""
function GaussianStates.BosonicGaussianState(hm::BosonicHybridizationMatrix)
    Δ = hm.Δ
    Δ isa AbstractMatrix || throw(ArgumentError(
        "该 BosonicHybridizationMatrix 由 split=true 构造，Δ = BosonicSplitDelta(M1, M2) " *
        "是等时算符次序拆分，不是配对振幅；玻色 OSEE 请用 " *
        "BosonicHybridizationMatrix(x)（split=false）或直接 BosonicGaussianState(x)"))
    if maximum(abs, imag.(Δ)) > 1e-12 * max(1.0, maximum(abs, real.(Δ)))
        throw(ArgumentError(
            "玻色 OSEE 只支持实 Δ（虚时 Matsubara contour）；当前 Δ 含非平凡" *
            "虚部（实时/混合 contour），系数态协方差会复化，快速算法失效"))
    end
    M = exp(-real.(Δ))                                # 配对振幅（矩阵指数；谱半径 < 1 ⟹ 可归一）
    n = size(M, 1)
    W = zeros(2n, 2n)                                 # [[0, M], [Mᵀ, 0]]（实对称）
    W[1:n, n+1:2n] .= M
    W[n+1:2n, 1:n] .= transpose(M)
    # 湮灭条件闭式：G = ⟨b b†⟩ = (I − WWᵀ)⁻¹、K = ⟨b b⟩ = W Gᵀ；Σ 块装配
    # （GaussianStates v1 `_bosonic_gamma_pairing` 同式；n=1 与稠密 Fock Choi
    # 仲裁至机器精度，辛本征值全 ½）。
    Id = Matrix{Float64}(I, 2n, 2n)
    G = (Id - W * transpose(W)) \ Id
    K = W * transpose(G)
    Σ = zeros(Float64, 4n, 4n)                        # (x₁,p₁,…,x_{2n},p_{2n})
    for i in 1:2n, j in 1:2n
        δ = ifelse(i == j, 0.5, 0.0)
        Σ[2i-1, 2j-1] = G[i,j] + K[i,j] - δ
        Σ[2i, 2j]     = G[i,j] - K[i,j] - δ
    end
    Σ = (Σ .+ transpose(Σ)) ./ 2
    return BosonicGaussianState(Σ)
end

GaussianStates.BosonicGaussianState(x::AbstractCorrelationFunction) =
    BosonicGaussianState(BosonicHybridizationMatrix(x))

end # module
