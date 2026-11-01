# =============================================================================
# ifmatrix —— corr → 配对（杂交）矩阵层：GaussianStates 扩展的底座
#
# ← QuAPI/src/dense/{common,fermion,boson}.jl 中「correlation function →
#   HybridizationMatrix」的转换层原样移植（kernel 未动）。稠密 IF **张量**
# 机器（GrassmannADT/GrassmannIF/BosonicADT/BosonicPT/Zvalue/margin/乘法）
# 不在移植范围（QuAPI.Dense 留守）；本层只含：
#   · contour_ordering / basis —— 统一 contour 节点序与腿基
#   · GrassmannHybridizationMatrix（含 2026-10-30 顶点位移装配，实时单 flavor
#     保留位移，τ/mixed 回退 cell 读取）——费米 corr → (腿基, C)
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

# --------------------- 顶点位置装配（费米 PT 修正） --------------------- #
# fermionic_Δτ/Δt 的核闭式（胞元积分）逐位等于严格 B-cell IF 的 O(V²) 系数，
# 但旧装配把顶点放在 cell 边界（滞后偏一格），对严格方案的残差因此停在 O(V²)
# （V 减半残差比值 4，performance/fermionicG4.jl 实测）。此处把顶点移到节点
# ——与玻色 non-diagonal PT 在 BosonicIF 里做的分支规则修复属同一层级的
# 装配修正，核数值完全复用：
#
#   虚时 τ 环（生成值重组为 skew-circulant，N 为节点数）：
#     c₀ = ηₖⱼ[2]，c₁ = ηⱼₖ[1]+ηₖⱼ[1]（等时和），c_k = ηⱼₖ[k]（k ≥ 2）
#     C[āᵢ, aⱼ] = i > j ? c_{i−j} : i < j ? −c_{N+i−j} : c₀
#   实时 Keldysh（节点 0..N，腿位 pos(j,br,ā) = 4j + (br==:+ ? 0:2) + (ā?2:1)）：
#     (ā⁺,a⁻)/(ā⁻,a⁺) 交叉块：值不变，仅开窗（ā⁺₀ 行、a⁻₀ 列 / a⁺_N 列、
#       ā⁻_N 行置零——顶点窗口 = cell 1..N）
#     (ā⁺,a⁺) 块：列移一位（aⱼ→aⱼ₊₁）；动能位 (j, j−1) 取 (ā⁻,a⁻) 块对角
#     (ā⁻,a⁻) 块：镜像（列移 aⱼ→aⱼ₋₁；动能位 (j−1, j) 取 (ā⁺,a⁺) 块对角）
#
# 精度：核内容对严格方案（exact_fermionic_Δτ/Δt 参照）的残差按 V⁴ 标度
# （比值≈16）——与玻色 non-diagonal PT 同级（O(V²) 系数精确 + V⁴ 截断）。
# 整体 Z/ρ 另带一个 O(V²) 归一化常数（"跳去又跳回"泄漏 = exact_fermionic_*
# 的 prefactor；迹归一化的观测量中自动消去）。
#
# 2026-10-30 生效范围收缩（与 contourindex.jl 访问器逐轮廓对齐）：仅
# 实时 Keldysh（单 flavor）保留顶点位移（ED 验证通过、误差常数好 ~25×）。
# τ 环位移使 g(τ) 离散误差常数恶化 ~30×（GTEMPO Toulouse 基准）、mixed
# 位移与物理 ED 矛盾 6–7% 且不随 δt 收敛（GTEMPO 全部 mixed ED 套件）——
# 两者连同 BCS τ 均回退 cell（节点值）读取（证据与复现：
# bugrepots/dense_mixed_vertexshift_convention.md）；BCS mixed 的位移与
# 访问器互一致（严格 BCS 链验证待补）；BCS RT 暂沿用旧装配。


function _vertex_shifted_rt(C::AbstractMatrix{T}, N::Int) where {T}
    pos(j0, br, ā) = 4 * j0 + (br === :+ ? 0 : 2) + (ā ? 2 : 1)
    diag₊₊ = C[pos(0, :+, true), pos(0, :+, false)]   # (+,+) 块对角（Toeplitz 常值）
    diag₋₋ = C[pos(0, :-, true), pos(0, :-, false)]   # (−,−) 块对角
    D = zeros(T, 4 * (N + 1), 4 * (N + 1))
    for j1 in 0:N, j2 in 0:N
        j1 ≥ 1 && j2 ≥ 1 &&                       # (ā⁺,a⁻)：值不变，开窗（行 ā⁺₀ / 列 a⁻₀）
            (D[pos(j1, :+, true), pos(j2, :-, false)] = C[pos(j1, :+, true), pos(j2, :-, false)])
        j2 ≤ N - 1 && j1 ≤ N - 1 &&               # (ā⁻,a⁺)：值不变，开窗（列 a⁺_N / 行 ā⁻_N）
            (D[pos(j1, :-, true), pos(j2, :+, false)] = C[pos(j1, :-, true), pos(j2, :+, false)])
        if j1 ≥ 1 && j2 ≤ N - 1                 # (ā⁺,a⁺)：列移一位，动能位取本支对角
            D[pos(j1, :+, true), pos(j2, :+, false)] =
                j2 == j1 - 1 ? diag₊₊ : C[pos(j1, :+, true), pos(j2 + 1, :+, false)]
        end
        if j1 ≤ N - 1 && j2 ≥ 1                 # (ā⁻,a⁻)：镜像（动能位取本支对角）
            D[pos(j1, :-, true), pos(j2, :-, false)] =
                j2 == j1 + 1 ? diag₋₋ : C[pos(j1, :-, true), pos(j2 - 1, :-, false)]
        end
    end
    return D
end


# 混合轮廓链位：τ_j（j=1..Nτ）在前，随后按时间 jj=1..Nt+1 的 (+,−) 对
function _mxnode(b::Symbol, j::Int, Nτ::Int)
	b === :τ && return j
	return Nτ + 2 * (j - 1) + (b === :+ ? 1 : 2)
end
_mjran(b::Symbol, Nτ::Int, Nt::Int) = b === :τ ? (1:Nτ) : (1:Nt+1)

# 行/列源与符号（单 flavor 混合轮廓；BCS 混合逐 band 同构复用）
function _mx_maps(b1::Symbol, j1::Int, b2::Symbol, j2::Int, Nτ::Int, Nt::Int)
	r = b1 === :τ ? (j1 ≥ 2 ? (:τ, j1 - 1) : nothing) :
		b1 === :+ ? (j1 == 1 ? (:τ, Nτ) : (:+, j1 - 1)) :
		(j1 ≤ Nt ? (:-, j1) : nothing)
	c = b2 === :τ ? (:τ, j2) :
		b2 === :+ ? (:+, j2) : (j2 ≥ 2 ? (:-, j2 - 1) : nothing)
	s = ((b1 === :+ && j1 ≥ 2) ⊻ (b2 === :+)) ? -1 : 1
	return r, c, s
end

function _vertex_shifted_bcs_mixed(C::AbstractMatrix, Nτ::Int, Nt::Int)
	# BCS 混合（每节点 a↑,ā↑,a↓,ā↓ 四腿）：正常块 (ā↑,a↑)/(ā↓,a↓) 与反常块
	# (ā↑,ā↓)/(a↓,a↑) 按单 flavor 规则逐块同构移位（严格 BCS 链验证待补）
	ipos(b, j, band, ā) = 4 * (_mxnode(b, j, Nτ) - 1) + (band - 1) * 2 + (ā ? 2 : 1)
	D = zero(C)
	blocks = (((1, true, 1, false), (1, true, 1, false)),   # (ā↑,a↑)=uu
			  ((2, true, 2, false), (2, true, 2, false)),   # (ā↓,a↓)=dd
			  ((1, true, 2, true), (1, true, 2, true)),     # (ā↑,ā↓)=ud
			  ((2, false, 1, false), (2, false, 1, false))) # (a↓,a↑)=du
	for b1 in (:τ, :+, :-), j1 in _mjran(b1, Nτ, Nt),
		b2 in (:τ, :+, :-), j2 in _mjran(b2, Nτ, Nt)
		r, c, s = _mx_maps(b1, j1, b2, j2, Nτ, Nt)
		(r === nothing || c === nothing) && continue
		for (dst, src) in blocks
			D[ipos(b1, j1, dst[1], dst[2]), ipos(b2, j2, dst[3], dst[4])] =
				s * C[ipos(r[1], r[2], src[1], src[2]), ipos(c[1], c[2], src[3], src[4])]
		end
	end
	return Matrix(D)
end

function _vertex_shifted_IF(C::AbstractMatrix, bs::Vector{GrassmannContourIndex})
    brs = branch.(bs)
    single = all(b -> b.band == 1, bs)
    mixed = any(==(:τ), brs) && any(b -> b === :+ || b === :-, brs)
    # 2026-10-30 与访问器（contourindex.jl）逐轮廓对齐——只有实时 Keldysh
    # （单 flavor）保留顶点位移，其余轮廓回退 cell（节点值）读取：
    #   · 实时 Keldysh（单 flavor）：位移保留——ED 验证通过，误差常数好 ~25×；
    #   · τ 环：回退——顶点重组使 g(τ) 离散误差常数恶化 ~30×（0.017 vs
    #     0.0006 @δτ=0.125，同为一阶；GTEMPO Toulouse 基准
    #     performance/firstorder/vertex_shift_accuracy.jl）；
    #   · mixed：回退——顶点位移与物理 ED 矛盾（GTEMPO 全部 mixed ED 套件
    #     偏差 6–7% 且不随 δt 收敛；cell 读取全部通过；证据与复现见
    #     bugrepots/dense_mixed_vertexshift_convention.md）；
    #   · BCS τ：随访问器回退 cell 读取；BCS mixed：访问器与 Dense 均为
    #     位移版（互一致，严格 BCS 链验证待补）；BCS RT：暂沿用旧装配。
    if single && !mixed
        all(b -> b === :τ, brs) && return C          # τ 环：raw（g(τ) 常数恶化 30×）
        return _vertex_shifted_rt(C, length(bs) ÷ 4 - 1)  # RT 单 flavor：位移保留（ED ✓）
    elseif !mixed
        return C                                     # BCS RT：暂沿用旧装配
    return C
    # BCS mixed/RT：cell 读取（与 BCS 访问器一致——其 τ/混合/RT 三分支均为
    # raw 直读；此前 Dense 的 _vertex_shifted_bcs_mixed 移位未经验证，已随
    # mixed/τ 一并回退；严格 BCS 链验证后再定最终约定）
    return C                     # mixed 单 flavor / BCS mixed：raw（与访问器一致）
end
    return C                     # mixed 单 flavor / BCS mixed：raw（兜底）
end

GrassmannHybridizationMatrix(x::IFSource) =
    GrassmannHybridizationMatrix(basis(x), _vertex_shifted_IF(_if_coefficients(x), basis(x)))

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
