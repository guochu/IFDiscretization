# =============================================================================
# exact/formb.jl —— FormB（形态 B）消元原语（实轴严格核族共享，内部符号）
#
# 费米（fermions/exact/real.jl）与 BCS（bcs/exact/realtime.jl）两处的共享底层：
# 反对称占位 / Q kernel 缝合 / 数值 Pfaffian / 谱离散。原语维度无关——BCS 的
# Nambu 翻倍装配消费同一套（2026-11-25 自 real.jl 抽出，纯移动；多接口槽
# qadd 变体为 BCS 翻倍缝合新增）。契约与金标准：
#   - 费米实时：test/exact/fermionic_realtime.jl
#   - BCS 实时：test/exact/bcs_realtime.jl
# =============================================================================

# ---------------- FormB（形态 B）消元原语（无 kick 严格核；← GTEMPO
# extensions/exactif compact_delta 的 GM 换代移植，2026-11-22 契约换代） ----------------

"反对称占位（1-based）：上三角 += v、下三角 −= v（i≠j）。"
_formb_put!(A::Matrix{ComplexF64}, i::Int, j::Int, value) = begin
    i == j && return A
    if i < j
        A[i, j] += value
        A[j, i] -= value
    else
        A[j, i] -= value
        A[i, j] += value
    end
    A
end

"接口槽（c = 1..N × 支 × (bar, eta)，G 矩阵 1-based 局部索引）。"
_formb_ifslot(c::Int, br::Symbol, stat::Symbol) =
    4 * (c - 1) + (br === :+ ? 1 : 3) + (stat === :bar ? 0 : 1)

_formb_partner(c::Int, br::Symbol, stat::Symbol) =
    _formb_ifslot(c, br, stat === :bar ? :eta : :bar)

"""Q kernel 缝合：把 (bar_imp, eta_imp) IF 对与 bath 输出 bars/输入 etas 经 U
缝合（反对称占位）；trace_bra 翻转输出行（⟨−ψ̄| 迹端约定）。U 行/列序 =
[杂质槽; 浴槽]（杂质 = 首行首列）。"""
function _formb_qadd!(A::Matrix{ComplexF64}, bar_imp::Int, eta_imp::Int,
                      bath_bars::Vector{Int}, bath_etas::Vector{Int}, U::AbstractMatrix;
                      trace_bra::Bool=false)
    m = length(bath_bars)
    size(U) == (m + 1, m + 1) || throw(DimensionMismatch("Q 传播子维度不匹配"))
    _formb_put!(A, bar_imp, eta_imp, U[1, 1])
    for j in eachindex(bath_etas)
        _formb_put!(A, bar_imp, bath_etas[j], U[1, j + 1])
    end
    for i in eachindex(bath_bars)
        value = U[i + 1, 1]
        trace_bra && (value = -value)
        _formb_put!(A, bath_bars[i], eta_imp, value)
        for j in eachindex(bath_etas)
            bath_value = U[i + 1, j + 1]
            trace_bra && (bath_value = -bath_value)
            _formb_put!(A, bath_bars[i], bath_etas[j], bath_value)
        end
    end
    return A
end

"""Q kernel 缝合（多接口槽推广；BCS Nambu 翻倍用）：imp_out/imp_in = 接口槽
向量（U 前段行/列，各 ni 个），bath_out/bath_in = 浴链槽（后段 m 个）。
U[a, b] →（反对称占位）(out_a, in_b)；trace_bra 只翻 bath 输出行（⟨−ψ̄| 迹端
约定，接口行不翻——与单槽版同口径）。"""
function _formb_qadd_ni!(A::Matrix{ComplexF64}, imp_out::Vector{Int}, imp_in::Vector{Int},
                         bath_out::Vector{Int}, bath_in::Vector{Int}, U::AbstractMatrix;
                         trace_bra::Bool=false)
    ni = length(imp_out)
    length(imp_in) == ni || throw(DimensionMismatch("接口 out/in 槽数不等"))
    m = length(bath_out)
    length(bath_in) == m || throw(DimensionMismatch("浴 out/in 槽数不等"))
    size(U) == (ni + m, ni + m) || throw(DimensionMismatch("Q 传播子维度不匹配"))
    for a in 1:ni, b in 1:ni
        _formb_put!(A, imp_out[a], imp_in[b], U[a, b])
    end
    for a in 1:ni, j in 1:m
        _formb_put!(A, imp_out[a], bath_in[j], U[a, ni + j])
    end
    for i in 1:m, b in 1:ni
        value = U[ni + i, b]
        trace_bra && (value = -value)
        _formb_put!(A, bath_out[i], imp_in[b], value)
    end
    for i in 1:m, j in 1:m
        value = U[ni + i, ni + j]
        trace_bra && (value = -value)
        _formb_put!(A, bath_out[i], bath_in[j], value)
    end
    return A
end

"数值 Pfaffian（Parlett–Reid，O(n³)）。"
function _formb_pfaffian(A0::Matrix{ComplexF64})
    n = size(A0, 1)
    (n % 2 == 0) || return zero(ComplexF64)
    A = copy(A0)
    pf = one(ComplexF64)
    for k in 1:2:n
        kp = k + 1
        kp > n && break
        if abs(A[kp, k]) < 1e-14
            r = 0
            for i in (kp + 1):n
                abs(A[i, k]) > 1e-14 && (r = i; break)
            end
            r == 0 && return zero(ComplexF64)
            A[[kp, r], :] = A[[r, kp], :]
            A[:, [kp, r]] = A[:, [r, kp]]
        end
        piv = A[kp, k]
        for i in (kp + 1):n
            c = A[i, k] / piv
            c == 0 && continue
            for j in (kp + 1):n
                A[i, j] -= c * A[kp, j]
                A[j, i] -= c * A[j, kp]
            end
        end
        pf *= piv
    end
    return pf
end

# ------------------------------ 谱离散（原 eigensolve.jl；GL-M 求积节点，
# 浴 h₀ 按需内联构造；← QuAPI/exactif/test/exactif_api.jl 原样移植） -------

"GL-M 求积节点上的谱数据（求积节点，非物理模式）。μ = 谱平移。"
function _quad_nodes(f::AbstractBoundedFunction, μ::Real, M::Int)
    lb, ub = lowerbound(f), upperbound(f)
    xs, ws = gauss_legendre(M)
    half = (ub - lb) / 2
    Es = [(lb + ub) / 2 + half * x - μ for x in xs]
    wΓ = [half * w * f((lb + ub) / 2 + half * x) for (x, w) in zip(xs, ws)]
    Vsq = sqrt.(max.(wΓ, 0.0))
    return Es, Vsq, wΓ
end

# 离散谱 = 物理模式直取（严格核的浴侧是二次型——离散浴下核天然精确，无求积
# 误差；模式数 = 谱线数，gfsolver 的 nbath 不消费，EDToulouseSolver docstring
# 「离散谱 = 谱线数目」的实装。此前缺失：DiracDelta 走 GL 路径在 f(ϵ) 求值处
# 抛「can not call to a delta function」、DiscreteSpectrum 的 ±Inf 撑界静默产
# 出 Inf/NaN——两者都不该被静默拒收/污染，2026-11-21 补齐）。
# 死模过滤契约沿用：wΓ ≤ 0 的线剔除（α = 0 线 = 死模，K ≥ 1 守卫不变）。
_quad_nodes(f::DiracDelta, μ::Real, ::Int) =
    ([f.ω - μ], [sqrt(max(f.α, 0.0))], [Float64(f.α)])

_quad_nodes(f::DiscreteSpectrum, μ::Real, ::Int) =
    (collect(Float64.(frequencies(f))) .- μ,
     sqrt.(max.(spectrumvalues(f), 0.0)),
     collect(Float64.(spectrumvalues(f))))
