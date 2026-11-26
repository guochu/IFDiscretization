# =============================================================================
# exact/formb.jl —— FormB（形态 B）消元原语（实轴严格核族共享，内部符号）
#
# 费米（fermions/exact/real.jl）与 BCS（bcs/exact/realtime.jl）两处的共享底层：
# 反对称占位 / Q kernel 缝合 / 数值 Pfaffian / 谱离散，以及 BCS 粒子字典链专用的
# Gaussian cell 重叠闭式（`_formb_gauss_cell`）与块三对角 + 角块 Schur 求解
# （`_formb_chain_schur`，2026-11-26）。原语维度无关。契约与金标准：
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

"""数值 Pfaffian（Parlett–Reid，O(n³)）——返回 (−1)^{n/2}·Pf（= 主元连乘 A[k+1,k]；
normdelta 与 fixture 锚的既定符号约定，勿改）。

主元 |A[k+1,k]| < 1e-14 时在同列向下找主元并**同时**交换行列；行/列对换 = 奇置换，
Pf(PᵀAP) = −Pf(A)，故每次交换结果变号。2026-11-25 修复：旧版列交换右端索引顺序写反
（`A[:, [kp, r]] = A[:, [kp, r]]` 为空操作）且不记录交换符号——一旦触发主元交换，矩阵
失去反对称性，返回值无意义（非仅差符号）。金标准 test/exact/bcs_realtime.jl「Pfaffian
主元交换回归」。"""
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
            pf = -pf
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

# --------------------- Gaussian cell 重叠闭式（BCS 粒子字典链的 cell；2026-11-26） ---------------------

"""二次哈密顿量 H = Σh_{ab}c†_ac_b + ½ΣD_{ab}c†_ac†_b + ½ΣD̄_{ab}c_bc_a（h 厄米、D 反对称）
的实时相干态重叠 ⟨−γ̄'|e^{−iδtH}|γ⟩ = c₀·exp(Q)，其中（相干态约定 = FormB 测度/迹约定：
|γ⟩ = Π(1 − c†γ)|0⟩，⟨−γ̄'| = ⟨0|Π(1 − γ̄'c)）

    Q = Σ_{ab}u_{ab}γ̄'_aγ_b + Σ_{a<b}v_{ab}γ̄'_aγ̄'_b + Σ_{a<b}w_{ab}γ_aγ_b

闭式（𝔘 = exp(+iδt·[[h, D],[−D̄, −hᵀ]]) = (UcU†, Uc†U†) 的 Bogoliubov 矩阵，(c; c†) 序）：

    u = conj(𝔘₂₂⁻¹)，v = (𝔘₂₂⁻¹𝔘₂₁)†，w = (𝔘₁₂𝔘₂₂⁻¹)†，|c₀|² = det(I + vv†)^{−1/2}

返回 `(u, v, w, c0sq)`；u/v/w 即指数的二次型系数（数守恒极限 u = e^{−iδt·h}、v = w = 0、
c₀ = 1）。⚠️ 复矩阵一律显式 `transpose`/`adjoint`（`'` 是伴随）。对 Fock 精确提取逐位
（金标准 test/exact/bcs_realtime.jl「闭式 cell」）。"""
function _formb_gauss_cell(h::Matrix{ComplexF64}, D::Matrix{ComplexF64}, δt::Real)
    M = size(h, 1)
    𝔘 = exp(1im * Float64(δt) .* [h D; -conj(D) -transpose(h)])
    U12 = 𝔘[1:M, M+1:end]
    U21 = 𝔘[M+1:end, 1:M]
    U22 = 𝔘[M+1:end, M+1:end]
    iU22 = inv(U22)
    u = conj(iU22)
    va = adjoint(iU22 * U21)
    wa = adjoint(U12 * iU22)
    v = (va .- transpose(va)) ./ 2
    w = (wa .- transpose(wa)) ./ 2
    c0sq = 1 / sqrt(real(det(I + v * adjoint(v))))
    return u, v, w, c0sq
end

# --------------------- 块三对角 + 角块 Schur 求解（BCS 粒子字典链的浴块；2026-11-26） ---------------------

"""块三对角 + 角块的 Schur 求解（浴块 B 沿轮廓 = 位置块对角 + 相邻位置耦合 + 热边角块）：

    B = T + Σ_{a,b}R[a,b]·(e_{rws[a]}e_{cls[b]}ᵀ − e_{cls[b]}e_{rws[a]}ᵀ)
    T = 块三对角：对角块 D[q]、下块 L[q] = T[q, q−1]（q = 2..Q），上块 = −L[q]ᵀ

`ℓ[q]`（nb×nb）= 下块的 (out, in) 象限（其余象限恒零——相邻位置只经 cell 的 (out,in)
条目耦合，`nb` = 块内 out/in 腿数）。返回 `(X, logdetB)`：X = B⁻¹·Ct、log|det B|。
逐块部分主元 LU（O(Q·nb³)）+ 角块 Woodbury（|rws| = |cls| = nb）。"""
function _formb_chain_schur(D::Vector{Matrix{ComplexF64}}, ℓ::Vector{Matrix{ComplexF64}},
                            R::Matrix{ComplexF64}, rws::Vector{Int}, cls::Vector{Int},
                            Ct::Matrix{ComplexF64})
    Q = length(D)
    nb = length(rws)
    length(cls) == nb || throw(DimensionMismatch("角块行列选择子长度不等"))
    nb4 = size(D[1], 1)
    nb4 == 2nb || throw(DimensionMismatch("对角块阶须为 2·nb（out/in 腿数），got $nb4 vs nb = $nb"))
    all(1 .<= rws .<= nb4) || throw(ArgumentError("角块行腿须落在首位置块内"))
    all((Q - 1) * nb4 .< cls .<= Q * nb4) || throw(ArgumentError("角块列腿须落在末位置块内"))
    ldet = 0.0
    # 块 LU：D̃_q = D_q + L_q·D̃_{q−1}⁻¹·L_qᵀ（L 只剩 (out,in) 象限 ⟹ 修正只落 (out,out) 象限）
    F = Vector{typeof(lu(D[1]))}(undef, Q)
    for q in 1:Q
        Dq = D[q]
        if q ≥ 2
            rhs = [zeros(ComplexF64, nb, nb); transpose(ℓ[q])]      # L_qᵀ 的非零部分（in 行）
            X = F[q-1] \ rhs                                          # nb4 × nb
            Dq = Dq .+ [ℓ[q] * X[nb+1:end, :] zeros(ComplexF64, nb, nb); zeros(ComplexF64, nb, nb4)]
        end
        F[q] = lu!(Dq)                                                # 就地：D[q] 之后不再使用
        ldet += logabsdet(F[q])[1]
    end
    # T⁻¹[Ct | E_rws | E_cls]（列分批，内存有界）
    m = size(Ct, 2)
    ncol = m + 2nb
    X0 = zeros(ComplexF64, nb4 * Q, ncol)
    batch = max(1, cld(4 * 2^20, nb4 * Q * 16))                       # ~4 MB / 批
    for c0 in 1:batch:ncol
        cols = c0:min(c0 + batch - 1, ncol)
        nb_c = length(cols)
        Yq = Vector{Matrix{ComplexF64}}(undef, Q)
        Zq = Vector{Matrix{ComplexF64}}(undef, Q)
        for q in 1:Q
            yq = zeros(ComplexF64, nb4, nb_c)
            for (j, col) in enumerate(cols)
                col ≤ m && (yq[:, j] .= Ct[(q - 1) * nb4 + 1:q * nb4, col])
            end
            if q == 1
                for a in 1:nb
                    j = findfirst(==(m + a), cols)
                    j === nothing || (yq[rws[a], j] = 1.0)
                end
            end
            if q == Q
                for a in 1:nb
                    j = findfirst(==(m + nb + a), cols)
                    j === nothing || (yq[cls[a] - (Q - 1) * nb4, j] = 1.0)
                end
            end
            if q ≥ 2
                yq[1:nb, :] .-= ℓ[q] * Zq[q-1][nb+1:end, :]           # L_q·z_{q−1}（只剩 out 行）
            end
            Yq[q] = yq
            Zq[q] = F[q] \ yq
        end
        for q in Q:-1:1
            rhs = Yq[q]
            if q < Q
                xout = X0[q * nb4 + 1:q * nb4 + nb, cols]             # x_{q+1} 的 out 行
                rhs[nb+1:end, :] .+= transpose(ℓ[q+1]) * xout        # L_{q+1}ᵀ·x_{q+1}（只剩 in 行）
            end
            X0[(q - 1) * nb4 + 1:q * nb4, cols] .= F[q] \ rhs
        end
    end
    # Woodbury：B = T + W·Yᵀ，W = [E_rws, E_cls]，Y = [E_cls·Rᵀ | −E_rws·R]
    W0 = X0[:, m+1:m+2nb]                                              # T⁻¹W
    Yt = zeros(ComplexF64, 2nb, nb4 * Q)
    for a in 1:nb, b in 1:nb
        Yt[a, cls[b]] += R[a, b]
        Yt[nb + a, rws[b]] -= R[b, a]
    end
    S = I + Yt * W0
    FS = lu(S)
    X = X0[:, 1:m] .- W0 * (FS \ (Yt * X0[:, 1:m]))
    return X, ldet + logabsdet(FS)[1]
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
