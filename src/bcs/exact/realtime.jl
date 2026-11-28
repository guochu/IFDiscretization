# =============================================================================
# bcs/exact/realtime.jl —— exact_bcs_Δt：BCS 浴实时 Keldysh 严格 IF 核（粒子字典）
#
# 【消费契约 = 冻结核字面放置语义（= GTEMPO AnomalousPairing 四键消费）】
#   两味 d↑/d↓ 皆物理粒子：ρ₀ 真空、两带门相位 e^{−iε_d·t}；
#   uu → (ā↑, a↑)、dd → (ā↓, a↓)、ud → (ā↑, ā↓)（对产生）、du → (a↓, a↑)（对湮灭）；
#   跨支 Keldysh 负号由消费侧 blockification 期收编（与费米/冻结核同构）。
#   下游零特例：PartialIF 装配走冻结核的同一 `_fullfactor` 路由。
#
# 构造（全 Gaussian，闭式——对 Fock 精确提取机器精度，金标准「闭式 cell / 结构化↔稠密」）：
#   ① cell：物理变量的相干态重叠 ⟨−γ̄'|e^{−iδt·H_cell}|γ⟩ = c₀·exp(Q)，
#      Q = γ̄'uγ + ½γ̄'vγ̄' + ½γwγ（闭式见 exact/formb.jl `_formb_gauss_cell`）；
#      H_cell = 星形耦合（杂质槽 ε=0）+ 浴 Σ_k[ε_k(n_P+n_Q) + Δ(P†Q† + QP)]。
#      ⚠️ 实时轴上该重叠是 Gaussian（虚时 e^{−βH} 的不是）。
#   ② 热边：配对浴 Gibbs 态的 double overlap ⟨−κ̄|e^{−βH_B}|κ⟩ **是** Gaussian，
#      逐对取其二次部分置于 (前向首位置 outs × 后向首位置 ins)，无需辅助模式。
#   ③ 浴块沿轮廓 = 块三对角（轮廓位置块：outs 2K + ins 2K）+ 热边角块：
#      `_formb_chain_schur`（逐块部分主元 LU + Woodbury，O(N·K³)）。
#   ④ 接口 Schur 消浴：G = antisym(E + C·B⁻¹·Cᵀ)，核 = −antisym(G⁻¹) 逐块读出
#      （conj ↔ eta、plain ↔ bar——与费米 FormB 核同一套读出规则）。
#
# 契约三件套（族标准 = fermions/exact/real.jl exact_fermionic_Δt docstring）：
#   ① 纯耦合项 ✓（H_cell 无 ε_d；μ≠0 拒绝）；
#   ② 严格性：冻结核字面放置消费的轮廓收缩 == 同调度直接 Trotter 离散链的迹
#      （ED 端到端证书 = 金标准 bcs_realtime.jl：Z≡1、n̄↑、n̄↓、⟨d↓d↑⟩、
#      ⟨d†↑d†↓⟩ 对 Fock ED，一般谱 × 实/复 Δ × orders 1/2）；
#   ③ 归一：Z = 轮廓收缩 × normdelta，normdelta = ν·Pf(G)，
#      ν = |c₀|^{2N}·a_edge·|Pf(B)|/Z_B（= 离散链浴配分函数 / 精确浴配分函数；
#      a_edge = Π_k⟨00|e^{−βH_k}|00⟩，Z_B = Π_k Tr e^{−βH_k}）。
#      Δ=0 锚：uu == dd == 费米核（谱 ε）逐位、ud = du ≡ 0、ν = ν_ferm²；
#      V→0 极限：核平凡、ν → 1。
#
# 历史：2026-11-25 的 Nambu 翻倍链版本（↓ 味 = 空穴模式 X，只可按 X 契约消费）
# 经用户裁决于 2026-11-26 退役——下游需要冻结核式直接消费；旧实现 @ git 82a2368。
# =============================================================================

"BCS 接口槽（c = 1..N × 支 × 味(1=↑,2=↓) × (out, in)，1-based 局部索引）：
前向槽 out = 级 c 的 bra、in = 级 c−1 的 ket；后向槽对调层级。"
_ifslot2(c::Int, br::Symbol, fl::Int, io::Symbol) =
    8 * (c - 1) + (br === :+ ? 1 : 5) + (fl === 1 ? 0 : 2) + (io === :out ? 0 : 1)

"谱离散（与费米严格核同源）：K 个离散浴模 (εs, Vs)——连续谱 GL-M 求积节点、
离散谱物理模式直取、死模剔除。μ≠0 拒绝（BCS 只支持 μ=0）。"
function _bcs_modes(f::AbstractBoundedFunction; μ::Real=0, gfsolver::ToulouseSolver=EDToulouseSolver(64))
    iszero(μ) || throw(ArgumentError("BCS 浴只支持 μ=0（μ = $μ）"))
    Es, Vsq, wΓ = _quad_nodes(f, μ, _nbath(gfsolver))
    keep = wΓ .> 0
    εs = Es[keep]
    Vs = Vsq[keep]
    K = length(εs)
    K >= 1 || throw(ArgumentError("谱在支撑上的积分为零（或非正）"))
    return εs, Vs
end

"cell 二次型与真空振幅：cell 模式（交错）1 = d↑、2 = d↓、2k+1 = P_k、2k+2 = Q_k。
δt < 0 即后向 cell ⟨−γ̄'|U†|γ⟩。|c₀|² → 0 = 相干态图表奇异（孤立 δt），拒绝。"
function _bcs_cell(εs::AbstractVector, Vs::AbstractVector, Δ::Number, δt::Real)
    K = length(εs)
    M = 2K + 2
    h = zeros(ComplexF64, M, M)
    D = zeros(ComplexF64, M, M)
    for k in 1:K
        h[1, 2k+1] = h[2k+1, 1] = Vs[k]                 # d↑ ↔ P_k
        h[2, 2k+2] = h[2k+2, 2] = Vs[k]                 # d↓ ↔ Q_k
        h[2k+1, 2k+1] = h[2k+2, 2k+2] = εs[k]
        D[2k+1, 2k+2] = Δ                               # 配对 P_k†Q_k†（D 反对称）
        D[2k+2, 2k+1] = -Δ
    end
    u, v, w, c0sq = try
        _formb_gauss_cell(h, D, δt)
    catch e
        e isa SingularException || rethrow()
        throw(ArgumentError("cell 相干态图表奇异（Bogoliubov 块不可逆）：δt = $δt 落在真空振幅零点附近，请换 N/δt"))
    end
    (isfinite(c0sq) && c0sq > 1e-13) || throw(ArgumentError(
        "cell 真空振幅 |c₀|² = $c0sq 过小（相干态图表奇异）：请换 N/δt"))
    return u, v, w, c0sq
end

"单对 (P,Q) 的 Gibbs 指数 e^{−βH_k}（基 |00⟩,|P⟩,|Q⟩,|11⟩；
H_k = ε(n_P+n_Q) + ΔP†Q† + Δ̄QP，偶宇称块 = ε·I + T，T = [[−ε, Δ̄],[Δ, ε]]，T² = E²·I，
E = √(ε² + |Δ|²)）。谱投影子形式（数值稳定，见函数体注释）。"
function _bcs_pair_gibbs(ε::Real, Δ::Number, β::Real)
    E = sqrt(ε^2 + abs(Δ)^2)
    x = β * E
    # 谱投影子形式：e^{−β·T} = e^{−x}·P₊ + e^{x}·P₋，P± = (I ± T/E)/2——系数恒正，
    # 负能量模（ε < 0）下 cosh + (ε/E)·sinh 形式的灾难性抵消在此消失
    T = E == 0 ? zeros(ComplexF64, 2, 2) : ComplexF64[-ε conj(Δ); Δ ε] ./ E
    Pp = (Matrix{ComplexF64}(I, 2, 2) .+ T) ./ 2
    Pm = (Matrix{ComplexF64}(I, 2, 2) .- T) ./ 2
    ev = exp(-β * ε) .* (exp(-x) .* Pp .+ exp(x) .* Pm)
    g = zeros(ComplexF64, 4, 4)
    g[1, 1] = ev[1, 1]; g[1, 4] = ev[1, 2]; g[4, 1] = ev[2, 1]; g[4, 4] = ev[2, 2]
    g[2, 2] = g[3, 3] = exp(-β * ε)
    return g
end

"""
    exact_bcs_Δt(f; β, N, t, Δ=0, μ=0, gfsolver=EDToulouseSolver(64))
        -> (BCSCorrelationFunction{ExactRealCorrelationFunction}, normdelta)

BCS（超导）浴严格 IF 核（实时 Keldysh 轮廓）——**纯耦合项严格核**（exact 核族统一契约，
见 exact_fermionic_Δt docstring；构造见文件头注）。

# 契约
1. **核 = 纯耦合项**：核内只含 BCS 浴（离散模 (ε_k, V_k) 的配对二次型
   Σ_k[ε_k(n_P+n_Q) + Δ(P†Q† + QP)]）+ 杂质-浴耦合（谱 `f` 经 `gfsolver` 离散；
   杂质双味 d↑/d↓ 各以其味耦合浴，杂质槽不配对）与热力学 β——不含任何杂质
   哈密顿量信息（无 ε_d、无劈分阶数、无 kick 相位）。μ≠0 抛 `ArgumentError`。
2. **消费契约 = 冻结核字面放置语义**（= GTEMPO AnomalousPairing 四键消费）。
   两味皆物理粒子（ρ₀ 真空、两带门相位 e^{−iε_d·t}）：

   | 容器字段 | 消费腿对 | 物理含义 |
   |---|---|---|
   | `uu` | (ā↑, a↑) | 正常（↑ 味） |
   | `dd` | (ā↓, a↓) | 正常（↓ 味） |
   | `ud` | (ā↑, ā↓) | 对产生 d†↑d†↓ |
   | `du` | (a↓, a↑) | 对湮灭 d↓d↑ |

   跨支 Keldysh 负号由消费侧 blockification 期收编。此契约下的轮廓收缩 = 同调度
   **直接 Trotter 离散链的迹，机器精确**（消费参考实现 = 包内金标准
   `test/exact/_worldline.jl` 的 `bcs_worldline_particle`；ED 端到端证书 =
   `test/exact/bcs_realtime.jl`）。
3. **归一契约**：`Z = 轮廓收缩 × normdelta`（比值观测量严格相消）；
   `normdelta = ν·Pf(G)`，ν = |c₀|^{2N}·a_edge·|Pf(B)|/Z_B
   （文件头注；离散链浴配分函数 / 精确浴配分函数）。Δ=0 锚：`uu` == `dd` ==
   费米核（谱 ε）逐位、`ud = du ≡ 0`、ν = ν_ferm²。
4. **裸核契约（2026-11-28）**：与费米严格核同款（exact_fermionic_Δt docstring
   契约 4）——Schur 装配吸收的 −1 接口分辨率单位（幂零像 `exp(−x̄x)`，逐
   支撑点 2..N+1 逐支逐**味**）构造期就地扣除，只落正常通道 `uu`/`dd` 的
   同支等时对角（反常键 (ā↑,ā↓)/(a↓,a↑) 无 plain-pair 位、无吸收）；消费者
   **不得**再乘接口分辨率反测度（compact 标定 `counter_c = +1` 已随裸核
   契约退役）。

# 构造（粒子字典 FormB 链，文件头注）
接口槽 8N（每 cell 每支 4 槽 × 味 × (out,in)）+ 浴块（2N+1 个轮廓位置 × 4K 腿）
经 cell Gaussian 重叠缝合、热边（Gibbs double overlap 二次部分）进浴块；
结构化 Schur 消浴块 `G = antisym(E + C·B⁻¹·Cᵀ)`，读矩阵 `−antisym(G⁻¹)`
逐块读出四块。腿基与费米核同构：容器点 1（ρ₀ 端）置零，核支撑 = 点 2..N+1
（点 p+1 ↔ 接口槽 p = cell p）；消费者的开放端恒等窗（节点 N+2）不携带核。
离散谱（DiracDelta/DiscreteSpectrum）物理模式直取；连续谱 GL-M 求积节点离散。
复 Δ 允许。

# Δ=0 退化（金标准 T1/T2）
Δ=0 时 BCS 浴 = 两个**独立**费米浴（P_k、Q_k 同在 +ε_k）：`uu` == `dd` ==
费米核（谱 ε）逐位，`ud = du ≡ 0`，normdelta = 费米 normdelta²。
"""
function exact_bcs_Δt(f::AbstractBoundedFunction; β::Real, N::Int, t::Real,
                      Δ::Number=0, μ::Real=0,
                      gfsolver::ToulouseSolver=EDToulouseSolver(64))
    N >= 1 || throw(ArgumentError("N ≥ 1（got N = $N）"))
    εs, Vs = _bcs_modes(f; μ=μ, gfsolver=gfsolver)
    K = length(εs)
    M = 2K + 2
    nb = 2K
    nb4 = 4K
    δt = Float64(t) / N
    u, v, w, c0sq = _bcs_cell(εs, Vs, Δ, δt)
    ub, vb, wb, _ = _bcs_cell(εs, Vs, Δ, -δt)           # 后向 cell = ⟨−γ̄'|U†|γ⟩
    # 轮廓位置 q = 1..Q（Q = 2N+1）：1..N = 前向级 0..N−1、N+1 = 迹级 N、
    # N+2..2N+1 = 后向级 N−1..0；块内腿序：1..2K = outs、2K+1..4K = ins
    Q = 2N + 1
    nif = 8N
    D = [zeros(ComplexF64, nb4, nb4) for _ in 1:Q]      # 位置对角块
    ℓ = [zeros(ComplexF64, nb, nb) for _ in 1:Q]        # 下块 (out,in) 象限（q ≥ 2）
    E = zeros(ComplexF64, nif, nif)                     # 接口块
    C = zeros(ComplexF64, nif, nb4 * Q)                 # 接口 × 浴耦合
    for q in 1:Q, m in 1:nb
        _formb_put!(D[q], m, nb + m, -1.0)              # Berezin 测度（out_m, in_m）
    end
    bleg(q, io, m) = (q - 1) * nb4 + (io === :out ? m : nb + m)
    "一个 cell 的条目放入 E / C / D / ℓ：qo = out 位置、qi = in 位置、iout/iin = 杂质
    out/in 腿号、flip = 迹端约定（含迹对浴 out 变量的条目取负，仅前向末 cell）"
    function place!(uu, vv, ww, qo, qi, iout, iin, flip)
        for a in 3:M, b in 3:M                         # (out 浴, in 浴)
            q = uu[a, b]
            abs(q) < 1e-18 && continue
            ℓ[qo][a - 2, b - 2] += flip ? -q : q
        end
        for a in 1:2, b in 3:M                         # (out 杂质, in 浴)
            q = uu[a, b]
            abs(q) < 1e-18 && continue
            C[iout(a), bleg(qi, :in, b - 2)] += q
        end
        for a in 3:M, b in 1:2                         # (out 浴, in 杂质)：A[out,in] 的转置位置
            q = uu[a, b]
            abs(q) < 1e-18 && continue
            C[iin(b), bleg(qo, :out, a - 2)] += flip ? q : -q
        end
        for a in 1:M, b in a+1:M                       # v：(out a, out b)；w：(in a, in b)
            oa = a ≥ 3
            ob = b ≥ 3
            q = vv[a, b]
            if abs(q) > 1e-18
                s = (flip && xor(oa, ob)) ? -q : q
                if oa && ob
                    _formb_put!(D[qo], a - 2, b - 2, s)
                elseif !oa && ob
                    C[iout(a), bleg(qo, :out, b - 2)] += s
                else
                    _formb_put!(E, iout(a), iout(b), s)
                end
            end
            q = ww[a, b]
            if abs(q) > 1e-18
                if oa && ob
                    _formb_put!(D[qi], nb + a - 2, nb + b - 2, q)
                elseif !oa && ob
                    C[iin(a), bleg(qi, :in, b - 2)] += q
                else
                    _formb_put!(E, iin(a), iin(b), q)
                end
            end
        end
        for a in 1:2, b in 1:2                         # (out 杂质, in 杂质)
            q = uu[a, b]
            abs(q) < 1e-18 && continue
            _formb_put!(E, iout(a), iin(b), q)
        end
    end
    for c in 1:N
        place!(u, v, w, c == N ? N + 1 : c + 1, c,
               b -> _ifslot2(c, :+, b, :out), b -> _ifslot2(c, :+, b, :in), c == N)
        place!(ub, vb, wb, 2N + 2 - c, c == N ? N + 1 : 2N + 1 - c,
               b -> _ifslot2(c, :-, b, :out), b -> _ifslot2(c, :-, b, :in), false)
    end
    # 热边：逐对 Gibbs double overlap 的二次部分（FormB 约定 ⟨−κ̄|e^{−βH_k}|κ⟩）：
    #   正常项 (κ̄, κ) → 角块 R（首位置 outs × 末位置 ins）；(κ̄_P,κ̄_Q) → 首位置 out-out；
    #   (κ_P,κ_Q) → 末位置 in-in。浴模式号：P_k ↔ 2k−1、Q_k ↔ 2k
    R = zeros(ComplexF64, nb, nb)
    aedge = 1.0
    ZB = 1.0
    for k in 1:K
        g = _bcs_pair_gibbs(εs[k], Δ, β)
        a = real(g[1, 1])
        P, Qm = 2k - 1, 2k
        R[P, P] += g[2, 2] / a
        R[Qm, Qm] += g[3, 3] / a
        _formb_put!(D[1], P, Qm, g[4, 1] / a)
        _formb_put!(D[Q], nb + P, nb + Qm, -g[1, 4] / a)
        aedge *= a
        ZB *= a + 2 * g[2, 2] + g[4, 4]
    end
    X, ldetB = _formb_chain_schur(D, ℓ, R, collect(1:nb),
                                  collect((Q - 1) * nb4 + nb + 1:Q * nb4), Matrix(transpose(C)))
    eff = E + C * X
    G = (eff .- transpose(eff)) ./ 2
    Ginv = inv(G)
    GA = (Ginv .- transpose(Ginv)) ./ 2
    # ν：离散链浴配分函数 / 精确浴配分函数（见文件头注）
    ν = c0sq^N * aedge * exp(ldetB / 2) / ZB
    sgn = -1                                            # FormB elim_sign：exp(−½jᵀG⁻¹j) 的整体负号
    # 四块读出（conj ↔ eta、plain ↔ bar；点 0（容器点 1，ρ₀ 端）置零）
    function blk(grow::Function, gcol::Function)
        pt(i, b1, j, b2) = (i == 0 || j == 0) ? zero(ComplexF64) :
                            sgn * GA[grow(i, b1), gcol(j, b2)]
        return ExactRealCorrelationFunction(
            [pt(i, :+, j, :+) for i in 0:N, j in 0:N],
            [pt(i, :+, j, :-) for i in 0:N, j in 0:N],
            [pt(i, :-, j, :+) for i in 0:N, j in 0:N],
            [pt(i, :-, j, :-) for i in 0:N, j in 0:N])
    end
    out(c, br, fl) = _ifslot2(c, br, fl, :out)
    inn(c, br, fl) = _ifslot2(c, br, fl, :in)
    uu = blk((i, b) -> inn(i, b, 1), (j, b) -> out(j, b, 1))   # ā↑Δa↑ = −GA[↑eta_i, ↑bar_j]
    dd = blk((i, b) -> inn(i, b, 2), (j, b) -> out(j, b, 2))   # ā↓Δa↓
    ud = blk((i, b) -> inn(i, b, 1), (j, b) -> inn(j, b, 2))   # ā↑Δā↓（对产生）
    du = blk((i, b) -> out(i, b, 2), (j, b) -> out(j, b, 1))   # a↓Δa↑（对湮灭）
    # 裸核契约：Schur 装配吸收的 −1 接口分辨率单位（幂零像 exp(−x̄x)，逐支撑
    # 点逐支逐味；见 docstring 契约 4）构造期就地扣除——正常通道 uu/dd 的同支
    # 等时对角 +1（容器点 2..N+1；ud/du 反常键无 plain-pair 位、无吸收）
    barefix(x::ExactRealCorrelationFunction) = begin
        pp = Matrix(x.G₊₊)
        mm = Matrix(x.G₋₋)
        for m in (pp, mm), p in 2:size(pp, 1)
            m[p, p] += 1.0
        end
        ExactRealCorrelationFunction(pp, x.G₊₋, x.G₋₊, mm)
    end
    return BCSCorrelationFunction(uu=barefix(uu), ud=ud, du=du, dd=barefix(dd)),
           ν * _formb_pfaffian(G)
end
