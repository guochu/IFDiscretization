# =============================================================================
# _bcs_physical_ref.jl —— 粒子字典 BCS 严格核的**独立**稠密参照（金标准测试专用 helper）
#
# 独立于 src 的两处闭式与结构化求解（exact_bcs_Δt 的对拍面）：
#   · cell 重叠 = Fock 空间矩阵指数 e^{−iδt·H_cell}（2^M 维）的相干态重叠多项式，
#     逐项提取二次型（并检验该重叠确为 Gaussian：P == c₀·exp(Q) 全单项式）；
#   · 热边 = 单对 Gibbs 指数 e^{−β·H_k}（4×4 矩阵指数）的 double overlap 二次部分；
#   · 浴块 Schur = 稠密 B\Cᵀ；normdelta = |c₀|^{2N}·a_edge·|Pf(B)|/Z_B（稠密 Pfaffian）。
#
# 模式约定（与 _bcs_ed_chain.jl 同，**交错**）：cell 变量 1 = d↑、2 = d↓、
#   2k+1 = P_k、2k+2 = Q_k；浴模式号 m = 变量 − 2（P_k ↔ 2k−1、Q_k ↔ 2k）。
# 相干态约定（FormB 测度/迹约定，|γ⟩ = Π(1 − c†γ)|0⟩，⟨−γ̄'| = ⟨0|Π(1 − γ̄'c)）：
#   ⟨−γ̄'|n⟩ = (−1)^{|n|}γ̄'_occ(n)，⟨m|γ⟩ = (−1)^{|m|+|m|(|m|−1)/2}γ_occ(m)。
#
# 铁律：参照侧不做任何谱恒等式加速；全 Fock / 全稠密；全套件单线程。
# 需先 include：_bcs_ed_chain.jl（ed_ann / ed_native_schedule / ed_bath_modes）。
# =============================================================================

# ---------------- Grassmann 多项式（bitmask，规范序 = 变量号升序） ----------------
const _GP = Dict{Int,ComplexF64}
function _gmul(A::_GP, B::_GP)
    C = _GP()
    for (ma, va) in A, (mb, vb) in B
        (ma & mb) == 0 || continue
        s = 0
        mm = ma
        while mm != 0
            i = trailing_zeros(mm) + 1
            s += count_ones(mb & ((1 << (i - 1)) - 1))
            mm &= mm - 1
        end
        m = ma | mb
        C[m] = get(C, m, 0.0im) + va * vb * (-1.0)^s
    end
    return C
end
_gadd(A::_GP, B::_GP) = (C = copy(A); for (k, v) in B; C[k] = get(C, k, 0.0im) + v; end; C)
_gscale(A::_GP, c) = _GP(k => v * c for (k, v) in A)
function _gexp(Q::_GP, maxn::Int)
    R = _GP(0 => 1.0 + 0.0im)
    term = _GP(0 => 1.0 + 0.0im)
    for n in 1:maxn
        term = _gscale(_gmul(term, Q), 1 / n)
        isempty(term) && break
        R = _gadd(R, term)
    end
    return R
end

"⟨−γ̄'|E|γ⟩ 多项式：out 变量 1..M（bra）、in 变量 M+1..2M（ket）"
function _fock_overlap(E::Matrix{ComplexF64}, M::Int)
    dim = 1 << M
    P = _GP()
    for n in 0:dim-1
        pn = count_ones(n)
        kb = 0
        for k in 1:M
            (n >> (k - 1)) & 1 == 1 && (kb |= 1 << (k - 1))
        end
        pb = pn == 0 ? _GP(0 => 1.0 + 0.0im) : _GP(kb => (-1.0)^pn)
        for m in 0:dim-1
            E[n+1, m+1] == 0 && continue
            pm = count_ones(m)
            ki = 0
            for k in 1:M
                (m >> (k - 1)) & 1 == 1 && (ki |= 1 << (k - 1))
            end
            pk = pm == 0 ? _GP(0 => 1.0 + 0.0im) :
                 _GP(ki << M => (-1.0)^(pm + pm * (pm - 1) ÷ 2))
            P = _gadd(P, _gscale(_gmul(pb, pk), E[n+1, m+1]))
        end
    end
    return P
end

"Fock cell 哈密顿量（2^M 维，模式交错：1 = d↑、2 = d↓、2k+1 = P_k、2k+2 = Q_k）"
function bcs_cell_hamiltonian(εs::AbstractVector, Vs::AbstractVector, Δ::Number)
    K = length(εs)
    M = 2K + 2
    cs = [ed_ann(M, k) for k in 1:M]
    H = zeros(ComplexF64, 1 << M, 1 << M)
    for k in 1:K
        Pk, Qk = cs[2k+1], cs[2k+2]
        H .+= Vs[k] .* (cs[1]' * Pk + Pk' * cs[1] + cs[2]' * Qk + Qk' * cs[2])
        H .+= εs[k] .* (Pk' * Pk + Qk' * Qk)
        H .+= Δ .* (Pk' * Qk') .+ conj(Δ) .* (Qk * Pk)
    end
    return H
end

"""
    bcs_cell_fock(εs, Vs, Δ, dt) -> (c0, u, v, w, resid)

Fock 精确提取的 cell 重叠：⟨−γ̄'|e^{−i·dt·H_cell}|γ⟩ = c0·exp(Q)，
Q = Σu_{ab}γ̄'_aγ_b + Σ_{a<b}v_{ab}γ̄'_aγ̄'_b + Σ_{a<b}w_{ab}γ_aγ_b（v、w 反对称）。
resid = max|P/c0 − exp(Q)|（全单项式；M > 8 时跳过，返回 −1）。
"""
function bcs_cell_fock(εs::AbstractVector, Vs::AbstractVector, Δ::Number, dt::Real)
    K = length(εs)
    M = 2K + 2
    E = exp(-1im * Float64(dt) .* bcs_cell_hamiltonian(εs, Vs, Δ))
    P = _fock_overlap(E, M)
    c0 = get(P, 0, 0.0im)
    u = zeros(ComplexF64, M, M)
    v = zeros(ComplexF64, M, M)
    w = zeros(ComplexF64, M, M)
    Q = _GP()
    for (m, val) in P
        count_ones(m) == 2 || continue
        i = trailing_zeros(m) + 1
        j = i + trailing_zeros(m >> i) + 1
        q = val / c0
        Q[m] = q
        if i ≤ M && j > M
            u[i, j-M] = q
        elseif i ≤ M
            v[i, j] = q
            v[j, i] = -q
        else
            w[i-M, j-M] = q
            w[j-M, i-M] = -q
        end
    end
    resid = -1.0
    if M ≤ 8
        Ex = _gexp(Q, M)
        resid = maximum(abs(get(P, m, 0.0im) / c0 - get(Ex, m, 0.0im)) for m in union(keys(P), keys(Ex)))
    end
    return c0, u, v, w, resid
end

"单对 Gibbs 指数 e^{−βH_k}（4×4 矩阵指数；基 |00⟩,|P⟩,|Q⟩,|11⟩）"
function bcs_pair_gibbs_fock(ε::Real, Δ::Number, β::Real)
    p, q = ed_ann(2, 1), ed_ann(2, 2)
    H = Matrix(ε .* (p' * p + q' * q) .+ Δ .* (p' * q') .+ conj(Δ) .* (q * p))
    return exp(-Float64(β) .* H)
end

"Fock Gibbs double overlap 的二次型条目：[(a, b, q, kind)]，浴模式号 P = 2k−1、Q = 2k"
function bcs_edge_fock(εs::AbstractVector, Δ::Number, β::Real)
    out = Tuple{Int,Int,ComplexF64,Symbol}[]
    aedge = 1.0
    ZB = 1.0
    for k in eachindex(εs)
        g = bcs_pair_gibbs_fock(εs[k], Δ, β)
        a = g[1, 1]
        P, Q = 2k - 1, 2k
        push!(out, (P, P, g[2, 2] / a, :normal))
        push!(out, (Q, Q, g[3, 3] / a, :normal))
        push!(out, (P, Q, g[4, 1] / a, :outout))        # ⟨11|κ⟩ = −κ_Pκ_Q ⟹ κ̄_Pκ̄_Q 系数 = g[4,1]
        push!(out, (P, Q, -g[1, 4] / a, :inin))         # ⟨−κ̄|11⟩ = +κ̄_Pκ̄_Q ⟹ −g[1,4]·κ_Pκ_Q
        aedge *= a
        ZB *= tr(g)
    end
    return out, aedge, ZB
end

"""
    bcs_chain_dense_ref(εs, Vs, Δ, β, N, dt; fock=true) -> (kern, normdelta, G, PfB)

稠密参照链（布局 = FormB：trace 对 + 前向 N 对 + 后向 N 对，每对 outs 2K + ins 2K）。
`fock = true`（缺省）：cell 与热边全部取自 Fock 矩阵指数（独立于 src 闭式）；
`fock = false`：cell 取 src 闭式 `_bcs_cell`（用于隔离结构化求解）。
核读出与 normdelta 公式同 src 契约（−antisym(G⁻¹) 逐块；ν = |c₀|^{2N}·a_edge·|Pf(B)|/Z_B）。
"""
function bcs_chain_dense_ref(εs::AbstractVector, Vs::AbstractVector, Δ::Number, β::Real, N::Int, dt::Real;
                             fock::Bool=true)
    K = length(εs)
    M = 2K + 2
    if fock
        c0f, u, v, w, = bcs_cell_fock(εs, Vs, Δ, dt)
        c0b, ub, vb, wb, = bcs_cell_fock(εs, Vs, Δ, -dt)
        c0sq = abs(c0f)^2
    else
        u, v, w, c0sq = IFD._bcs_cell(εs, Vs, Δ, dt)
        ub, vb, wb, _ = IFD._bcs_cell(εs, Vs, Δ, -dt)
    end
    nif = 8N
    npairs = 2N + 1
    A = zeros(ComplexF64, nif + 4K * npairs, nif + 4K * npairs)
    islot(c, br, b, stat) = ((c - 1) * 2 + (br === :+ ? 0 : 1)) * 4 + (b - 1) * 2 + (stat === :bar ? 1 : 2)
    ib(p, m, io) = nif + (p - 1) * 4K + (io === :out ? m : 2K + m)
    put!(a, b, q) = (abs(q) > 0 || return; A[a, b] += q; A[b, a] -= q)
    function addcell!(uu, vv, ww, fmap, flipout)
        for a in 1:M, b in 1:M
            q = uu[a, b]
            abs(q) < 1e-18 && continue
            flipout && a ≥ 3 && (q = -q)
            put!(fmap(a, :out), fmap(b, :in), q)
        end
        for a in 1:M, b in a+1:M
            q = vv[a, b]
            if abs(q) > 1e-18
                flipout && xor(a ≥ 3, b ≥ 3) && (q = -q)
                put!(fmap(a, :out), fmap(b, :out), q)
            end
            q = ww[a, b]
            abs(q) > 1e-18 && put!(fmap(a, :in), fmap(b, :in), q)
        end
    end
    for c in 1:N
        op = c == N ? 1 : 2 + c                     # 输出对：trace 或 fwd(c)
        ip = 2 + c - 1                             # 输入对：fwd(c−1)
        addcell!(u, v, w, (m, io) -> m ≤ 2 ? islot(c, :+, m, io === :out ? :bar : :eta) :
                                        (io === :out ? ib(op, m - 2, :out) : ib(ip, m - 2, :in)), c == N)
        op = 2 + N + c - 1                         # 输出对：bwd(c−1)
        ip = c == N ? 1 : 2 + N + c                # 输入对：trace 或 bwd(c)
        addcell!(ub, vb, wb, (m, io) -> m ≤ 2 ? islot(c, :-, m, io === :out ? :bar : :eta) :
                                         (io === :out ? ib(op, m - 2, :out) : ib(ip, m - 2, :in)), false)
    end
    edge, aedge, ZB = bcs_edge_fock(εs, Δ, β)
    for (a, b, q, kind) in edge
        if kind === :inin
            put!(ib(2 + N, a, :in), ib(2 + N, b, :in), q)                 # bwd(0) = 对 2+N
        elseif kind === :outout
            put!(ib(2, a, :out), ib(2, b, :out), q)                       # fwd(0) = 对 2
        else
            put!(ib(2, a, :out), ib(2 + N, b, :in), q)
        end
    end
    for p in 1:npairs, m in 1:2K
        put!(ib(p, m, :out), ib(p, m, :in), -1.0)
    end
    E = A[1:nif, 1:nif]
    C = A[1:nif, nif+1:end]
    B = A[nif+1:end, nif+1:end]
    eff = E + C * (B \ transpose(C))
    G = 0.5 .* (eff .- transpose(eff))
    Ginv = inv(G)
    GA = 0.5 .* (Ginv .- transpose(Ginv))
    function blk(grow, gcol)
        pt(i, b1, j, b2) = (i == 0 || j == 0) ? zero(ComplexF64) : -GA[grow(i, b1), gcol(j, b2)]
        return IFD.ExactRealCorrelationFunction(
            [pt(i, :+, j, :+) for i in 0:N, j in 0:N], [pt(i, :+, j, :-) for i in 0:N, j in 0:N],
            [pt(i, :-, j, :+) for i in 0:N, j in 0:N], [pt(i, :-, j, :-) for i in 0:N, j in 0:N])
    end
    out(c, br, fl) = islot(c, br, fl, :bar)
    inn(c, br, fl) = islot(c, br, fl, :eta)
    kern = IFD.BCSCorrelationFunction(
        uu=blk((i, b) -> inn(i, b, 1), (j, b) -> out(j, b, 1)),
        dd=blk((i, b) -> inn(i, b, 2), (j, b) -> out(j, b, 2)),
        ud=blk((i, b) -> inn(i, b, 1), (j, b) -> inn(j, b, 2)),
        du=blk((i, b) -> out(i, b, 2), (j, b) -> out(j, b, 1)))
    # 裸核口径（2026-11-28，与 src exact_bcs_Δt 同约定、独立实现）：Schur 的
    # −1 接口分辨率吸收（幂零像 exp(−x̄x)，逐支撑点逐支逐味）就地扣除——
    # uu/dd 同支等时对角 +1（容器点 2..N+1；ud/du 反常键无吸收）
    function barefix(x)
        pp = Matrix(x.G₊₊)
        mm = Matrix(x.G₋₋)
        for m in (pp, mm), p in 2:size(pp, 1)
            m[p, p] += 1.0
        end
        IFD.ExactRealCorrelationFunction(pp, x.G₊₋, x.G₋₊, mm)
    end
    kern = IFD.BCSCorrelationFunction(uu=barefix(kern.uu), dd=barefix(kern.dd),
                                      ud=kern.ud, du=kern.du)
    PfB = pf_std(B)
    PfG = pf_std(G)
    ν = c0sq^N * aedge * abs(PfB) / ZB
    return kern, ν * PfG, G, PfB
end
