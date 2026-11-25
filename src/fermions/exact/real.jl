# =============================================================================
# exact/real.jl —— exact_fermionic_Δt：实时 Keldysh 严格 IF 核（FormB 消元）
# 契约与构造见 exact_fermionic_Δt docstring；带踢算符忠实核不在本文件
#（其构造保存于 test/exact/_kicks_helper.jl，金标准测试专用）。
# FormB 消元原语（_formb_put!/_formb_qadd!/_formb_pfaffian/_quad_nodes）住
# src/exact/formb.jl（2026-11-25 抽出，与 BCS 翻倍核共享）。
# =============================================================================
# ------------------------------ 实时 ------------------------------

"腿位（节点 j0 = 0..N；+ 支在前，支内 (a, ā) 相邻），1-based"
_exrt_legpos(j0::Int, br::Symbol, ā::Bool) = 4 * j0 + (br === :+ ? 0 : 2) + (ā ? 2 : 1)

"""
    exact_fermionic_Δt(f; β, N, t, μ=0, gfsolver=EDToulouseSolver(64))
        -> (ExactRealCorrelationFunction, normdelta)

严格 IF 核（实时 Keldysh 轮廓）——**纯耦合项严格核**
（exact 核族的统一契约；Δt/Δτ/Δm 与 BCS 同族按此对齐）：

# 契约
1. **核 = 纯耦合项**：核内只含浴 + 杂质-浴耦合 h₀（谱 `f` 经 `gfsolver`
   离散的星形模）与热力学 β——**不含任何杂质哈密顿量信息**（无 ε_d、无
   劈分阶数、无 kick 相位）。IF 装配对一般杂质模型无假设（纯浴对象）。
2. **严格性语义**：消费侧（杂质演化**全部**走 K 链门 + 接口反测度
   `counter_c = +1` + Keldysh 折返 + ρ₀）的轮廓收缩 = 同调度**直接
   Trotter 离散链的迹，任意 ε_d 机器精确**（消费参考实现 =
   GTEMPO `exactif_pipeline`；机器锚 = GTEMPO test/exactif 阶梯）。
3. **归一契约**：`Z = 轮廓收缩 × normdelta`（比值观测量严格相消）；
   `normdelta = ν·Pf(G)`，ν = 浴链配分比 `det(I+BRB†)/det(I+R)`。

# 构造（形态 B / FormB 消元）
接口槽（c = 1..N × 支 × (bar, eta)）+ bath 配对（trace_final + forward(N) +
backward(N)，每对 (γ̄,γ) 测度 −1）的总 Gaussian 经耦合演化
W = e^{−i·cell·h} 缝合、热边 e^{−βε} 进 bath 块；Schur 消 bath 块
`effective = E + C·(B\\Cᵀ)`，反对称化 `G = antisym(effective)`，读矩阵
`Δ = −antisym(G⁻¹)`。

# 腿基与谱
腿基 = 轮廓点 1..N+1 × ± × (a, ā)（与带踢路径同 legpos 布局）；点 1
（t = 0 / ρ₀ 端）与点 N+1（开放端恒等窗）置零，Δ 支撑 = 接口点 2..N。
离散谱（DiracDelta/DiscreteSpectrum）物理模式直取——核对该离散链机器
精确；连续谱 gfsolver 求积节点离散（GL-M，收敛到连续极限）。

# 与带踢核的边界
带踢「算符忠实」核（核内携带杂质踢相位 x_k = e^{−iαε_d}，Z ≡ 1、观测量 =
门序列的精确收缩）**不属于本契约**——它违反纯耦合项要求（核依赖杂质
哈密顿量）。其构造保存于测试 helper（test/exact/_kicks_helper.jl），仅作
底层 W/Schur 数学的 Z≡1 锚。历史注：本函数 2026-11-22 前的无 kick 语义 =
顶点移位离散 Γ(W+(x−1)P₁)^N（ε_d ≠ 0 时 Z 偏离 1 O(δt)），已废弃。
"""
function exact_fermionic_Δt(f::AbstractBoundedFunction; β::Real, N::Int, t::Real,
                            μ::Real=0, gfsolver::ToulouseSolver=EDToulouseSolver(64))
    M = _nbath(gfsolver)
    Es, Vsq, wΓ = _quad_nodes(f, μ, M)
    keep = wΓ .> 0                      # 剔除谱为零区域的节点（死模不改配分）
    εs = Es[keep]
    Vs = Vsq[keep]
    K = length(εs)
    K >= 1 || throw(ArgumentError("谱在支撑上的积分为零（或非正）"))
    δt = Float64(t) / N
    cells = fill(δt, N)
    nif = 4 * N
    total = nif + 2K * (2N + 1)
    A = zeros(ComplexF64, total, total)
    h = zeros(ComplexF64, K + 1, K + 1)
    for k in 1:K
        h[1, k+1] = conj(Vs[k])
        h[k+1, 1] = Vs[k]
        h[k+1, k+1] = εs[k]
    end
    Us = [exp(-1im * cell .* h) for cell in cells]
    cursor = nif
    pair_data = NamedTuple[]
    for label in (:trace_final, :forward, :backward)
        count = label === :trace_final ? 1 : N
        for step in 0:(count - 1)
            bars = collect((cursor + 1):(cursor + K))
            etas = collect((cursor + K + 1):(cursor + 2K))
            push!(pair_data, (bars = bars, etas = etas))
            cursor += 2K
        end
    end
    trace_pair = pair_data[1]
    forward_pairs = pair_data[2:(N + 1)]
    backward_pairs = pair_data[(N + 2):end]
    for step in 0:(N - 1)
        input_pair = forward_pairs[step + 1]
        output_pair = step == N - 1 ? trace_pair : forward_pairs[step + 2]
        _formb_qadd!(A, _formb_ifslot(step + 1, :+, :bar), _formb_ifslot(step + 1, :+, :eta),
                     output_pair.bars, input_pair.etas, Us[step + 1];
                     trace_bra = (step == N - 1))
    end
    for step in 0:(N - 1)
        output_pair = backward_pairs[step + 1]
        input_pair = step == N - 1 ? trace_pair : backward_pairs[step + 2]
        _formb_qadd!(A, _formb_ifslot(step + 1, :-, :bar), _formb_ifslot(step + 1, :-, :eta),
                     output_pair.bars, input_pair.etas, adjoint(Us[step + 1]))
    end
    thermal = Diagonal(exp.(Float64(β) .* (.-εs)))
    for i in 1:K, j in 1:K
        _formb_put!(A, forward_pairs[1].bars[i], backward_pairs[1].etas[j], thermal[i, j])
    end
    for pair in pair_data, mode in 1:K
        _formb_put!(A, pair.bars[mode], pair.etas[mode], -1.0)
    end
    E = A[1:nif, 1:nif]
    C = A[1:nif, nif+1:end]
    B = A[nif+1:end, nif+1:end]
    effective = E + C * (B \ transpose(C))
    G = 0.5 .* (effective .- transpose(effective))
    # ν：浴链配分比（det 形式 O(K³)；B 的链乘序 = cells 升序左乘，与认证参照一致）
    Bprod = Matrix{ComplexF64}(I, K, K)
    for c in 1:N
        Bprod = Us[c][2:end, 2:end] * Bprod
    end
    R = Diagonal(exp.(Float64(β) .* (.-εs)))
    ν = det(I + Bprod * R * Bprod') / det(I + R)
    Ginv = inv(G)
    GA = (Ginv .- transpose(Ginv)) ./ 2
    sgn = -1                            # FormB elim_sign：exp(−½jᵀG⁻¹j) 的整体负号
    L = 4 * (N + 1)
    Δ = zeros(ComplexF64, L, L)
    for c in 1:N, br in (:+, :-), c2 in 1:N, br2 in (:+, :-)
        v = sgn * GA[_formb_partner(c, br, :bar), _formb_partner(c2, br2, :eta)]
        iszero(v) && continue
        Δ[_exrt_legpos(c, br, true), _exrt_legpos(c2, br2, false)] = v
    end
    basis = [GrassmannContourIndex(j0 + 1, br, 1, c)
             for j0 in 0:N for br in (:+, :-) for c in (false, true)]
    return _strict_real_from_legpos(basis, Δ), ν * _formb_pfaffian(G)
end

# 严格容器的分支块切片：腿基序（chain/legpos 同构）→ 各分支 (ā 行, a 列) 块
_strict_blocks_from_legpos(basis::Vector{GrassmannContourIndex}, Δ::Matrix{ComplexF64}) =
    (b1, b2) -> Δ[[i for (i, l) in enumerate(basis) if l.branch == b1 && l.conj],
                  [j for (j, l) in enumerate(basis) if l.branch == b2 && !l.conj]]
_strict_real_from_legpos(basis::Vector{GrassmannContourIndex}, Δ::Matrix{ComplexF64}) =
    (B = _strict_blocks_from_legpos(basis, Δ);
     ExactRealCorrelationFunction(B(:+, :+), B(:+, :-), B(:-, :+), B(:-, :-)))
