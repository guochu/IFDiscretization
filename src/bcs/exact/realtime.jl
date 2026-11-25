# =============================================================================
# bcs/exact/realtime.jl —— exact_bcs_Δt：BCS 浴实时 Keldysh 严格 IF 核
# （FormB 消元的 Nambu 翻倍推广）
#
# 装配全链路照搬费米 FormB（src/exact/formb.jl），翻倍字典：
#   单粒子 𝓗 = [[h₊, B],[B†, −h₊ᵀ]]（2(K+1)²；h₊ = 星形（杂质 ε=0 + V），
#   B = Δ·diag(0,1..1)——杂质槽不配对，配对属浴内容）；
#   置换序 = [杂质粒子, 杂质空穴, 浴粒子, 浴空穴]；
#   接口槽（c = 1..N × 支 × 味 × (out, in)）：↑对 = (ā↑, a↑)（粒子字典），
#   ↓对 = (a↓, ā↓)（空穴字典：out = Q 生成元、in = Q̄ 生成元）；
#   链对每对 out 2K（P̄‖Q）+ in 2K（P‖Q̄）；每对测度：两扇区同 −1
#   （Berezin RI 权重 e^{−ψ̄ψ} 逐味；空穴扇区 +1 的 Nambu 共轭表述被
#    K=1 ED 探针否决，2026-11-25 标定）；
#   热边 = e^{−β·𝓗_B}（浴-only BdG）置于 (fwd outs × bwd ins)——空穴扇区携
#   e^{+βε} 权重（Nambu 虚时传播子 e^{−τ𝓗} 的扇区结构；探针钉定：空穴块
#   e^{−β𝓗_B} 命中、e^{−βε} 副本被否，2026-11-25）。
# 读出（消费约定，bcscorr.jl 指数四项；GA = antisym(G⁻¹)，sgn = −1）：
#   (out,in) 型对（uu/ud/du）走 partner 交叉式 −GA[partner(row), partner(col)]；
#   (in,out) 型对（dd：消费指数 ā↓ 在前，bcscorr.jl 2026-11-03 装配修复同款
#   陷阱）走直读式 −GA[ā↓, a↓]——同属一个二次型系数抽取约定（跨味推广；
#   dd 于 Δ=0 机器精度命中费米核，探针钉定 2026-11-25）。
#   uu = −GA[↑in₁,↑out₂]；dd = −GA[↓in₁,↓out₂]（直读）；
#   ud = −GA[↑in₁,↓out₂]；du = −GA[↓in₁,↑out₂]。
#   row = Φ_p 侧时间、col = Φ_q† 侧，等时走 greater 侧。
# 契约三件套（族标准 = real.jl exact_fermionic_Δt docstring）：
#   ① 纯耦合项 ✓（𝓗 无 ε_d；μ≠0 拒绝，照搬退役 bcs_nambu_data 守卫）；
#   ② 严格性语义 = 消费收缩 == 同调度直接 Trotter 离散链的迹（门 =
#      exp(−iα·ε_d·(n↑+n↓))）；⚠️ 包内 ED 端到端证书（Z≡1 + n̄ 对拍）依赖
#      BCS 消费契约定义（GTEMPO 消费面立项的组成部分）——本包内已锚定：
#      Δ=0 退化逐位（uu/dd == 费米 FormB 核机器精度、ud/du ≡ 0）+ ③ 的
#      二倍关系；异常块取向的 ED 终裁随消费面立项闭环；
#   ③ 归一契约 Z = 轮廓收缩 × normdelta；normdelta = ν_BCS·Pf(G)，
#      ν_BCS = det(I + 𝔅·ℛ·𝔅†)/det(I + ℛ)（𝔅 = 浴块链乘、ℛ = e^{−β𝓗_B}；
#      构造侧语义 = 翻倍浴链配分比，与费米 bath_norm 的二次量子化迹比
#      同源）。Δ=0 二倍关系锚（金标准 T4a）：ν_BCS(Δ=0) = ν_ferm² 机器
#      精确（det(I+e^{−β𝓗}) 的 BdG 翻倍每对模计两次）。
# =============================================================================

"BCS 接口槽（c = 1..N × 支 × 味(1=↑,2=↓) × (out, in)，1-based 局部索引）：
每支点 4 槽 [(↑out, ↑in), (↓out, ↓in)]，− 支在后。"
_ifslot2(c::Int, br::Symbol, fl::Int, io::Symbol) =
    8 * (c - 1) + (br === :+ ? 1 : 5) + (fl === 1 ? 0 : 2) + (io === :out ? 0 : 1)

"""BdG 单粒子矩阵（置换序 [杂质粒子, 杂质空穴, 浴粒子, 浴空穴]，2(K+1)²）
与浴-only 版 𝓗_B（2K²，BdG 去杂质行/列）。h₊ = 星形（杂质 ε=0 + V 耦合），
B = Δ·diag(0, 1..1)（杂质槽不配对）。μ≠0 拒绝（照搬退役 bcs_nambu_data 守卫）。"""
function _bcs_bdgh(f::AbstractBoundedFunction, Δ::Number; β::Real, μ::Real=0,
                   gfsolver::ToulouseSolver=EDToulouseSolver(64))
    iszero(μ) || throw(ArgumentError("BCS 浴只支持 μ=0（μ = $μ）"))
    M = _nbath(gfsolver)
    Es, Vsq, wΓ = _quad_nodes(f, μ, M)
    keep = wΓ .> 0                      # 剔除谱为零区域的节点（死模不改配分）
    εs = Es[keep]
    Vs = Vsq[keep]
    K = length(εs)
    K >= 1 || throw(ArgumentError("谱在支撑上的积分为零（或非正）"))
    n = K + 1
    d = 2n
    ℋ = zeros(ComplexF64, d, d)
    for k in 1:K
        ℋ[1, 2+k] = conj(Vs[k])         # h₊：杂质粒子行 × 浴粒子列
        ℋ[2+k, 1] = Vs[k]
        ℋ[2+k, 2+k] = εs[k]             # 浴粒子对角
        ℋ[n+1+k, n+1+k] = -εs[k]        # −h₊ᵀ：浴空穴对角
        ℋ[2, n+1+k] = -Vs[k]            # −h₊ᵀ：杂质空穴行 × 浴空穴列
        ℋ[n+1+k, 2] = -conj(Vs[k])
        ℋ[2+k, n+1+k] = Δ               # B：浴粒子 ↔ 浴空穴（杂质槽不配对）
        ℋ[n+1+k, 2+k] = conj(Δ)         # B†
    end
    ℋ_B = zeros(ComplexF64, 2K, 2K)
    for k in 1:K
        ℋ_B[k, k] = εs[k]
        ℋ_B[K+k, K+k] = -εs[k]
        ℋ_B[k, K+k] = Δ
        ℋ_B[K+k, k] = conj(Δ)
    end
    return (; εs, Vs, K, ℋ, ℋ_B)
end

"""
    exact_bcs_Δt(f; β, N, t, Δ=0, μ=0, gfsolver=EDToulouseSolver(64))
        -> (BCSCorrelationFunction{ExactRealCorrelationFunction}, normdelta)

BCS（超导）浴严格 IF 核（实时 Keldysh 轮廓）——**纯耦合项严格核**
（exact 核族统一契约，见 exact_fermionic_Δt docstring；FormB 消元的
Nambu 翻倍推广，构造见文件头注）。

# 契约
1. **核 = 纯耦合项**：核内只含 BCS 浴（BdG 翻倍星形模）+ 杂质-浴耦合
   （谱 `f` 经 `gfsolver` 离散；杂质双味 d↑/d↓ 各以其味耦合浴，杂质槽
   不配对）与热力学 β——不含任何杂质哈密顿量信息（无 ε_d、无劈分阶数、
   无 kick 相位）。μ≠0 抛 `ArgumentError`（μ 只支持 0，照搬退役守卫）。
2. **严格性语义**：消费侧（杂质演化**全部**走 K 链门 + 接口反测度
   `counter_c = +1` + Keldysh 折返 + ρ₀）的轮廓收缩 = 同调度**直接
   Trotter 离散链的迹，任意 ε_d 机器精确**（门 = exp(−iα·ε_d·(n↑+n↓))；
   消费参考实现 = 包内金标准 test/exact/bcs_realtime.jl 的 ED 链锚；
   GTEMPO exact-BCS 消费面立项后接 AnomalousPairing 四键语义）。
3. **归一契约**：`Z = 轮廓收缩 × normdelta`（比值观测量严格相消）；
   `normdelta = ν_BCS·Pf(G)`，ν_BCS = det(I + 𝔅·ℛ·𝔅†)/det(I + ℛ)
   （𝔅 = 浴块链乘升序左乘、ℛ = e^{−β·𝓗_B}；Δ=0 二倍关系与 ED 痕比锚 =
   金标准 T4a）。

# 构造（FormB Nambu 翻倍，文件头注）
接口槽 8N + 链对 2N+1 × (out 2K + in 2K) 的大 Gaussian 经翻倍演化
𝒰 = e^{−i·cell·𝓗} 缝合、热边 e^{−β·𝓗_B} 进浴块（每对测度：粒子 −1/
空穴 +1）；Schur 消浴块 `effective = E + C·(B\\Cᵀ)`，反对称化
`G = antisym(effective)`，读矩阵 `Δ = −antisym(G⁻¹)` partner 交叉读出
四 Nambu 块。腿基与费米同构：点 1（t = 0/ρ₀ 端）与点 N+1（开放端恒等窗）
置零，Δ 支撑 = 接口点 1..N。离散谱（DiracDelta/DiscreteSpectrum）物理模式
直取；连续谱 GL-M 求积节点离散。复 Δ 允许（𝓗 厄米 ⟹ 演化幺正）。
"""
function exact_bcs_Δt(f::AbstractBoundedFunction; β::Real, N::Int, t::Real,
                      Δ::Number=0, μ::Real=0,
                      gfsolver::ToulouseSolver=EDToulouseSolver(64))
    data = _bcs_bdgh(f, Δ; β=β, μ=μ, gfsolver=gfsolver)
    (; εs, Vs, K, ℋ, ℋ_B) = data
    d = size(ℋ, 1)
    δt = Float64(t) / N
    cells = fill(δt, N)
    Us = [exp(-1im * cell .* ℋ) for cell in cells]
    nif = 8 * N
    total = nif + 4K * (2N + 1)
    A = zeros(ComplexF64, total, total)
    cursor = nif
    pair_data = NamedTuple[]
    for label in (:trace_final, :forward, :backward)
        count = label === :trace_final ? 1 : N
        for step in 0:(count - 1)
            outs = collect((cursor + 1):(cursor + 2K))
            ins = collect((cursor + 2K + 1):(cursor + 4K))
            push!(pair_data, (outs = outs, ins = ins))
            cursor += 4K
        end
    end
    trace_pair = pair_data[1]
    forward_pairs = pair_data[2:(N + 1)]
    backward_pairs = pair_data[(N + 2):end]
    # Q kernel 缝合（多接口槽版：U 前两行/列 = 杂质粒子/空穴，后段 = 浴）
    for step in 0:(N - 1)
        input_pair = forward_pairs[step + 1]
        output_pair = step == N - 1 ? trace_pair : forward_pairs[step + 2]
        _formb_qadd_ni!(A, [_ifslot2(step + 1, :+, 1, :out), _ifslot2(step + 1, :+, 2, :out)],
                           [_ifslot2(step + 1, :+, 1, :in), _ifslot2(step + 1, :+, 2, :in)],
                           output_pair.outs, input_pair.ins, Us[step + 1];
                           trace_bra = (step == N - 1))
    end
    for step in 0:(N - 1)
        output_pair = backward_pairs[step + 1]
        input_pair = step == N - 1 ? trace_pair : backward_pairs[step + 2]
        _formb_qadd_ni!(A, [_ifslot2(step + 1, :-, 1, :out), _ifslot2(step + 1, :-, 2, :out)],
                           [_ifslot2(step + 1, :-, 1, :in), _ifslot2(step + 1, :-, 2, :in)],
                           output_pair.outs, input_pair.ins, adjoint(Us[step + 1]))
    end
    # 热边：e^{−β·𝓗_B} 置 (fwd outs × bwd ins)（空穴扇区携 e^{+βε}）
    thermal = exp(Float64(β) .* (.-ℋ_B))
    for i in 1:2K, j in 1:2K
        _formb_put!(A, forward_pairs[1].outs[i], backward_pairs[1].ins[j], thermal[i, j])
    end
    # 每对测度：两扇区同 −1（Berezin RI 权重 e^{−ψ̄ψ} 逐味；探针标定 2026-11-25）
    for pair in pair_data, k in 1:2K
        _formb_put!(A, pair.outs[k], pair.ins[k], -1.0)
    end
    E = A[1:nif, 1:nif]
    C = A[1:nif, nif+1:end]
    B = A[nif+1:end, nif+1:end]
    effective = E + C * (B \ transpose(C))
    G = 0.5 .* (effective .- transpose(effective))
    # ν_BCS：浴链配分比（2K 维 det 形式；𝔅 链乘序 = cells 升序左乘，同费米）
    𝔅prod = Matrix{ComplexF64}(I, 2K, 2K)
    for c in 1:N
        𝔅prod = Us[c][3:d, 3:d] * 𝔅prod
    end
    ν = det(I + 𝔅prod * thermal * 𝔅prod') / det(I + thermal)
    Ginv = inv(G)
    GA = (Ginv .- transpose(Ginv)) ./ 2
    sgn = -1                            # FormB elim_sign：exp(−½jᵀG⁻¹j) 的整体负号
    # 四 Nambu 块读出（partner 交叉；row = Φ_p 侧、col = Φ_q† 侧；
    # 点 0/N+1 置零——容器 (N+1)² 网格的开放端恒等窗）
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
    uu = blk((i, b) -> inn(i, b, 1), (j, b) -> out(j, b, 1))   # ā↑Δa↑ = −GA[↑in₁,↑out₂]
    dd = blk((i, b) -> inn(i, b, 2), (j, b) -> out(j, b, 2))   # ā↓Δa↓ = −GA[↓in₁,↓out₂]
    ud = blk((i, b) -> inn(i, b, 1), (j, b) -> out(j, b, 2))   # ā↑Δā↓（ED 锚校准中）
    du = blk((i, b) -> inn(i, b, 2), (j, b) -> out(j, b, 1))   # a↓Δa↑（ED 锚校准中）
    return BCSCorrelationFunction(uu=uu, ud=ud, du=du, dd=dd), ν * _formb_pfaffian(G)
end
