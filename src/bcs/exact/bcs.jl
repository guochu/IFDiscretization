# =============================================================================
# exact —— BCS（超导）浴严格装配（Nambu/BdG 对角化）
#
# ← QuAPI/exactif/test/exact_bcs_api.jl 移植；2026-11-19 严格化修复：
#
#   ① 四块改存装配级严格容器（ExactReal/ExactMixedCorrelationFunction）。
#      raw Nambu 收缩的分支块相互独立（bath 端点闭式，无冻结核的 Toeplitz/
#      sector 结构），点值容器对非 Toeplitz 存储的对角块走 sector 推导
#      （_G₊₊ 从交叉块重构）——对严格核不成立（lossy）。费米严格核同因
#      迁严格容器（见 src/fermions/exact 头注）。
#
#   ② 实时/混合块取向修正为包约定：branch(b1,b2)[i,j] = 场₁@(b1,i)、
#      场₂@(b2,j) 槽位的 raw 收缩（row = Φ_p 侧时间/支，col = Φ_q† 侧）。
#      该槽位语义由 ED 验证过的 Grassmann 装配（test/api/bcs_grassmann.jl：
#      G[i+1,j+1] 乘 (场₁(u,i), 场₂(v,j))）与冻结核 branch 约定钉死。
#      原移植版存成交换支对+转置（G₊₋ 槽放 (−,+) 块的转置、G₋₊ 槽放
#      (+,−) 块的转置），按包约定消费时逐位错位。
#
#   ③ Δm 增补 τ₋/₋τ 两块（原点值容器靠 sector 推导，对 raw 收缩失真）；
#      G₊τ 改 T_C 闭式 ⟨T_C Φ_p(t⁺)Φ_q†(τ)⟩ = W(t)·M₀·W(im τ)（原版存
#      anti-T_C 物体 [W(im τ)M₀W(t)†]）。Gτ₊ 的 lesser 闭式本已正确。
#
# 费米 exactif 严格装配（星形 h₀ 对角化）的 Nambu 推广：对角化"杂质-热库
# 系数矩阵" = BdG 矩阵
#     𝓗 = [[h₊, B],[B†, −h₊ᵀ]]（厄米；h₊ = 星形杂质(ε=0)+bath，
#          B = Δ·diag(0,1..1)——杂质槽不配对；Δ 实/复均可）
# 热协方差：M₀ = V·diag(1/(1+e^{−βλ}))·V⁻¹ = I − f(𝓗)（整体式，逐元
# = ⟨ΦΦ†⟩；⟨Φ†Φ⟩ = f(𝓗) 逐元成立）。
# 时间域闭式（机器精度，与任何分割无关）：
#   虚时   𝒢_pq(τ) = +[e^{−τ𝓗}M₀]_{imp p,q}          （τ>0；KMS 反周期
#           延拓 𝒢_pq(−τ) = −𝒢_pq(β−τ)；δ 源(−1−𝒢(β⁻))住对角，同费米
#           exactif ηⱼₖ[1]=G(0⁻) 约定）
#   实时   raw 收缩 C(v,w) = ⟨T_C Φ_p(t_v)Φ_q†(t_w)⟩：
#           pos(v) ≥ pos(w)：[U_v M₀ U_w†]_{imp}（greater，M₀ 权）
#           pos(v) <  pos(w)：−[conj(U_w)(I−M₀)U_vᵀ]_{qp}（lesser，
#           反对易翻号；U = 轮廓演化算符，含正/反支结构）
#   混合   τ 交叉：Gτ± 走 lesser 分支（τ 早于一切 Keldysh 点，有界）、
#           ±τ 走 greater 闭式 [W(±t)·e^{+𝓗τ}·M₀]（KMS 上半平面延拓；
#           e^{λτ}/(1+e^{−βλ}) = e^{−λ(β−τ)}/(1+e^{βλ}) ⟹ τ∈[0,β] 有界）。
# 条目为 imp 2×2 Nambu 块：(1,1)=uu(正常↑↑)、(2,2)=dd(正常↓↓，洞侧
# Nambu(2,2) partner，Δ=0 时非对角 = −uuᵀ、对角差正则反对易接触)、
# (1,2)=ud(反常 cc)、(2,1)=du(反常 nn)，
# raw 收缩装入 BCSCorrelationFunction（无提取符号——Fock ED 逐位锚定 raw，
# 金标准 test/exact/bcs_{imagtime,realtime,mixedtime}.jl）。
# 谱离散：GL-M 节点（求积节点，M→∞ 收敛连续谱）；ε_d 不入核（调用方
# kick 簿记，同费米 exactif）。μ=0 convention（μ≠0 抛 ArgumentError）。
#
# 族契约（2026-11-22 对齐，族标准 = real.jl `exact_fermionic_Δt` docstring）：
#   ① 核 = 纯耦合项 ✓（Δ 配对属浴内容；无 ε_d、无 kick 相位；μ≠0 拒绝）；
#   ② 严格性语义 = 装配级 raw 收缩逐位 == BCS Fock ED（金标准
#      test/exact/bcs_{imagtime,realtime,mixedtime}.jl，实/复 Δ 定义级锚）；
#   ③ **normdelta 留白（刻意，待 BCS 严格后端立项）**：raw 收缩容器经
#      GTEMPO AnomalousPairing 的 deta 折叠在消费端转指数系数，归一常数
#      的定义依赖消费装配框架（Z = 收缩 × normdelta 的「收缩」未定）——
#      推导路线 = 轮廓 Nambu Gaussian（FormB-BCS：浴 Nambu 对沿轮廓 +
#      BdG 演化 + 测度/闭合边）Schur 消元，normdelta = ν·Pf(G_interface)，
#      Z 证书 = normdelta × 装配 Pfaffian == BCS Fock ED 迹；Δ=0 退化
#      交叉锚 = 费米 det(BB) 的二倍关系。路线图见 plan §8。
# 接口调整（rev 4）：`M::Int=64` → `gfsolver::ToulouseSolver
# = EDToulouseSolver(64)`，节点数取 `_nbath(gfsolver)`；同 gfsolver 下与原版
# 逐位一致。δτ/δt 旋钮（2026-11-19）：τ 网格步长显式化（默认 β/N，费米
# exact 核同款），统一入口 Δτ/Δm 的 δτ 由此透传。
# =============================================================================

"Nambu/BdG 严格数据：𝓗₀（厄米）本征分解 + M₀ + imp 槽位。μ≠0 拒绝。"
function bcs_nambu_data(f::AbstractBoundedFunction, Δ::Number; β::Real, μ::Real=0,
                        gfsolver::ToulouseSolver=EDToulouseSolver(64))
    iszero(μ) || throw(ArgumentError("BCS 浴只支持 μ=0（μ = $μ）"))
    M = _nbath(gfsolver)
    Es, Vsq, _ = _quad_nodes(f, 0.0, M)
    n = M + 1
    h₊ = zeros(ComplexF64, n, n)
    for k in 1:M
        h₊[1, 1+k] = Vsq[k]; h₊[1+k, 1] = conj(Vsq[k]); h₊[1+k, 1+k] = Es[k]
    end
    B = zeros(ComplexF64, n, n)
    for k in 2:n
        B[k, k] = Δ
    end
    𝓗₀ = [[h₊ B]; [conj(B) -transpose(h₊)]]
    vals, vecs = eigen(Hermitian(𝓗₀))
    M₀ = vecs * Diagonal(1 ./ (1 .+ exp.(-β * vals))) * vecs'
    imp = [1, n+1]                            # (d↑, d̄↓†) 槽位
    return (; n, 𝓗₀, vals, vecs, M₀, imp)
end

"𝒢_pq(τ>0) = +[e^{−τ𝓗}M₀]_{pq}（本征分解闭式；费米 exactif 族同号，
 Δ=0 时逐位退化为 Σa2ₚe^{−λτ}/(1+e^{−βλ)}）"
function bcs_Gtau(D, τ::Real, p::Int, q::Int)
    A = D.vecs * Diagonal(exp.(-D.vals * τ)) * (D.vecs' * D.M₀)
    return A[D.imp[p], D.imp[q]]
end

"𝒢_pq(τ≤0) KMS 反周期延拓：−𝒢_pq(β+τ)（δ 源属调用方环簿记，benchmark δrow）"
bcs_Gtau_neg(D, τ::Real, p::Int, q::Int, β::Real) =
    -bcs_Gtau(D, β + τ, p, q)

"""
    exact_bcs_Δτ(f; β, N, Δ, μ=0, δτ=β/N, gfsolver=EDToulouseSolver(64))
        -> BCSCorrelationFunction{ImagCorrelationFunction}

严格装配虚时核：四分量环 𝒢_pq(τ_m)（点值 + KMS/δ 跳变闭式），τ_m=(d−1)δτ。
环是 τ-差的 Toeplitz（KMS），CorrelationMatrix 向量存储无损，故保留点值容器
（费米 exact_fermionic_Δτ 同款）。

族契约（见文件头注）：纯耦合项 ✓、ED 数据锚 ✓；**normdelta 留白**（待
BCS 严格后端立项——τ 环 Nambu Gaussian 消元闭式 + Z 证书，plan §8）。
"""
function exact_bcs_Δτ(f::AbstractBoundedFunction; β::Real, N::Int, Δ::Number=0,
                      μ::Real=0, δτ::Real=β / N,
                      gfsolver::ToulouseSolver=EDToulouseSolver(64))
    D = bcs_nambu_data(f, Δ; β=β, μ=μ, gfsolver=gfsolver)
    Gpt(τ, p, q) = τ > 0 ? bcs_Gtau(D, τ, p, q) : bcs_Gtau_neg(D, τ, p, q, β)
    blocks = Matrix{ImagCorrelationFunction}(undef, 2, 2)
    for p in 1:2, q in 1:2
        ηⱼₖ = [Gpt((d - 1) * δτ, p, q) for d in 1:N]      # d=1 → G(0⁻) 全量对角
        ηₖⱼ = vcat([zero(Gpt(0.0, p, q))],                # 对角由 ηⱼₖ[1] 独担
                    [Gpt(-(d - 1) * δτ, p, q) for d in 2:N])
        blocks[p, q] = ImagCorrelationFunction(CorrelationMatrix(ηⱼₖ, ηₖⱼ))
    end
    # 反常块存 raw Nambu 收缩（无提取符号——sgn 是 benchmark 方案侧的
    # Grassmann 腿序校正，参照类对象不需要；ED 锚定锁定 raw）
    uu = blocks[1, 1]; dd = blocks[2, 2]
    ud = blocks[1, 2]; du = blocks[2, 1]
    return BCSCorrelationFunction(ud, uu, dd, du)   # (cc, cn, nc, nn) 位置序
end

"""
    exact_bcs_Δt(f; β, N, t, Δ, μ=0, gfsolver=EDToulouseSolver(64))
        -> BCSCorrelationFunction{ExactRealCorrelationFunction}

严格装配实时核：raw 轮廓收缩 C(v,w) = ⟨T_C Φ_p(t_v)Φ_q†(t_w)⟩（greater=M₀ 权 /
lesser=(I−M₀) 权 + 反对易翻号），四 Nambu 分量各自给出四分支块
G[b1,b2][i,j] = C(v=(b1,i), w=(b2,j))——row = Φ_p 侧、col = Φ_q† 侧的
包约定槽位（含 (+,+)、(−,−) 对角块；等时点走 greater/0⁺ 侧）。

族契约（见文件头注）：纯耦合项 ✓、ED 数据锚 ✓；**normdelta 留白**（待
BCS 严格后端立项——Keldysh 线 Nambu Gaussian 消元 + Z 证书，plan §8）。
"""
function exact_bcs_Δt(f::AbstractBoundedFunction; β::Real, N::Int, t::Real,
                      Δ::Number=0, μ::Real=0,
                      gfsolver::ToulouseSolver=EDToulouseSolver(64))
    D = bcs_nambu_data(f, Δ; β=β, μ=μ, gfsolver=gfsolver)
    δt = t / N
    n1 = N + 1
    W(x) = D.vecs * Diagonal(exp.(-im * D.vals * x)) * D.vecs'
    # 轮廓演化：v = 正支 i（pos = i），v = 反支 i（pos = N+(N−i)+1）
    U = Vector{Matrix{ComplexF64}}(undef, 2n1)
    pos = Vector{Int}(undef, 2n1)
    for i in 0:N
        U[i+1] = W(i * δt)
        pos[i+1] = i
        U[n1+i+1] = W(t) * W(i * δt)'      # 反支 i=0 为转折点(物理时间 t)
        pos[n1+i+1] = N + (N - i) + 1
    end
    # ⟨Φ†Φ⟩ = f(𝓗) 的轮廓传播因子 = conj(V)·diag(f)·Vᵀ（逐元恒等式）
    L = conj(D.vecs) * Diagonal(1 .- 1 ./ (1 .+ exp.(-β * D.vals))) * transpose(D.vecs)
    # 逐节点 2×n 切片：greater = (U_v·M₀)[imp,:]·U_w†[:,imp]；
    # lesser = −[(conj(U_w)·L)[imp,:]·U_vᵀ[:,imp]]_{qp}
    Pg = [U[v] * D.M₀ for v in 1:2n1]                       # n×n
    Rl = [conj(U[w]) * L for w in 1:2n1]                    # n×n
    function Cimp(v::Int, w::Int)
        if pos[v] >= pos[w]
            # (U_v·M₀·U_w†)[imp_p, imp_q] = Σ_k (U_vM₀)_{p,k}·conj(U_w)_{q,k}
            return [conj(dot(Pg[v][D.imp[p], :], U[w][D.imp[q], :])) for p in 1:2, q in 1:2]
        else
            Bm = Rl[w] * transpose(U[v])
            return [-Bm[D.imp[q], D.imp[p]] for p in 1:2, q in 1:2]
        end
    end
    blocks = Vector{ExactRealCorrelationFunction}(undef, 2 * 2)
    for p in 1:2, q in 1:2
        G₊₊ = [Cimp(i + 1, j + 1)[p, q] for i in 0:N, j in 0:N]
        G₊₋ = [Cimp(i + 1, n1 + j + 1)[p, q] for i in 0:N, j in 0:N]
        G₋₊ = [Cimp(n1 + i + 1, j + 1)[p, q] for i in 0:N, j in 0:N]
        G₋₋ = [Cimp(n1 + i + 1, n1 + j + 1)[p, q] for i in 0:N, j in 0:N]
        blocks[2 * (p - 1) + q] = ExactRealCorrelationFunction(G₊₊, G₊₋, G₋₊, G₋₋)
    end
    return BCSCorrelationFunction(blocks[2], blocks[1], blocks[4], blocks[3])  # (cc,cn,nc,nn)
end

"""
    exact_bcs_Δm(f; β, Nτ, t, Nt, Δ, μ=0, δτ=β/Nτ, gfsolver=EDToulouseSolver(64))
        -> BCSCorrelationFunction{ExactMixedCorrelationFunction}

严格装配混合核（Kadanoff-Baym L 轮廓，τ→+→−）：九分支块
G[b1,b2][i,j] = raw 收缩 ⟨T_C Φ_p(b1,i)Φ_q†(b2,j)⟩ 的包约定槽位——

  ττ = Δτ 同款 KMS 环；四 Keldysh 块 = Δt 同构造（含 (+,+)、(−,−) 对角）；
  τ 交叉（τ 节点早于一切 Keldysh 点）：
    Gτ±[i,j] = lesser 闭式（−[conj(U_w)·L·W(τ)ᵀ]，(I−M₀) 权，有界）
    G±τ[i,j] = greater 闭式 [W(±t)·e^{+𝓗τ}·M₀]（KMS 上半平面延拓；
               e^{λτ}/(1+e^{−βλ}) = e^{−λ(β−τ)}/(1+e^{βλ}) ⟹ τ∈[0,β] 有界）

节点序同冻结核 contour_ordering 混合序：τ i = 1..Nτ（τ_i=(i−1)δτ）、
Keldysh j = 1..Nt+1（每节点 + 先 − 后，t_{j−1}=(j−1)δt）。

族契约（见文件头注）：纯耦合项 ✓、ED 数据锚 ✓；**normdelta 留白**（待
BCS 严格后端立项——L 轮廓 Nambu Gaussian 消元 + Z 证书，plan §8）。
"""
function exact_bcs_Δm(f::AbstractBoundedFunction; β::Real, Nτ::Int, t::Real,
                      Nt::Int, Δ::Number=0, μ::Real=0, δτ::Real=β / Nτ,
                      gfsolver::ToulouseSolver=EDToulouseSolver(64))
    D = bcs_nambu_data(f, Δ; β=β, μ=μ, gfsolver=gfsolver)
    δt = t / Nt
    n1 = Nt + 1
    W(x) = D.vecs * Diagonal(exp.(-im * D.vals * x)) * D.vecs'
    Wim(x) = D.vecs * Diagonal(exp.(-D.vals * x)) * D.vecs'
    L = conj(D.vecs) * Diagonal(1 .- 1 ./ (1 .+ exp.(-β * D.vals))) * transpose(D.vecs)
    # Keldysh 段（同 Δt）：+ 支节点 1..n1，− 支节点 n1+1..2n1
    U = Vector{Matrix{ComplexF64}}(undef, 2n1)
    kpos = Vector{Int}(undef, 2n1)
    for i in 0:Nt
        U[i+1] = W(i * δt)
        kpos[i+1] = i
        U[n1+i+1] = W(t) * W(i * δt)'      # 反支 i=0 为转折点(物理时间 t)
        kpos[n1+i+1] = Nt + (Nt - i) + 1
    end
    Pg = [U[v] * D.M₀ for v in 1:2n1]
    Rl = [conj(U[w]) * L for w in 1:2n1]
    function Cimp(v::Int, w::Int)          # v, w 为 Keldysh 段腿编号
        if kpos[v] >= kpos[w]
            return [conj(dot(Pg[v][D.imp[p], :], U[w][D.imp[q], :])) for p in 1:2, q in 1:2]
        else
            Bm = Rl[w] * transpose(U[v])
            return [-Bm[D.imp[q], D.imp[p]] for p in 1:2, q in 1:2]
        end
    end
    # τ 交叉闭式：轮廓位 τ_i = i ≺ 一切 Keldysh 点（kpos 从 0 起）。
    # Gτ± = lesser（−[conj(U_w)·L·W(τ_i)ᵀ]_{qp}）；G±τ = greater 闭式
    # ⟨Φ(t^±)Φ†(τ)⟩ = [W(t)·e^{+𝓗τ}·M₀]_{imp}（KMS 上半平面延拓；
    # τ∈[0,β] 有界：e^{λτ}/(1+e^{−βλ}) = e^{−λ(β−τ)}/(1+e^{βλ}) ≤ 1）。
    Wτ = [Wim((i - 1) * δτ) for i in 1:Nτ]
    Wτ⁻ = [D.vecs * Diagonal(exp.(D.vals .* ((i - 1) * δτ))) * D.vecs' for i in 1:Nτ]
    function Ximp(i::Int, v::Int, lesser::Bool)
        if lesser                           # (τ_i, v)：τ 早 ⟹ lesser
            Bm = Rl[v] * transpose(Wτ[i])
            return [-Bm[D.imp[q], D.imp[p]] for p in 1:2, q in 1:2]
        else                                # (v, τ_i)：Keldysh 晚 ⟹ greater
            return [conj(dot(Pg[v][D.imp[p], :], Wτ⁻[i][D.imp[q], :]))
                    for p in 1:2, q in 1:2]
        end
    end
    Gpt(τ, p, q) = τ > 0 ? bcs_Gtau(D, τ, p, q) : bcs_Gtau_neg(D, τ, p, q, β)
    blocks = Vector{ExactMixedCorrelationFunction}(undef, 2 * 2)
    for p in 1:2, q in 1:2
        ηⱼₖ = [Gpt((d - 1) * δτ, p, q) for d in 1:Nτ]      # d=1 → G(0⁻) 全量对角
        ηₖⱼ = vcat([zero(Gpt(0.0, p, q))],
                    [Gpt(-(d - 1) * δτ, p, q) for d in 2:Nτ])
        Gττ = zeros(ComplexF64, Nτ, Nτ)
        for i in 1:Nτ, j in 1:Nτ                          # KMS 环：i≥j 取 +τ 差
            Gττ[i, j] = i >= j ? ηⱼₖ[i - j + 1] : ηₖⱼ[j - i + 1]
        end
        G₊₊ = [Cimp(i + 1, j + 1)[p, q] for i in 0:Nt, j in 0:Nt]
        G₊₋ = [Cimp(i + 1, n1 + j + 1)[p, q] for i in 0:Nt, j in 0:Nt]
        G₋₊ = [Cimp(n1 + i + 1, j + 1)[p, q] for i in 0:Nt, j in 0:Nt]
        G₋₋ = [Cimp(n1 + i + 1, n1 + j + 1)[p, q] for i in 0:Nt, j in 0:Nt]
        Gτ₊ = [Ximp(i, j, true)[p, q] for i in 1:Nτ, j in 1:n1]
        Gτ₋ = [Ximp(i, n1 + j, true)[p, q] for i in 1:Nτ, j in 1:n1]
        G₊τ = [Ximp(j, i, false)[p, q] for i in 1:n1, j in 1:Nτ]
        G₋τ = [Ximp(j, n1 + i, false)[p, q] for i in 1:n1, j in 1:Nτ]
        blocks[2 * (p - 1) + q] = ExactMixedCorrelationFunction(
            Gττ, Gτ₊, Gτ₋, G₊τ, G₋τ, G₊₊, G₊₋, G₋₊, G₋₋)
    end
    return BCSCorrelationFunction(blocks[2], blocks[1], blocks[4], blocks[3])  # (cc,cn,nc,nn)
end
