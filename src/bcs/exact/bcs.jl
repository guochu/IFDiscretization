# =============================================================================
# exact —— BCS（超导）浴严格装配（Nambu/BdG 对角化）
#
# ← QuAPI/exactif/test/exact_bcs_api.jl 原样移植（kernel 未动）。
#
# 费米 exactif 严格装配（星形 h₀ 对角化）的 Nambu 推广：对角化"杂质-热库
# 系数矩阵" = BdG 矩阵
#     𝓗 = [[h₊, B],[B†, −h₊ᵀ]]（厄米；h₊ = 星形杂质(ε=0)+bath，
#          B = Δ·diag(0,1..1)——杂质槽不配对；Δ 实/复均可）
# 热协方差：M₀ = V·diag(1/(1+e^{−βλ}))·V⁻¹ = I − f(𝓗)（整体式，逐元
# = ⟨ΦΦ†⟩；⟨Φ†Φ⟩ = f(𝓗) 逐元成立——benchmark bcs_M0 同一恒等式）。
# 时间域闭式（机器精度，与任何分割无关）：
#   虚时   𝒢_pq(τ) = +[e^{−τ𝓗}M₀]_{imp p,q}          （τ>0；KMS 反周期
#           延拓 𝒢_pq(−τ) = −𝒢_pq(β−τ)；δ 源(−1−𝒢(β⁻))属调用方环簿记，
#           同费米 exactif ηⱼₖ[1]=G(0⁻) 约定）
#   实时   轮廓收缩 C(v,w) = ⟨T_C Φ_p(t_v)Φ_q†(t_w)⟩：
#           pos(v) ≥ pos(w)：[U_v M₀ U_w†]_{imp}（greater，M₀ 权）
#           pos(v) <  pos(w)：−[conj(U_w)(I−M₀)U_vᵀ]_{qp}（lesser，
#           反对易翻号；U = 轮廓演化算符，含正/反支结构）
#   混合   G₊τ[a,i] = [e^{−τᵢ𝓗}M₀e^{−i𝓗t_a}]_{imp}（晚侧闭式，有界——
#           环归一化 M₀ 吸收 e^{+|λ|β}，同费米版教训）；Gτ₊ 由 lesser
#           分支给出（有界）。
# 条目为 imp 2×2 Nambu 块：(1,1)=uu(正常↑↑)、(2,2)=dd(正常↓↓)、
# (1,2)=ud(反常 cc)、(2,1)=du(反常 nn)，raw 收缩装入
# BCSCorrelationFunction（无提取符号——Fock ED 逐位锚定 raw）。
# 谱离散：GL-M 节点（求积节点，M→∞ 收敛连续谱）；ε_d 不入核（调用方
# kick 簿记，同费米 exactif）。μ=0 convention（μ≠0 抛 ArgumentError）。
#
# 接口调整（rev 4）：`M::Int=64` → `gfsolver::ToulouseGreenFunctionSolver
# = EigenSolver(64)`，节点数取 `_nbath(gfsolver)`；同 gfsolver 下与原版
# 逐位一致。
# =============================================================================

"Nambu/BdG 严格数据：𝓗₀（厄米）本征分解 + M₀ + imp 槽位。μ≠0 拒绝。"
function bcs_nambu_data(f::AbstractBoundedFunction, Δ::Number; β::Real, μ::Real=0,
                        gfsolver::ToulouseGreenFunctionSolver=EigenSolver(64))
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
    exact_bcs_Δτ(f; β, N, Δ, μ=0, gfsolver=EigenSolver(64))
        -> BCSCorrelationFunction{ImagCorrelationFunction}

严格装配虚时核：四分量环 𝒢_pq(τ_m)（点值 + KMS/δ 跳变闭式），τ_m=(d−1)δτ。
"""
function exact_bcs_Δτ(f::AbstractBoundedFunction; β::Real, N::Int, Δ::Number=0,
                      μ::Real=0, gfsolver::ToulouseGreenFunctionSolver=EigenSolver(64))
    D = bcs_nambu_data(f, Δ; β=β, μ=μ, gfsolver=gfsolver)
    δτ = β / N
    Gpt(τ, p, q) = τ > 0 ? bcs_Gtau(D, τ, p, q) : bcs_Gtau_neg(D, τ, p, q, β)
    blocks = Matrix{ImagCorrelationFunction}(undef, 2, 2)
    for p in 1:2, q in 1:2
        ηⱼₖ = [Gpt((d - 1) * δτ, p, q) for d in 1:N]      # d=1 → G(0⁻)
        ηₖⱼ = [Gpt(-(d - 1) * δτ, p, q) for d in 1:N]
        blocks[p, q] = ImagCorrelationFunction(CorrelationMatrix(ηⱼₖ, ηₖⱼ))
    end
    # 反常块存 raw Nambu 收缩（无提取符号——sgn 是 benchmark 方案侧的
    # Grassmann 腿序校正，参照类对象不需要；ED 锚定锁定 raw）
    uu = blocks[1, 1]; dd = blocks[2, 2]
    ud = blocks[1, 2]; du = blocks[2, 1]
    return BCSCorrelationFunction(ud, uu, dd, du)   # (cc, cn, nc, nn) 位置序
end

"""
    exact_bcs_Δt(f; β, N, t, Δ, μ=0, gfsolver=EigenSolver(64))
        -> BCSCorrelationFunction{RealCorrelationFunction}

严格装配实时核：轮廓收缩 C(v,w)（greater=M₀ 权 / lesser=(I−M₀) 权 + 反对易
翻号），四 Nambu 分量各自给出 (G₊₋, G₋₊) 存储对。
"""
function exact_bcs_Δt(f::AbstractBoundedFunction; β::Real, N::Int, t::Real,
                      Δ::Number=0, μ::Real=0,
                      gfsolver::ToulouseGreenFunctionSolver=EigenSolver(64))
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
    function Cimp(v::Int, w::Int)
        if pos[v] >= pos[w]
            A = U[v] * D.M₀ * U[w]'
            return [A[D.imp[p], D.imp[q]] for p in 1:2, q in 1:2]
        else
            Bm = conj(U[w]) * L * transpose(U[v])
            return [-Bm[D.imp[q], D.imp[p]] for p in 1:2, q in 1:2]
        end
    end
    blocks = Matrix{RealCorrelationFunction}(undef, 2, 2)
    for p in 1:2, q in 1:2
        G₊₋ = [Cimp(n1 + j + 1, i + 1)[p, q] for i in 0:N, j in 0:N]
        G₋₊ = [Cimp(j + 1, n1 + i + 1)[p, q] for i in 0:N, j in 0:N]
        blocks[p, q] = RealCorrelationFunction(G₊₋, G₋₊)
    end
    ud = blocks[1, 2]; du = blocks[2, 1]
    return BCSCorrelationFunction(ud, blocks[1, 1], blocks[2, 2], du)
end

"""
    exact_bcs_Δm(f; β, Nτ, t, Nt, Δ, μ=0, gfsolver=EigenSolver(64))
        -> BCSCorrelationFunction{MixedCorrelationFunction}

严格装配混合核：ττ = Δτ 环数据；Keldysh 交叉 = Δt 同构造；τ-交叉闭式
G₊τ[a+1,i] = [e^{−τᵢ𝓗}M₀e^{−i𝓗t_a}]_{imp}（环归一化有界），Gτ₊ 走 lesser
分支（有界）。
"""
function exact_bcs_Δm(f::AbstractBoundedFunction; β::Real, Nτ::Int, t::Real,
                      Nt::Int, Δ::Number=0, μ::Real=0,
                      gfsolver::ToulouseGreenFunctionSolver=EigenSolver(64))
    D = bcs_nambu_data(f, Δ; β=β, μ=μ, gfsolver=gfsolver)
    δτ = β / Nτ
    δt = t / Nt
    n1 = Nt + 1
    W(x) = D.vecs * Diagonal(exp.(-im * D.vals * x)) * D.vecs'
    Wim(x) = D.vecs * Diagonal(exp.(-D.vals * x)) * D.vecs'
    L = conj(D.vecs) * Diagonal(1 .- 1 ./ (1 .+ exp.(-β * D.vals))) * transpose(D.vecs)
    # Keldysh 交叉块（同 Δt）
    U = Vector{Matrix{ComplexF64}}(undef, 2n1)
    pos = Vector{Int}(undef, 2n1)
    for i in 0:Nt
        U[i+1] = W(i * δt)
        pos[i+1] = i
        U[n1+i+1] = W(t) * W(i * δt)'      # 反支 i=0 为转折点(物理时间 t)
        pos[n1+i+1] = Nt + (Nt - i) + 1
    end
    function Cimp(v::Int, w::Int)
        if pos[v] >= pos[w]
            A = U[v] * D.M₀ * U[w]'
            return [A[D.imp[p], D.imp[q]] for p in 1:2, q in 1:2]
        else
            Bm = conj(U[w]) * L * transpose(U[v])
            return [-Bm[D.imp[q], D.imp[p]] for p in 1:2, q in 1:2]
        end
    end
    # τ-交叉闭式：G₊τ = [Wim(τᵢ)·M₀·W(t_a)']_{imp}；Gτ₊ 走 lesser 分支（有界）
    function Ximp(i::Int, a::Int)
        Uτ = Wim((i - 1) * δτ)
        Ut = W(a * δt)
        A = Uτ * D.M₀ * Ut'
        Bm = conj(Ut) * L * transpose(Uτ)
        return ([(A[D.imp[p], D.imp[q]]) for p in 1:2, q in 1:2],
                [-(Bm[D.imp[q], D.imp[p]]) for p in 1:2, q in 1:2])
    end
    Gpt(τ, p, q) = τ > 0 ? bcs_Gtau(D, τ, p, q) : bcs_Gtau_neg(D, τ, p, q, β)
    blocks = Matrix{MixedCorrelationFunction}(undef, 2, 2)
    for p in 1:2, q in 1:2
        ηⱼₖ = [Gpt((d - 1) * δτ, p, q) for d in 1:Nτ]     # d=1 → G(0⁻)
        ηₖⱼ = [Gpt(-(d - 1) * δτ, p, q) for d in 1:Nτ]
        G₊₋ = [Cimp(n1 + j + 1, i + 1)[p, q] for i in 0:Nt, j in 0:Nt]
        G₋₊ = [Cimp(j + 1, n1 + i + 1)[p, q] for i in 0:Nt, j in 0:Nt]
        G₊τ = [Ximp(i, a)[1][p, q] for a in 0:Nt, i in 1:Nτ]
        Gτ₊ = [Ximp(i, a)[2][p, q] for i in 1:Nτ, a in 0:Nt]
        blocks[p, q] = MixedCorrelationFunction(G₊₋, G₋₊, G₊τ, Gτ₊,
                                                CorrelationMatrix(ηⱼₖ, ηₖⱼ))
    end
    ud = blocks[1, 2]; du = blocks[2, 1]
    return BCSCorrelationFunction(ud, blocks[1, 1], blocks[2, 2], du)
end
