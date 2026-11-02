# =============================================================================
# strictkernel.jl —— ExactIF 严格离散 hybridization 核（费米虚时反周期环）
#
# 【劈分与记号】H = H_imp + H_hyb（H_hyb = 浴 + 杂质-热库耦合，杂质槽 ε=0；
# PRB 107, 125103 默认分割——IF 不依赖 H₁ 的**矩阵元**，正文 §III.B 明文：
# "the influence functional is independent of the details of Ĥ₁"）。
# n 阶劈分 cell 的单粒子矩阵（对角 H_imp：杂质踢相位 Π D(α_j)，Π α_j = δτ）：
#     W_cell = cell(order, δτ; x)   （x = e^{−δτ·ε_d}，见 test 金标准）
# 反周期环矩阵 M_full = −I + kron(W_cell, S)（S 反周期移位），杂质通道
# Schur 补
#     Δ(x, order) ≡ II − IB·BB⁻¹·BI
# 即**严格离散 hybridization 核**：装配 B = B(Δ) 单体（测度/着装动能/杂交
# 全部内含）后，Pfaffian 收缩满足金标准恒等式
#     Z_fock = c(N)·det(BB)·Pf(B)、n̄ 与 G(τ_m) 与同阶直接 Trotter 演化逐位
#（test/exact/strict_kernel.jl，orders 1/2/4/6 × N=1..8 数值钉死）。
#
# 【关于 ε_d（重要）】ε_d = 0（x = 1）时 Δ 是纯浴对象；ε_d ≠ 0 时踢相位
# 同时进入杂质动能通道与杂交收缩行（IB → x^{α/δτ}·IB⁰ 等），Δ 依赖
# (order, x)——这是有限 ε_d 下严格性的**必要条件**（金标准数值证明：
# 无踢核在同阶直接演化下 n̄/G 偏差 O(1)）。连续理论中 IF 不依赖 H₁ 与此
# 不矛盾：IF 依赖的是**劈分约定**（H₁ 以整步传播子在 H_hyb cell 之外），
# 矩阵元无关。
#
# 【平移不变性】order 1/2：Δ 保持双向量环结构（正向滞后只依赖 i−j、反向
# 只依赖 j−i）→ 可存 CorrelationMatrix；order 4/6 的 fractal 子相位在
# cell 内非均匀 ⟹ 双向量可能破坏，返回原始矩阵。接触约定：对角全量入
# ηⱼₖ[1]，ηₖⱼ[1] = 0。
# =============================================================================

"""
    StrictFermionicKernel

ExactIF 严格离散 hybridization 核（虚时反周期环，(order, x) 特定）：

- `K`:  N×N 核矩阵 Δ(x, order)（测度/着装动能/Schur 杂交全部内含）；
- `N`:  环节点数；
- `detBB`:  浴涨落行列式 det(BB)（Z 归一因子；n̄/G 观测量不需要）。

装配契约（金标准）：`B[ā_i, a_j] = Δ[i, j]` 单体；Pfaffian 收缩满足
`Z_fock = c(N)·det(BB)·Pf(B)`、n̄/G 与同阶直接 Trotter 演化逐位。
"""
struct StrictFermionicKernel
    K::Matrix{ComplexF64}
    N::Int
    detBB::ComplexF64
end

"n 阶劈分 cell 的单粒子矩阵（对角 H_imp：每子步踢 D(α) = x^α·I 嵌入，Π α_j = δτ）"
function _split_cell(order::Int, θ::Real, Wb::AbstractMatrix, x::Number)
    D(α) = begin E = Matrix{ComplexF64}(I, size(Wb, 1), size(Wb, 1)); E[1,1] = x^(α/θ); E end
    cell1(θ) = D(θ) * Wb
    cell2(θ) = D(θ/2) * Wb * D(θ/2)
    function cell(order, θ)
        order == 1 && return cell1(θ)
        order == 2 && return cell2(θ)
        sub = order - 2
        p = θ/(4 - 4^(1/(order-1)))
        r = cell(sub, p); r = cell(sub, p)*r; r = cell(sub, θ-4p)*r; r = cell(sub, p)*r
        cell(sub, p)*r
    end
    return cell(order, θ)
end

"""
    strict_fermionic_Δτ(f; β, N, order=2, μ=0, δτ=β/N,
                        gfsolver=EigenSolver(64), imp_phase=1.0)
        -> StrictFermionicKernel

ExactIF 严格离散 hybridization 核（虚时反周期环，(order, imp_phase) 特定）。

- `f`:  `AbstractBoundedFunction`（H_hyb 谱；GL-M 节点数 = `gfsolver` 的
  `nbath`）；
- `order`:  劈分阶数（1 = 全步踢一端；2 = Strang 半步两端；4/6 = Suzuki
  fractal）；
- `imp_phase`:  杂质动能相位 x = e^{−δτ·ε_d}（对角单能级 H_imp 的 symbol；
  默认 1 即 ε_d = 0）。x ≠ 1 时踢相位进入动能通道与杂交收缩行——有限
  ε_d 严格性的必要条件（金标准数值证明）。

与点值格林函数路线（`pointwise_fermionic_Gτ`）不同，本输出是**装配级
配对核**：单体装配后与同阶直接 Trotter 演化严格一致。
"""
function strict_fermionic_Δτ(f::AbstractBoundedFunction;
                             β::Real, N::Int, order::Int=2, μ::Real=0,
                             δτ::Real=β / N,
                             gfsolver::ToulouseGreenFunctionSolver=EigenSolver(64),
                             imp_phase::Number=1.0)
    M = _nbath(gfsolver)
    Es, Vs2, _ = _quad_nodes(f, μ, M)
    h0 = zeros(ComplexF64, M + 1, M + 1)
    for k in 1:M
        h0[1, 1+k] = Vs2[k]; h0[1+k, 1] = conj(Vs2[k]); h0[1+k, 1+k] = Es[k]
    end
    Wb = exp(-δτ .* h0)
    Wcell = _split_cell(order, δτ, Wb, imp_phase)
    S = zeros(ComplexF64, N, N)
    for k in 2:N; S[k, k-1] = 1.0; end
    S[1, N] = -1.0                                    # 反周期扭转
    MM = -Matrix{ComplexF64}(I, (M+1)*N, (M+1)*N) + kron(Wcell, S)
    II, IB, BI, BB = MM[1:N, 1:N], MM[1:N, N+1:end], MM[N+1:end, 1:N], MM[N+1:end, N+1:end]
    Δ = II - IB * (BB \ BI)
    detBB = det(BB)
    # 双向量（order 1/2）：存 CorrelationMatrix；order 4/6 fractal：原始矩阵
    if order ∈ (1, 2)
        ηⱼₖ = [Δ[1+d, 1] for d in 0:N-1]
        ηₖⱼ = vcat([0.0 + 0.0im], [Δ[1, 1+d] for d in 1:N-1])
        K = Matrix(CorrelationMatrix(ηⱼₖ, ηₖⱼ))
        return StrictFermionicKernel(K, N, detBB)
    end
    return StrictFermionicKernel(Δ, N, detBB)
end

"点值格林函数路线（MPS-IF 列递推消费的对象；非装配配对核）。"
function pointwise_fermionic_Gτ(f::AbstractBoundedFunction;
                                β::Real, N::Int, μ::Real=0, δτ::Real=β / N,
                                gfsolver::ToulouseGreenFunctionSolver=EigenSolver(64))
    Es, Vs2, _ = _quad_nodes(f, μ, _nbath(gfsolver))
    ev = _bath_eig(Es, Vs2)
    a2 = abs2.(ev.vectors[1, :])
    Mp1 = length(Es) + 1
    G0(τ) = τ > 0 ?
            sum(a2[p] * exp(-ev.values[p] * τ) / (1 + exp(-β * ev.values[p])) for p in 1:Mp1) :
            -sum(a2[p] * exp(-ev.values[p] * (τ + β)) / (1 + exp(-β * ev.values[p])) for p in 1:Mp1)
    ηⱼₖ = [G0((d - 1) * δτ) for d in 1:N]
    ηₖⱼ = [G0(-(d - 1) * δτ) for d in 1:N]
    return ImagCorrelationFunction(CorrelationMatrix(ηⱼₖ, ηₖⱼ))
end
