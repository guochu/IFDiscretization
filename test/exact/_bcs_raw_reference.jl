# =============================================================================
# _bcs_raw_reference.jl —— 退役 raw 收缩闭式核（测试侧参照 helper，D4 裁决）
#
# 2026-11-25 轴面收敛退役的 `exact_bcs_Δt`（Nambu/BdG 本征分解闭式轮廓收缩，
# 16 维 Fock ED 逐位锚定）整段恢复为测试侧参照：与新 FormB 翻倍核**不是同一
# 物体**（raw 收缩 vs 消元核系数），二者共同锚 = ED，不互相对拍。
# 用途：T5 参照自检（旧 T1 断言复活：四块 × 四支对 × 全网格 == ED 轮廓收缩）
# ——证明参照面自身有效，为 GTEMPO 消费面立项保留 ED 锚机器。
# 出处 = git `3cdae62^` src/bcs/exact/bcs.jl（μ=0 守卫与槽位约定逐字保留）。
# =============================================================================
"""
    raw_reference_bcs_Δt(f; β, N, t, Δ=0, μ=0, gfsolver=EDToulouseSolver(64))
        -> BCSCorrelationFunction{ExactRealCorrelationFunction}

退役 raw 收缩闭式：raw 轮廓收缩 C(v,w) = ⟨T_C Φ_p(t_v)Φ_q†(t_w)⟩
（greater = M₀ 权 / lesser = (I−M₀) 权 + 反对易翻号），四 Nambu 分量各自
给出四分支块（row = Φ_p 侧、col = Φ_q† 侧，等时走 greater 侧）。
"""
function raw_reference_bcs_Δt(f::AbstractBoundedFunction; β::Real, N::Int, t::Real,
                              Δ::Number=0, μ::Real=0,
                              gfsolver::ToulouseSolver=EDToulouseSolver(64))
    D = _raw_nambu_data(f, Δ; β=β, μ=μ, gfsolver=gfsolver)
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

"Nambu/BdG 严格数据：𝓗₀（厄米）本征分解 + M₀ + imp 槽位。μ≠0 拒绝。
（退役 bcs_nambu_data 逐字，函数改名避让）"
function _raw_nambu_data(f::AbstractBoundedFunction, Δ::Number; β::Real, μ::Real=0,
                         gfsolver::ToulouseSolver=EDToulouseSolver(64))
    iszero(μ) || throw(ArgumentError("BCS 浴只支持 μ=0（μ = $μ）"))
    M = IFD._nbath(gfsolver)
    Es, Vsq, _ = IFD._quad_nodes(f, 0.0, M)
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
