# =============================================================================
# exact/mixed.jl —— exact_fermionic_Δm：混合 Kadanoff–Baym 轮廓严格 IF 核
# 契约 = 纯耦合项严格核（族契约见 real.jl exact_fermionic_Δt docstring）。
# 2026-11-22 契约换代：kicksτ/kickst 接口（杂质相位进核）从公共 API 移除——
# 带踢构造保留于 _mixed_kernel（内部），金标准测试经 test/exact/_kicks_helper.jl
# 的 exact_fermionic_Δm_kicks 消费（同 exact_fermionic_Δt_cells 先例）。
# =============================================================================
# ------------------------------ 混合时间 ------------------------------ #

"""
    exact_fermionic_Δm(f; β, Nτ, t, Nt, μ=0, δτ=β/Nτ, gfsolver=EDToulouseSolver(64))
        -> (ExactMixedCorrelationFunction, normdelta)

ExactEqualDistantKernel 严格 IF 核（混合 Kadanoff–Baym L 轮廓）——**纯耦合项
严格核**（exact 核族统一契约；Δτ/Δt 同族按此对齐）：

# 契约
1. **核 = 纯耦合项**：核内只含浴 + 杂质-浴耦合 h₀（谱 `f` 经 `gfsolver`
   离散的星形模）与热力学 β——**不含任何杂质哈密顿量信息**（无 ε_d、无
   劈分阶数、无 kick 相位）。IF 装配对一般杂质模型无假设（纯浴对象）。
   （历史：2026-11-22 前公共 API 带 kicksτ/kickst 杂质相位接口，已移测试
   helper `exact_fermionic_Δm_kicks`——带踢 Z ≡ τ 环踢乘积迹证书是底层
   W/Schur 数学的最锐利锚，金标准专用。）
2. **严格性语义**：装配级消费（紧凑 (ā,a) 链序装配 = 测度 −1 + 单位链 +
   两条闭合边，金标准 `test/exact/fermionic_mixedtime.jl` 的
   `mixed_assemble`）的轮廓收缩 Z 与 τ 环直接 Fock 演化迹 **机器精确**：
   Z == Tr_Fock[e^{−β·h₀ 二次量子化}]/Z_bath（纯核与杂质无关；
   带踢版同式 == τ 环踢乘积迹/Z_bath——Keldysh 段算符层面精确相消
   P₊P₋ = I，与实时段离散化无关）。腿基同冻结核（contour_ordering
   混合序），可直接替换 `fermionic_Δm` 的 Δ。
3. **归一契约**：`Z = 装配收缩 × normdelta`（比值观测量严格相消）；
   **normdelta = (−1)^{n(K+1)}·det(BB)/Z_bath**（n = Nτ + 2(Nt+1) 轮廓
   节点数；折叠 Berezin 符号 + impurity 折叠，K=1/K=2 × 奇偶 n 四点机器
   钉定）。BB 消元走环状块结构求解（单环仿射前推，O(n·K²·Limp + K³)；
   稠密参照 `_exrt_mixed_kernel_dense`）。死模过滤与 Δt/Δτ 一致。

轮廓（单环）τ₁→…→τ_{Nτ}→(+,1)→…→(+,Nt+1)→(−,Nt+1)→…→(−,1)→τ₁：τ cells
（含结合 cell τ_{Nτ}→+₁）用 W^τ = e^{−δτ·h}，实时 ± cells 用 W^t = e^{−iδt·h}
（− 支共轭），(+,Nt+1)→(−,Nt+1) 闭合边带费米迹负号、(−,1)→τ₁ 环回边权 1
（热演化在 τ 段显式进行，无热边 r）。
"""
function exact_fermionic_Δm(f::AbstractBoundedFunction; β::Real, Nτ::Int, t::Real,
                            Nt::Int, μ::Real=0, δτ::Real=β / Nτ,
                            gfsolver::ToulouseSolver=EDToulouseSolver(64))
    Nτ >= 1 || throw(ArgumentError("Nτ 至少为 1"))
    Nt >= 1 || throw(ArgumentError("Nt 至少为 1"))
    _mixed_kernel(f, β, Nτ, t, Nt, Float64(δτ), μ, _nbath(gfsolver),
                  ones(ComplexF64, Nτ), ones(ComplexF64, Nt); dense=false)
end

# 装配本体（dense=false 结构化求解 / dense=true 稠密参照）
function _mixed_kernel(f::AbstractBoundedFunction, β::Real, Nτ::Int, t::Real, Nt::Int,
                       δτ::Float64, μ::Real, M::Int,
                       kicksτ::Vector{ComplexF64}, kickst::Vector{ComplexF64};
                       dense::Bool)
    Es, Vsq, wΓ = _quad_nodes(f, μ, M)
    keep = wΓ .> 0
    εs = Es[keep]
    Vs = Vsq[keep]
    K = length(εs)
    K >= 1 || throw(ArgumentError("谱在支撑上的积分为零（或非正）"))
    _W(step, sgn) = begin
        h = zeros(ComplexF64, K + 1, K + 1)
        for k in 1:K
            h[1, k+1] = h[k+1, 1] = Vs[k]
            h[k+1, k+1] = εs[k]
        end
        exp(sgn * step .* h)
    end
    Wτ = _W(δτ, -1.0 + 0.0im)                    # 虚时 cell（含结合 cell 共用）
    Wt = _W(t / Nt, -1.0im)                      # 实时 cell
    n = Nτ + 2 * (Nt + 1)                        # 轮廓节点数（链序）
    ipos(c, ā) = 2 * c - (ā ? 0 : 1)             # 杂质腿（a = 2c−1, ā = 2c）
    bidx(k, c) = (k - 1) * n + c                 # bath 配对槽（行 γ̄，列 γ）
    Limp = 2n
    Nb = K * n
    II = zeros(ComplexF64, Limp, Limp)
    IB = zeros(ComplexF64, Limp, Nb)
    BI = zeros(ComplexF64, Nb, Limp)
    BB = dense ? zeros(ComplexF64, Nb, Nb) : Matrix{ComplexF64}(undef, 0, 0)
    for c in 1:n                                 # Berezin 测度
        II[ipos(c, true), ipos(c, false)] -= 1
        dense && for k in 1:K
            BB[bidx(k, c), bidx(k, c)] -= 1
        end
    end
    # cell 边（dest ← src）：+ 方向 D(x)·W 杂质输出行 ×x；− 方向 W†D† 列 ×x̄
    function edge!(cdest, csrc, W, x)
        II[ipos(cdest, true), ipos(csrc, false)] += x * W[1, 1]
        dense && for km in 1:K, kn in 1:K
            BB[bidx(km, cdest), bidx(kn, csrc)] += W[km+1, kn+1]
        end
        for km in 1:K
            IB[ipos(cdest, true), bidx(km, csrc)] += x * W[1, km+1]
            BI[bidx(km, cdest), ipos(csrc, false)] += W[km+1, 1]
        end
    end
    function edge_adj!(cdest, csrc, W, x)
        # 伴随 W†D†：conj-转置 + 杂质输入列 ×x̄
        xc = conj(x)
        II[ipos(cdest, true), ipos(csrc, false)] += xc * conj(W[1, 1])
        dense && for km in 1:K, kn in 1:K
            BB[bidx(km, cdest), bidx(kn, csrc)] += conj(W[kn+1, km+1])
        end
        for km in 1:K
            IB[ipos(cdest, true), bidx(km, csrc)] += conj(W[km+1, 1])
            BI[bidx(km, cdest), ipos(csrc, false)] += xc * conj(W[1, km+1])
        end
    end
    for c in 2:Nτ                               # τ 内部 cells
        edge!(c, c - 1, Wτ, kicksτ[c - 1])
    end
    edge!(Nτ + 1, Nτ, Wτ, kicksτ[Nτ])           # 结合 cell（τ_Nτ → β ≡ +₁）
    for jj in 1:Nt                              # + cells（cp(jj) → cp(jj+1)）
        edge!(Nτ + 2jj + 1, Nτ + 2jj - 1, Wt, kickst[jj])
    end
    for jj in 1:Nt                              # − cells（cm(jj+1) → cm(jj)，后向）
        edge_adj!(Nτ + 2jj, Nτ + 2jj + 2, Wt, kickst[jj])
    end
    for k in 1:K                                # bath 闭合边与环回边
        dense && begin
            BB[bidx(k, Nτ + 2Nt + 2), bidx(k, Nτ + 2Nt + 1)] += -1   # 费米迹
            BB[bidx(k, 1), bidx(k, Nτ + 2)] += 1                     # (−,1)→τ₁
        end
    end
    # 杂质轮廓闭合边（Schur 前，每腿一次）：Keldysh 迹负号 + 环回单位
    II[ipos(Nτ + 2Nt + 2, true), ipos(Nτ + 2Nt + 1, false)] += -1
    II[ipos(1, true), ipos(Nτ + 2, false)] += 1
    # 消浴：结构化（环）或稠密参照（混合轮廓无热边——τ 段显式演化）
    Wbτ = Wτ[2:end, 2:end]
    Wbt = Wt[2:end, 2:end]
    if dense
        X = BB \ BI
        detBB = det(BB)
    else
        X, detBB = _exrt_schur_mixed_cyclic(Wbτ, Wbt, Nτ, Nt, BI, K)
    end
    Mt = II - IB * X
    # 单位骨架：测度 + 沿轮廓单位链 + 两条闭合边
    C0 = zeros(ComplexF64, Limp, Limp)
    for c in 1:n
        C0[ipos(c, true), ipos(c, false)] -= 1
    end
    C0[ipos(Nτ + 1, true), ipos(Nτ, false)] += 1
    C0[ipos(Nτ + 2Nt + 2, true), ipos(Nτ + 2Nt + 1, false)] += -1
    C0[ipos(1, true), ipos(Nτ + 2, false)] += 1
    for c in 2:Nτ
        C0[ipos(c, true), ipos(c - 1, false)] += 1
    end
    for jj in 1:Nt
        C0[ipos(Nτ + 2jj + 1, true), ipos(Nτ + 2jj - 1, false)] += 1
        C0[ipos(Nτ + 2jj, true), ipos(Nτ + 2jj + 2, false)] += 1
    end
    basis = GrassmannContourIndex[
        vcat([GrassmannContourIndex(j, :τ, 1, c) for j in 1:Nτ for c in (false, true)],
             [GrassmannContourIndex(jj, br, 1, c)
              for jj in 1:Nt+1 for br in (:+, :-) for c in (false, true)])...]
    Zb = prod(1 .+ exp.(-β .* εs))
    # (−1)^{n(K+1)}：折叠 Berezin 符号 (−1)^{nK}（bath 侧，K = 模式数）
    # × impurity 折叠 (−1)^n（本 det 收缩机器相对 QuAPI Zvalue 机器的约定差，
    # K=1/Nτ=3 vs Nτ=2 四点机器钉定）
    pref = (-1.0 + 0.0im)^(n * (K + 1)) * detBB / Zb
    return _strict_mixed_from_legpos(basis, Mt - C0), pref
end

# 稠密参照路径（机器等价性检验用）
function _exrt_mixed_kernel_dense(f::AbstractBoundedFunction, β::Real, Nτ::Int, t::Real,
                                  Nt::Int, δτ::Real, μ::Real, M::Int,
                                  kicksτ::Vector{ComplexF64}, kickst::Vector{ComplexF64})
    _mixed_kernel(f, β, Nτ, t, Nt, Float64(δτ), μ, M, kicksτ, kickst; dense=true)
end

"""
混合轮廓 BB 的环状块结构求解（分配函数内部）。

 bath 块沿轮廓构成**单环**（每行/列恰一个非对角块）：
    1→2→…→Nτ→Nτ+1→Nτ+3→…→n−1→n→n−2→…→Nτ+2→1
 边块依次 [Wbτ×(Nτ−1), Wbτ(结合), Wbt×Nt, −I(迹), adj(Wbt)×Nt(逆序), +I(环回)]，
 diag 全为 −I。环方向仿射前推解 BB·X = BI（O(n·K²·Limp + K³)）；
 **det(BB) = (−1)^{Kn}·det(I_K − Π)**，Π = 环方向边块有序积 B[n]⋯B[1]。
"""
function _exrt_schur_mixed_cyclic(Wbτ::Matrix{ComplexF64}, Wbt::Matrix{ComplexF64},
                                  Nτ::Int, Nt::Int, BI::Matrix{ComplexF64}, K::Int)
    n = Nτ + 2 * (Nt + 1)
    L = size(BI, 2)
    # 环节点序 + 边块（edge i：v[i] → v[i+1]，v 环回）
    cyc = collect(1:Nτ)
    for jj in 1:(Nt + 1)
        push!(cyc, Nτ + 2jj - 1)                # + 节点（奇段）
    end
    push!(cyc, n)                               # − 末节点（迹角入口）
    for jj in (Nt+1):-1:2
        push!(cyc, Nτ + 2jj - 2)                # − 节点（偶段，逆序）
    end
    @assert cyc[1] == 1 && length(cyc) == n
    edges = Vector{Matrix{ComplexF64}}(undef, n)
    for j in 2:Nτ
        edges[j-1] = Wbτ
    end
    edges[Nτ] = Wbτ                             # 结合 cell（Nτ → Nτ+1）
    for jj in 1:Nt                              # + 段：cyc[Nτ+jj] → cyc[Nτ+jj+1]
        edges[Nτ + jj] = Wbt
    end
    edges[Nτ + Nt + 1] = -Matrix{ComplexF64}(I, K, K)   # 迹角（+ 末 → − 末）
    for j in 1:Nt                                       # − 段（逆序 adj）
        edges[Nτ + Nt + 1 + j] = adjoint(Wbt)
    end
    edges[n] = Matrix{ComplexF64}(I, K, K)           # 环回（Nτ+2 → 1）
    # Y 视图（节点 → BI 块行）
    Yv(v) = view(BI, (0:K-1) .* n .+ v, :)
    # 仿射前推：X[v[i]] = A[i]·P + b[i]，P := X[1]
    A = Vector{Matrix{ComplexF64}}(undef, n)
    b = Vector{Matrix{ComplexF64}}(undef, n)
    A[1] = Matrix{ComplexF64}(I, K, K)
    b[1] = zeros(ComplexF64, K, L)
    for i in 2:n
        A[i] = edges[i-1] * A[i-1]
        b[i] = edges[i-1] * b[i-1] - Matrix{ComplexF64}(Yv(cyc[i]))
    end
    # 一致性（节点 1 方程）：X[1] = edges[n]·X[v[n]] − Y[1]
    P = (Matrix{ComplexF64}(I, K, K) - edges[n] * A[n]) \ (edges[n] * b[n] - Matrix{ComplexF64}(Yv(1)))
    # 回代装配 X（BI 行布局：节点 c 的块行 = (0:K-1)*n .+ c）
    X = Matrix{ComplexF64}(undef, size(BI)...)
    x = P
    X[(0:K-1) .* n .+ cyc[1], :] .= x
    for i in 2:n
        x = edges[i-1] * x - Matrix{ComplexF64}(Yv(cyc[i]))
        X[(0:K-1) .* n .+ cyc[i], :] .= x
    end
    # det(BB) = (−1)^{Kn}·det(I_K − Π)，Π = edges 沿环有序积 B[n]⋯B[1]
    Π = edges[n]
    for i in (n-1):-1:1
        Π = Π * edges[i]
    end
    detBB = (-1.0 + 0.0im)^(K * n) *
            det(Matrix{ComplexF64}(I, K, K) - Π)
    return X, detBB
end
