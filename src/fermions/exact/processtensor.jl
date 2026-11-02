# =============================================================================
# processtensor.jl —— 严格离散影响泛函：过程张量（Ng–Reichman 2023 实时路线）
#
# ← QuAPI/exactif/brute.jl 原样移植（已验证机器：runtests L0/L5/L7 金标准）。
#
# 【语义】有限热库多体暴力参照（定义"真值"）。单自旋扇区（两自旋 Anderson
# 的 IF 按自旋因子化，浴完全相同）。
#
# Trotter 链（Strang，杂质半步在最外）：
#   Û_tot = imp(½)·W(Δt)·imp(Δt)·W(Δt)···imp(Δt)·W(Δt)·imp(½)
# 其中 W = exp(−iΔtĤ₂)，Ĥ₂ = Σ_k E_k ĉ_k†ĉ_k + Σ_k(V_k d̂†ĉ_k + h.c.)
#（杂质 onsite ε 放进 H₁，即核的单粒子矩阵取 ε=0 —— 论文默认分割：
#   IF 不依赖杂质哈密顿量 Ĥ₁ 的细节，PRB 107, 125103 正文 §III.B）。
# 顶点布局（每分支每步 2 个顶点，共 4N 个 Grassmann 变量，式 (S39)）：
#   正向（ket）支：α_n⁺ = s⁺_{n+}（时刻 nΔt，踢前）  n=0..N−1
#                  β_m⁺ = s⁺_{m−}（时刻 mΔt，踢后）  m=1..N
#   反向（bra）支：α_n⁻, β_m⁻ 同上。
#   W(Δt) 连接 α_n → β_{n+1}；杂质演化连接 β_m → α_m（内步全步，
#   两端 s_0 → α_0 与 β_N → s_N 为半步）。
#
# 路径值（IF 的计算基定义，无任何 Grassmann/相干态假设）：
#   I[S] = Tr_B[ 𝒲⁺(S⁺) · ρ_B · 𝒲⁻(S⁻)† ]，
#   𝒲⁺ = X[β_N⁺, α_{N−1}⁺]·X[β_{N−1}⁺, α_{N−2}⁺]···X[β_1⁺, α_0⁺]  （晚者居左）
#   𝒲⁻ = X[β_1⁻, α_0⁻]·X[β_2⁻, α_1⁻]···X[β_N⁻, α_{N−1}⁻]        （早者居左）
#   X[s,s'] ≡ ⟨s|Ŵ(Δt)|s'⟩（杂质占据态矩阵元，2^M×2^M 浴算符）。
#
# 【消费契约】Ivals 交 contract_impurity 做对角 kick 收缩（ε_d/U 只在
# 收缩侧进入）——金标准：收缩 ρ_imp 与同劈分直接演化 trotter_exact 逐位
# 一致（test/exact/processtensor.jl）。
#
# 【复杂度】I[{s}] 共 2^{4N} 个值（每值一次 2^M 维 bath 迹），N ≤ 5 实用。
# =============================================================================
using LinearAlgebra: eigen, exp, diagm, Hermitian, I, tr, det

# ------------------------------ 多体机构 ------------------------------ #

"""
    mb_generator(h) -> Matrix

单粒子矩阵 h（n×n）对应的多体生成元 G = Σ_ij h_ij f̂_i† f̂_j（2^n×2^n）。
基：占据数基，模 1 = 位 1（Jordan–Wigner 序）。

Jordan–Wigner 约定：f̂_j = s_j a_j（a_j = 硬核玻色湮灭，s_j = 前 j−1 位的奇偶），
f̂_i†f̂_j 作用在 |J⟩ 上的符号 = (−1)^(sj+si)。
"""
function mb_generator(h::AbstractMatrix)
    n = size(h, 1)
    dim = 1 << n
    G = zeros(ComplexF64, dim, dim)
    for J in 0:dim-1
        for j in 1:n                       # c_j（湮灭，作用在浴/杂质模 j）
            (J >> (j - 1)) & 1 == 0 && continue
            sj = count_ones(J & ((1 << (j - 1)) - 1))   # c_j 的 JW 符号
            Jj = J ⊻ (1 << (j - 1))
            for i in 1:n                   # c_i†
                (Jj >> (i - 1)) & 1 == 1 && continue
                si = count_ones(Jj & ((1 << (i - 1)) - 1))  # c_i† 的 JW 符号
                Istate = Jj ⊻ (1 << (i - 1))
                h[i, j] == 0 && continue
                G[Istate+1, J+1] += (-1)^(sj + si) * h[i, j]
            end
        end
    end
    return G
end

"""
    single_particle_h(Es, Vs) -> (1+M)×(1+M)

杂质（位 1，ε=0）+ 浴；h[1,1+k] = V_k，h[1+k,1] = conj(V_k)，h[1+k,1+k] = E_k δ。
"""
function single_particle_h(Es::AbstractVector{<:Real}, Vs::AbstractVector{<:Number})
    M = length(Es)
    h = zeros(ComplexF64, M + 1, M + 1)
    for k in 1:M
        h[1, 1+k] = Vs[k]
        h[1+k, 1] = conj(Vs[k])
        h[1+k, 1+k] = Es[k]
    end
    return h
end

"""
    impurity_blocks(Wmb, M) -> X（(s,sp) → 2^M×2^M 矩阵的字典）

X[s,s'] = ⟨s|Ŵ_mb|s'⟩：把 Wmb 按杂质位（位 1）分块。
"""
function impurity_blocks(Wmb::AbstractMatrix, M::Int)
    dimb = 1 << M
    X = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    for s in 0:1, sp in 0:1
        blk = zeros(ComplexF64, dimb, dimb)
        for b1 in 0:dimb-1, b2 in 0:dimb-1
            # 模序：mode 1 = 杂质（bit 0），modes 2..M+1 = 浴（bits 1..M）
            blk[b1+1, b2+1] = Wmb[((b1 << 1) | s) + 1, ((b2 << 1) | sp) + 1]
        end
        X[(s, sp)] = blk
    end
    return X
end

# ------------------------------ 过程张量 / 路径值 ------------------------------ #

"""
    if_full_tensor(Es, Vs, β, N, Δt) -> NamedTuple

**完整过程张量**：I[{s}]（每顶点一个二值 s ∈ {0,1}，共 4N 个，
(S39) 顶点序）。I[{s}] = Tr_B[𝒲⁺({s⁺})·ρ_B·𝒲̃⁻({s⁻})]，
其中每支的 W 链因子每元胞一项：X[β_m^·, α_{m−1}^·]（β 作行、α 作列）。
矩阵元视角：I[{s}] = ⟨s⁻ 轮廓|𝒯|s⁺ 轮廓⟩，𝒯 是 2N 个轮廓模上的算符。

顶点位（同 (S39)）：β_m⁺ 在 4m−1，α_{m−1}⁺ 在 4m−3；
β_m⁻ 在 4m，α_{m−1}⁻ 在 4m−2。
"""
function if_full_tensor(Es, Vs, β, N, Δt)
    M = length(Es)
    h = single_particle_h(Es, Vs)
    Wmb = exp(-im * Δt * mb_generator(h))
    X = impurity_blocks(Wmb, M)
    nb = [1 / (1 + exp(β * Es[k])) for k in 1:M]
    ρB = diagm(0 => [prod((1 - nb[k], nb[k])[(j >> (k - 1)) & 1 + 1] for k in 1:M)
                     for j in 0:(1<<M)-1])
    nvert = 4N
    val(S, p) = (S >> (p - 1)) & 1
    Ivals = Vector{ComplexF64}(undef, 1 << nvert)
    for S in 0:(1 << nvert) - 1
        # ket 支：𝒲⁺ = X[β_N⁺,α_{N−1}⁺]···X[β₂⁺,α₁⁺]·X[β₁⁺,α₀⁺]（最晚居左）
        Wp = Matrix{ComplexF64}(I, 1 << M, 1 << M)
        for m in 1:N
            Wp = X[(val(S, 4m - 1), val(S, 4m - 3))] * Wp
        end
        # bra 支：𝒲̃⁻ = X[β₁⁻,α₀⁻]†·X[β₂⁻,α₁⁻]†···（最早居左）
        Wm = Matrix{ComplexF64}(I, 1 << M, 1 << M)
        for m in 1:N
            Wm = Wm * X[(val(S, 4m), val(S, 4m - 2))]'
        end
        Ivals[S+1] = tr(Wp * ρB * Wm)
    end
    return (; Ivals, X, ρB, Wmb, nvert)
end

"""
    if_path_values(Es, Vs, β, N, Δt; return_more=false) -> NamedTuple

过程张量 + 对角配对核提取：`c` = I[∅]，`Θ`[u,v] = I[{u,v}]/c
（4N×4N 反对称配对核——Gaussian 收缩与逐对占据值的完备数据）。
"""
function if_path_values(Es, Vs, β, N, Δt; return_more::Bool=false)
    full = if_full_tensor(Es, Vs, β, N, Δt)
    Ivals = full.Ivals
    nvert = 4N
    c = Ivals[1]
    Θ = zeros(ComplexF64, nvert, nvert)
    for u in 1:nvert, v in u+1:nvert
        Θ[u, v] = Ivals[((1 << (u - 1)) | (1 << (v - 1))) + 1] / c
        Θ[v, u] = -Θ[u, v]
    end
    out = (; Ivals, c, Θ)
    return_more && (out = merge(out, (; full)))
    return out
end
