# _bcs_ed_chain.jl —— BCS 门序列 ED 参照（任意 K 模浴；JW；模式序 d↑=bit0、d↓=bit1、
# P_k=bit(2k)、Q_k=bit(2k+1)，k = 1..K；维数 4·4^K）
#
# 与 IFD.exact_bcs_Δt 的浴约定同构：H = Σ_k [ε_k(n_{P_k} + n_{Q_k}) + Δ(P_k†Q_k† + Q_kP_k)]
#   + Σ_k V_k Σ_{σ=↑,↓}(d_σ† c_{σk} + h.c.)；谱数据 (εs, Vs) 取自 IFD._quad_nodes。
# 调度（GTEMPO native_schedule 同源）：每 gap 先门 D(link_g) = e^{−i·link_g·ε_d(n↑+n↓)} 后 cell
#   C(cell_g) = e^{−i·cell_g·H}（g = 1..N；末 link_{N+1} 为开放端窗）。
#   order 1：links = [δt…δt, 0]；order 2（Strang）：links = [δt/2, δt…, δt/2]。
# 初态 ρ₀ = |00⟩_imp ⊗ ρ_B（BCS 浴热态）。参照侧不做任何谱恒等式加速（全 Fock 矩阵）。

"JW 湮没算符 c_k（n 模，mode k ↔ bit k−1）"
function ed_ann(n::Int, k::Int)
    dim = 1 << n
    D = zeros(ComplexF64, dim, dim)
    for s in 0:dim-1
        (s >> (k - 1)) & 1 == 1 || continue
        D[s ⊻ (1 << (k - 1)) + 1, s + 1] = (-1)^count_ones(s & ((1 << (k - 1)) - 1))
    end
    return D
end

"单模对 (P, Q) 的 BCS 浴哈密顿量 4×4（基矢 |n_P n_Q⟩，bit0 = P、bit1 = Q）与热态"
function ed_bath_mode(ε::Real, Δ::Number, β::Real)
    p, q = ed_ann(2, 1), ed_ann(2, 2)
    Hb = Hermitian(ε * (p' * p + q' * q) + Δ * (p' * q') + conj(Δ) * (q * p))
    ev = eigen(Hb)
    w = exp.(-β .* (ev.values .- minimum(ev.values)))
    ρ = ev.vectors * Diagonal(w) * ev.vectors'
    return Hermitian(ρ ./ tr(ρ))
end

"谱数据 → (εs, Vs)（与 exact_bcs_Δt 同一离散：_quad_nodes + 死模剔除）"
function ed_bath_modes(f, gfsolver)
    Es, Vsq, wΓ = IFD._quad_nodes(f, 0.0, IFD._nbath(gfsolver))
    keep = wΓ .> 0
    return Es[keep], Vsq[keep]
end

"""
    bcs_ed_chain(εs, Vs, Δ, β, cells, links, εd) -> (Z, n̄↑, n̄↓, ⟨d↓d↑⟩, ⟨d†↑d†↓⟩)

门序列 ED（直接离散链的迹）。`cells` = N 个 cell 时长、`links` = N+1 个门时长。
"""
function bcs_ed_chain(εs::AbstractVector, Vs::AbstractVector, Δ::Number, β::Real,
                      cells::AbstractVector, links::AbstractVector, εd::Real)
    K = length(εs)
    length(Vs) == K || throw(DimensionMismatch("εs/Vs 长度不等"))
    length(links) == length(cells) + 1 || throw(DimensionMismatch("links 须比 cells 多一（末位开放端窗）"))
    nm = 2 + 2K                         # d↑, d↓, (P_k, Q_k)_k
    dup, ddn = ed_ann(nm, 1), ed_ann(nm, 2)
    H = zeros(ComplexF64, 1 << nm, 1 << nm)
    ρb = ones(ComplexF64, 1, 1)
    for k in 1:K
        Pk, Qk = ed_ann(nm, 2k + 1), ed_ann(nm, 2k + 2)        # mode 号 = bit + 1
        H .+= Vs[k] .* (dup' * Pk + Pk' * dup + ddn' * Qk + Qk' * ddn)
        H .+= εs[k] .* (Pk' * Pk + Qk' * Qk)
        H .+= Δ .* (Pk' * Qk') .+ conj(Δ) .* (Qk * Pk)
        ρb = kron(ed_bath_mode(εs[k], Δ, β), ρb)       # 浴模式 1 = 最低浴位（bit2）
    end
    H = Hermitian(H)
    ρ = kron(ρb, ComplexF64[1 0; 0 0], ComplexF64[1 0; 0 0])   # 杂质真空（bit1、bit0）
    U = Matrix{ComplexF64}(I, 1 << nm, 1 << nm)
    for g in eachindex(links)
        Dg = Diagonal([exp(-im * links[g] * εd * (((s >> 0) & 1) + ((s >> 1) & 1))) for s in 0:(1 << nm)-1])
        C = g > length(cells) || cells[g] == 0 ? Matrix{ComplexF64}(I, 1 << nm, 1 << nm) : exp(-1im * cells[g] .* H)
        U = C * Dg * U                          # 时间序：D_g 先、C_g 后（后作用者左乘）
    end
    R = U * ρ * U'
    Z = tr(R)
    nu = tr(dup' * dup * R) / Z
    nd = tr(ddn' * ddn * R) / Z
    a1 = tr(ddn * dup * R) / Z                   # ⟨d↓ d↑⟩
    a2 = tr(dup' * ddn' * R) / Z                 # ⟨d†↑ d†↓⟩
    return Z, real(nu), real(nd), a1, a2
end

"order ∈ (1, 2) 的 native 调度：返回 (cells, links)"
function ed_native_schedule(order::Int, N::Int, δt::Real)
    N >= 1 || throw(ArgumentError("N ≥ 1"))
    if order == 1
        return fill(Float64(δt), N), vcat(fill(Float64(δt), N), 0.0)
    elseif order == 2
        return fill(Float64(δt), N), vcat(δt / 2, fill(Float64(δt), N - 1), δt / 2)
    end
    throw(ArgumentError("order 须为 1 或 2"))
end
