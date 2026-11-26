# =============================================================================
# _worldline.jl —— 严格核消费的世界线（占据数）求值器（金标准测试专用 helper）
#
# 语义 = GTEMPO 严格核消费层（sysdynamics_exact + PartialIF 消费）的紧凑翻译，逐因子经
# GTEMPO 本体 oracle dump 标定（2026-11-25）：每站 (节点, 支, 带) 持 (a = plain, ā = conj)
# 一对腿；门/反测度/折返/核都是因子 1 + c·θ_i∧θ_j；ρ₀ 占据投影是纯单项式。
#
# 求值恒等式（均经暴力 Grassmann 展开数值标定，比值全 +1）：
#   Tr_B[e^{½θᵀMθ}] = Pf(M + Ω)                       （Ω = 站内 (a,ā) 配对形式）
#   Tr_B[P·θ_{l1}∧…∧θ_{l2m}] = ±Pf((M+Ω)^{(l1,…,l2m)})（去掉这些腿的余子式；符号 = 排到升序的置换奇偶）
# 其中 Tr_B = 逐站 Berezin 迹（vac/pair 权重 +1，odd 扇区 0）= GM.Zvalue 的站级语义。
#
# 消费语义（粒子字典 = 冻结核语义）：带 = 物理粒子，ρ₀ 真空（或占据投影），门 x = e^{−i·link·ε_d}；
# BCS exact_bcs_Δt 的四块按键放 (ā↑,a↑) / (ā↓,a↓) / (ā↑,ā↓) / (a↓,a↑)
#（= GTEMPO AnomalousPairing 四键消费，见 bcs_factors_particle 头注）。
# =============================================================================

"标准定义 Pfaffian（包内例程 = (−1)^{L/2}·标准定义）"
pf_std(A::Matrix{ComplexF64}) = IFD._formb_pfaffian(A) * (-1.0)^(size(A, 1) ÷ 2)

"因子表 → 反对称矩阵 M（规范序 = 变量号升序）。factors = [(i, j, c)]：因子 1 + c·θ_i∧θ_j"
function wl_matrix(L::Int, factors)
    M = zeros(ComplexF64, L, L)
    for (i, j, c) in factors
        i == j && error("same-leg factor")
        lo, hi = minmax(i, j)
        v = (i < j) ? c : -c
        M[lo, hi] += v
        M[hi, lo] -= v
    end
    return M
end
"站内配对形式 Ω（站 s = 变量 2s−1, 2s）：Ω_{2s−1,2s} = +1"
function wl_omega(L::Int)
    Ω = zeros(ComplexF64, L, L)
    for s in 1:L÷2
        Ω[2s-1, 2s] = 1; Ω[2s, 2s-1] = -1
    end
    return Ω
end
"Tr_B[e^{½θᵀMθ}] = Pf(M + Ω)"
wl_Z(M::Matrix{ComplexF64}) = pf_std(M + wl_omega(size(M, 1)))
"去掉 legs 的余子式 Pfaffian"
function wl_cof(M::Matrix{ComplexF64}, legs::Int...)
    nst = size(M, 1)
    Nm = M + wl_omega(nst)
    keep = [k for k in 1:nst if !(k in legs)]
    return pf_std(Nm[keep, keep])
end
"按乘积序插入：Tr_B[P·θ_{l1}∧…∧θ_{l2m}]（重复腿 ⟹ 0）"
function wl_readout(M::Matrix{ComplexF64}, legs::Int...)
    ls = collect(legs)
    allunique(ls) || return zero(ComplexF64)
    p = sortperm(ls)
    inv = sum(Int(p[a] > p[b]) for a in eachindex(p) for b in a+1:length(p))
    return (-1.0)^inv * wl_cof(M, ls[p]...)
end

# ---------------- 费米（单带）：站 s(n, br) = 2(n−1) + (br==:+ ? 1 : 2) ----------------
f_plain(n, br) = 2 * (2 * (n - 1) + (br === :+ ? 1 : 2)) - 1
f_conj(n, br) = 2 * (2 * (n - 1) + (br === :+ ? 1 : 2))
"native 调度：门 (1 + x·ā_{g+1}∧a_g)、反测度 1 + ā_p∧a_p（节点 2..Nt−1）、折返 1 + ā⁻_{Nt}∧a⁺_{Nt}、
核四块（跨支取负）放 (ā_i, a_j)；ρ₀ 真空（占据投影走 wl_readout 的两条腿）"
function fermi_factors(kernel, N::Int, links, εd::Real)
    Nt = N + 2
    F = Tuple{Int,Int,ComplexF64}[]
    for g in 1:(Nt - 1)
        x = exp(-im * links[g] * εd)
        push!(F, (f_conj(g + 1, :+), f_plain(g, :+), x))
        push!(F, (f_conj(g, :-), f_plain(g + 1, :-), conj(x)))
    end
    for p in 2:(Nt - 1), br in (:+, :-)
        push!(F, (f_conj(p, br), f_plain(p, br), 1.0 + 0.0im))
    end
    push!(F, (f_conj(Nt, :-), f_plain(Nt, :+), 1.0 + 0.0im))
    for b1 in (:+, :-), b2 in (:+, :-)
        blk = IFD.branch(kernel; b1=b1, b2=b2)
        s = b1 === b2 ? 1 : -1
        for i in 1:N+1, j in 1:N+1
            v = s * blk[i, j]
            abs(v) < 1e-16 && continue
            push!(F, (f_conj(i, b1), f_plain(j, b2), v))
        end
    end
    return F
end
"""
    fermi_worldline(kernel, nd, N, links, εd; occupied=false) -> (Z, n̄)

费米 uniform 核的世界线端到端：Z = nd·Tr_B[P·ρ₀]；ρ₀ = 真空（常数）或占据投影 a⁻_1∧ā⁺_1
（GTEMPO `_fockop_factor` 的给定序）；n̄ = −Tr_B[P·ρ₀·(a⁺_{Nt}∧ā⁻_{Nt})]/Tr_B[P·ρ₀]
（strict_occupation 的 −⟨η⁺@Nt·η̄⁻@Nt⟩ 约定）。
"""
function fermi_worldline(kernel, nd::Number, N::Int, links, εd::Real; occupied::Bool=false)
    Nt = N + 2
    M = wl_matrix(4 * (N + 2), fermi_factors(kernel, N, links, εd))
    r0 = occupied ? (f_plain(1, :-), f_conj(1, :+)) : ()
    tr = isempty(r0) ? wl_Z(M) : wl_readout(M, r0...)
    Z = nd * tr
    n = -wl_readout(M, r0..., f_plain(Nt, :+), f_conj(Nt, :-)) / tr
    return Z, real(n)
end

# ---------------- BCS（两带）：站 s(n, br, b) = 4(n−1) + 2·(br==:−) + b；变量 plain = 2s−1、conj = 2s ----------------
bcs_plain(n, br, b) = (2 * (n - 1) + (br === :+ ? 0 : 1)) * 4 + (b - 1) * 2 + 1
bcs_conj(n, br, b) = (2 * (n - 1) + (br === :+ ? 0 : 1)) * 4 + (b - 1) * 2 + 2

"冻结核字面放置语义装配（bcscorr.jl 指数四项 = GTEMPO AnomalousPairing 四键消费 =
exact_bcs_Δt 的契约）：两带都是物理粒子（ρ₀ 真空、门 x = e^{−i·link·ε_d}），跨支取负，
uu ↔ (ā↑,a↑)、dd ↔ (ā↓,a↓)、ud ↔ (ā↑,ā↓)（对产生）、du ↔ (a↓,a↑)（对湮灭）。"
function bcs_factors_particle(kern, N::Int, links, εd::Real)
    Nt = N + 2
    F = Tuple{Int,Int,ComplexF64}[]
    for g in 1:(Nt - 1)
        x = exp(-im * links[g] * εd)
        for b in 1:2
            push!(F, (bcs_conj(g + 1, :+, b), bcs_plain(g, :+, b), x))
            push!(F, (bcs_conj(g, :-, b), bcs_plain(g + 1, :-, b), conj(x)))
        end
    end
    for p in 2:(Nt - 1), br in (:+, :-), b in 1:2
        push!(F, (bcs_conj(p, br, b), bcs_plain(p, br, b), 1.0 + 0.0im))
    end
    for b in 1:2
        push!(F, (bcs_conj(Nt, :-, b), bcs_plain(Nt, :+, b), 1.0 + 0.0im))
    end
    blocks = ((kern.uu, :conj, 1, :plain, 1), (kern.dd, :conj, 2, :plain, 2),
              (kern.ud, :conj, 1, :conj, 2), (kern.du, :plain, 2, :plain, 1))
    for b1 in (:+, :-), b2 in (:+, :-)
        s = b1 === b2 ? 1 : -1
        for (blk, k1, f1, k2, f2) in blocks
            m = IFD.branch(blk; b1=b1, b2=b2)
            for i in 1:N+1, j in 1:N+1
                v = s * m[i, j]
                abs(v) < 1e-16 && continue
                li = k1 === :conj ? bcs_conj(i, b1, f1) : bcs_plain(i, b1, f1)
                lj = k2 === :conj ? bcs_conj(j, b2, f2) : bcs_plain(j, b2, f2)
                push!(F, (li, lj, v))
            end
        end
    end
    return F
end

"""
    bcs_worldline_particle_full(kern, N, links, εd) -> (tr, n̄↑, n̄↓, ⟨d↓d↑⟩, ⟨d†↑d†↓⟩)

exact_bcs_Δt 的端到端消费（冻结核字面放置语义）：tr = 轮廓收缩（Z = normdelta·tr）；
n̄_b = −⟨a_b⁺@Nt ∧ ā_b⁻@Nt⟩（折返点 Nt：bra 侧 − 支 conj、ket 侧 + 支 plain）；
反常 ⟨d↓d↑⟩ = ⟨a↓⁺ ∧ a↑⁺⟩、⟨d†↑d†↓⟩ = ⟨ā↑⁻ ∧ ā↓⁻⟩（符号由 ED 标定，2026-11-26）。
"""
function bcs_worldline_particle_full(kern, N::Int, links, εd::Real)
    Nt = N + 2
    M = wl_matrix(8 * (N + 2), bcs_factors_particle(kern, N, links, εd))
    tr = wl_Z(M)
    rd(i, j) = wl_readout(M, i, j) / tr
    nup = real(-rd(bcs_plain(Nt, :+, 1), bcs_conj(Nt, :-, 1)))
    ndn = real(-rd(bcs_plain(Nt, :+, 2), bcs_conj(Nt, :-, 2)))
    a1 = rd(bcs_plain(Nt, :+, 2), bcs_plain(Nt, :+, 1))         # ⟨d↓d↑⟩
    a2 = rd(bcs_conj(Nt, :-, 1), bcs_conj(Nt, :-, 2))           # ⟨d†↑d†↓⟩
    return tr, nup, ndn, a1, a2
end

"exact_bcs_Δt 的端到端消费（带归一）：(Z, n̄↑, n̄↓, ⟨d↓d↑⟩, ⟨d†↑d†↓⟩)，Z = nd·tr"
function bcs_worldline_particle(kern, nd::Number, N::Int, links, εd::Real)
    tr, nup, ndn, a1, a2 = bcs_worldline_particle_full(kern, N, links, εd)
    return nd * tr, nup, ndn, a1, a2
end
