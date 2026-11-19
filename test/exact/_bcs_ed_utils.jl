# BCS 金标准共享 ED 工具（1 模浴，16 维 Fock；JW，模式序 d↑=1, c↑=2, d↓=3, c↓=4）
# 出处：QuAPI/exactif/test/exact_bcs_check.jl 的 bcs_ed_h/mb_generator 体系 +
# T1 的轮廓收缩约定（物理时间 Heisenberg + 轮廓位排序 + ρ 前置）。

"""
    mb_generator(h) -> 2ⁿ×2ⁿ

单粒子 h（n 模，JW）的多体生成元（← QuAPI/exactif/brute.jl 原样拷贝）。
"""
function _bcs_mb_generator(h::AbstractMatrix)
    n = size(h, 1)
    dim = 1 << n
    G = zeros(ComplexF64, dim, dim)
    for J in 0:dim-1
        for j in 1:n                       # c_j（湮没，作用在浴/杂质模 j）
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

function _bcs_ann16(k::Int)
    D = zeros(ComplexF64, 16, 16)
    for s in 0:15
        (s >> (k - 1)) & 1 == 1 || continue
        sgn = (-1)^count_ones(s & ((1 << (k - 1)) - 1))
        D[s ⊻ (1 << (k - 1)) + 1, s + 1] = sgn
    end
    return D
end

"1 模 BCS 浴多体 H（16 维）：星形 (d↑,c↑)⊕(d↓,c↓) + Δ(c↑†c↓† + h.c.)"
function _bcs_ed_h(ε::Real, V::Real, Δ::Number)
    h2 = [0.0 V; V ε]
    G = _bcs_mb_generator(h2)
    H = zeros(ComplexF64, 16, 16)
    H += kron(G, Matrix{ComplexF64}(I, 4, 4))
    H += kron(Matrix{ComplexF64}(I, 4, 4), G)
    c2, c4 = _bcs_ann16(2), _bcs_ann16(4)
    H += Δ * (c2' * c4') + conj(Δ) * (c4 * c2)
    return Hermitian(H)
end

"""
    bcs_ed_factory(f, β; gfsolver) -> (Δ; Nt, t, Nτ) -> (ed, opH, ρ)

ED 轮廓收缩机器：`ed(v, w, p, q)` = ⟨T_C Φ_p(v)Φ_q†(w)⟩（Φ = (d↑, d†↓)），
轮廓点 `v = (:τ, k)`（τ_k = (k−1)δτ；k = Nτ+1 表 β⁻ 锚）或 `(br, a)`
（br ∈ (:+, :-)，物理时间：+ 支 a·δt、− 支 (Nt−a)·δt）。约定 = QuAPI T1
（物理时间 Heisenberg：τ 支虚时 e^{+Hτ}·op·e^{−Hτ}、Keldysh 支实时；
轮廓位排序 pos(v) ≥ pos(w) ⟹ tr(ρ·O_v·O_w)，反序反对易翻号）。
`opH(x; p, dag)` 取裸算符的轮廓 Heisenberg 形（β⁻ 锚与环对角用）。
"""
function bcs_ed_factory(f::ImpurityModelBase.AbstractBoundedFunction, β::Real;
                        gfsolver=EDToulouseSolver(1))
    ε1s, V1s, _ = IFD._quad_nodes(f, 0.0, 1)
    nodag = [_bcs_ann16(1), _bcs_ann16(3)']      # Φ₁ = d↑, Φ₂ = d†↓
    dag = [_bcs_ann16(1)', _bcs_ann16(3)]
    return function (Δ::Number; Nt::Int, t::Real, Nτ::Int)
        δt = t / Nt
        δτ = β / Nτ
        H = _bcs_ed_h(ε1s[1], V1s[1], Δ)
        ev = eigen(H)
        ρ = ev.vectors * Diagonal(exp.(-β .* ev.values)) * ev.vectors' /
            sum(exp.(-β .* ev.values))
        Uτ(x) = ev.vectors * Diagonal(exp.(ev.values .* x)) * ev.vectors'
        Ut(x) = ev.vectors * Diagonal(exp.(-im * ev.values .* x)) * ev.vectors'
        function opH(x; p, dag_)
            base = dag_ ? dag[p] : nodag[p]
            if x[1] === :τ
                τv = x[2] == Nτ + 1 ? β : (x[2] - 1) * δτ
                A = Uτ(τv)
                return A * base * inv(A)          # e^{+Hτ}·op·e^{−Hτ}
            else
                tphys = x[1] === :+ ? x[2] * δt : (Nt - x[2]) * δt
                B = Ut(tphys)
                return B \ base * B               # e^{+iHt}·op·e^{−iHt}
            end
        end
        cpos(x) = x[1] === :τ ? x[2] :
                  x[1] === :+ ? Nτ + x[2] + 1 : Nτ + Nt + 2 + (Nt - x[2])
        ed = (v, w, p, q) -> begin
            Ov = opH(v; p=p, dag_=false)
            Ow = opH(w; p=q, dag_=true)
            return cpos(v) >= cpos(w) ? tr(ρ * Ov * Ow) : -tr(ρ * Ow * Ov)
        end
        return ed, opH, ρ
    end
end

# 四 Nambu 块的 (容器, p, q) 槽表（uu=(1,1) 正常↑↑、dd=(2,2) 正常↓↓、
# ud=(1,2) 反常 cc、du=(2,1) 反常 nn）
bcs_blocks(ex) = ((ex.uu, 1, 1), (ex.dd, 2, 2), (ex.ud, 1, 2), (ex.du, 2, 1))
