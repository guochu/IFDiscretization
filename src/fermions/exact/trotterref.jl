# =============================================================================
# trotterref.jl —— Trotter 化精确参照（直接演化，多体矩阵直乘，机器精度）
#
# ← QuAPI/exactif/brute.jl 的 trotter_exact 原样移植。
#
# 两自旋、每自旋独立浴的 Trotter 化全动力学：Strang 链
#   Utot = imp(½)·W·imp·W···imp(½)（与 processtensor.jl 的 IF 同劈分）
# 初态 ρ0 ⊗ ρB(β)，ρ(T) = Utot·ρtot·Utot†，迹浴得 ρ_imp。
# 金标准测试中的"直接演化"参照（test/exact/processtensor.jl）。
# =============================================================================

"""
    trotter_exact(Es, Vs, β, N, Δt; εs=(0.0,0.0), U=0.0, ρ0=nothing)
        -> ρ_imp（4×4，基 |0⟩,|↑⟩,|↓⟩,|↑↓⟩）

两自旋、每自旋独立浴的 Trotter 化全动力学（多体矩阵直乘，机器精度）。
ρ0：4×4 初始杂质密度矩阵（默认 |0⟩⟨0|）。
"""
function trotter_exact(Es, Vs, β, N, Δt; εs=(0.0, 0.0), U=0.0, ρ0=nothing)
    M = length(Es)
    hσ = single_particle_h(Es, Vs)
    # 模序：d↑, bath↑…, d↓, bath↓…
    h = zeros(ComplexF64, 2 * (M + 1), 2 * (M + 1))
    h[1:M+1, 1:M+1] .= hσ
    h[M+2:end, M+2:end] .= hσ
    Wmb = exp(-im * Δt * mb_generator(h))
    # 踢：对角相位 ε_↑n↑ + ε_↓n↓ + U n↑n↓（d↑ = 模 1，d↓ = 模 M+2）
    dim = 1 << (2 * (M + 1))
    Eimp = [εs[1] * ((J >> 0) & 1) + εs[2] * ((J >> (M + 1)) & 1) +
            U * ((J >> 0) & 1) * ((J >> (M + 1)) & 1) for J in 0:dim-1]
    Kfull = diagm(0 => exp.(-im * Δt .* Eimp))
    Khalf = diagm(0 => exp.(-(im * Δt / 2) .* Eimp))
    Utot = Khalf * Wmb * (Kfull * Wmb)^(N - 1) * Khalf
    # 热浴（两自旋独立）
    nb = [1 / (1 + exp(β * Es[k])) for k in 1:M]
    ρB1 = diagm(0 => [prod(((1 - nb[k], nb[k])[(j >> (k - 1)) & 1 + 1]) for k in 1:M)
                      for j in 0:(1<<M)-1])
    ρB = kron(ρB1, ρB1)
    ρ0 === nothing && (ρ0 = zeros(ComplexF64, 4, 4); ρ0[1, 1] = 1.0)
    # 初态：ρ0 ⊗ ρB，嵌入到全模序（d↑,…,d↓,…）：杂质占据 = 位0(d↑) 与位 M+1(d↓)
    ρtot = _embed_impurity_bath(ρ0, ρB, M)
    ρt = Utot * ρtot * Utot'
    return _traceout_bath(ρt, M)
end

# ρ0(4×4) ⊗ ρB(4^M×4^M) → 全基（模序 d↑,bath↑(M),d↓,bath↓(M)）矩阵
function _embed_impurity_bath(ρ0, ρB, M)
    dim = 1 << (2 * (M + 1))
    out = zeros(ComplexF64, dim, dim)
    dimb = 1 << M
    for iup in 0:1, idn in 0:1
        for bup in 0:dimb-1, bdn in 0:dimb-1
            row = iup | (bup << 1) | (idn << (M + 1)) | (bdn << (M + 2))
            for jup in 0:1, jdn in 0:1
                for cup in 0:dimb-1, cdn in 0:dimb-1
                    col = jup | (cup << 1) | (jdn << (M + 1)) | (cdn << (M + 2))
                    out[row+1, col+1] = ρ0[2*iup + idn + 1, 2*jup + jdn + 1] *
                                        ρB[bup*dimb + bdn + 1, cup*dimb + cdn + 1]
                end
            end
        end
    end
    return out
end

function _traceout_bath(ρt, M)
    dimb = 1 << M
    out = zeros(ComplexF64, 4, 4)
    for iup in 0:1, idn in 0:1, jup in 0:1, jdn in 0:1
        s = 0.0im
        for bup in 0:dimb-1, bdn in 0:dimb-1
            row = iup | (bup << 1) | (idn << (M + 1)) | (bdn << (M + 2))
            col = jup | (bup << 1) | (jdn << (M + 1)) | (bdn << (M + 2))
            s += ρt[row+1, col+1]
        end
        out[2*iup + idn + 1, 2*jup + jdn + 1] = s
    end
    return out
end
