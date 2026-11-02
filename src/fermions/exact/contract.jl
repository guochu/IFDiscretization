# =============================================================================
# contract.jl —— 核 → 占据路径值 → 杂质密度矩阵（端到端收缩，任意 N 严格）
#
# ← QuAPI/exactif/contract.jl 原样移植（runtests L5 金标准已验证）。
#
# 路径值求值：精确过程张量（processtensor.jl 的 Ivals——严格离散 IF 本身，
# 机器精度；Hafnian 字典等研究对照路线未移植，见 QuAPI/exactif）。
#
# 收缩：对角 kick（H₁ = ε_σn_σ + Un_↑n_↓）强制每支轨迹 v_0..v_N
#（内步 α_m = β_m，两端半步）。ρ_imp 元素 = Σ_traj I×I×kick 相位×ρ₀。
# 与 trotter_exact（同劈分直接演化，trotterref.jl）逐位对照：
# N=1..4、任意 εs、U=0/U≠0 机器精度（QuAPI/exactif runtests L5；
# IFD 的金标准测试 test/exact/processtensor.jl 同款断言）。
# =============================================================================
using LinearAlgebra: I

"""
    contract_impurity(Ivals; N, Δt, εs=(0.0,0.0), U=0.0, ρ0=nothing) -> 4×4 ρ_imp(NΔt)

精确张量收缩（任意 N 严格）。`Ivals`：单自旋过程张量
（processtensor.jl 的 `if_path_values`）。
基 |0⟩,|↑⟩,|↓⟩,|↑↓⟩（索引 2n↑+n↓+1）；ρ0 缺省 |0⟩⟨0|。
"""
function contract_impurity(Ivals::AbstractVector; N::Int, Δt::Real,
                           εs=(0.0, 0.0), U=0.0, ρ0=nothing)
    ρ0 === nothing && (ρ0 = zeros(ComplexF64, 4, 4); ρ0[1, 1] = 1.0)
    out = zeros(ComplexF64, 4, 4)
    L = N + 1
    # 轨迹 → 顶点掩码（(S39)：α_n⁺=4n+1→bit 4n，β_{n+1}⁺=4n+3→bit 4n+2，
    # α_n⁻=4n+2→bit 4n+1，β_{n+1}⁻=4n+4→bit 4n+3）
    function supp(trk, trb)
        s = 0
        for n in 0:N-1
            trk[n+1] == 1 && (s |= 1 << (4n))
            trb[n+1] == 1 && (s |= 1 << (4n + 1))
        end
        for m in 1:N
            trk[m+1] == 1 && (s |= 1 << (4m - 2))
            trb[m+1] == 1 && (s |= 1 << (4m - 1))
        end
        return s
    end
    kickE(vu, vd) = εs[1] * vu + εs[2] * vd + U * vu * vd
    for path in 0:(1 << (4 * L)) - 1
        ku = [(path >> i) & 1 for i in 0:L-1]
        kd = [(path >> (L + i)) & 1 for i in 0:L-1]
        bu = [(path >> (2L + i)) & 1 for i in 0:L-1]
        bd = [(path >> (3L + i)) & 1 for i in 0:L-1]
        w = Ivals[supp(ku, bu) + 1] * Ivals[supp(kd, bd) + 1]
        w == 0 && continue
        ph = 1.0 + 0.0im
        for m in 0:N
            fac = (m == 0 || m == N) ? Δt / 2 : Δt
            ph *= exp(-im * fac * kickE(ku[m+1], kd[m+1]))
            ph *= exp(im * fac * kickE(bu[m+1], bd[m+1]))
        end
        out[2ku[end]+kd[end]+1, 2bu[end]+bd[end]+1] +=
            w * ph * ρ0[2ku[1]+kd[1]+1, 2bu[1]+bd[1]+1]
    end
    return out
end
