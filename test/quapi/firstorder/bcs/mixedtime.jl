# =============================================================================
# test/quapi/firstorder/bcs/mixedtime.jl —— bcs_Δm 朴素混合装配（bcs_mixed_naive）⇒ 全局一阶
#
# 求解器快照拷贝自 performance/firstorder/bcs/mixedtime.jl（main 与 PROGRAM_FILE 守卫已删，其余逐字未动；
# 相对 include 路径在 test/quapi 镜像树内原样解析）。外包 module 恢复
# performance「每脚本独立进程」的命名隔离，使本文件自包含、可任意顺序
# include；断言只经模块限定名访问求解器。与原件同步是人工责任——基准数值
# 由 archive_regression.jl 运行原件逐字节兜底，拷贝漂移则阶数断言自暴露。
# =============================================================================
module FirstOrderBCSMixedtime
# =============================================================================
# firstorder/bcs/mixedtime.jl —— BCS 浴混合轮廓：一阶冻结核（naive）vs 严格
# 1/2 阶 Trotter。对照：严格解（全平衡 G₁>(t) = [S(t)M₀]₁₁）。
# 输出 results/mixedtime.csv
#
# 与 secondorder/bcs/mixedtime.jl 同骨架、同核数据（单次 bcs_Δm 调用 + 原始
# uv/vu 加权 Δm 块），但全部 K 侧要素退回 naive：
#   ① τ 支 4 分量 Nambu 环：ξ 块（branch(τ,τ)）朴素矩形权重（无梯形 0.5、
#      无跳变常数代入）；跳变源只在 Nambu 对角分量（imagtime.jl 同构）；
#   ② γ 初关联收缩：梯形 + 跳变延拓端点，无消峰分拆（一阶病态来源）；
#   ③ 实时 march：face-value 递推（无半步 f、无接触对角隐式、无梯形）：
#      X_{m+1} = P·(X_m − Σ_{k≤m−1} 𝒦(m+1,k+1)·X_k + [C] h·S_m)，
#      正常槽位 −(P+X)（Δm 约定）、反常槽位 Ŵ_uv − conj(Ŵ_vu)（原始块）；
#   ④ C 通道源：矩形 Duhamel 卷积 S_k(m) = −i·h·Σ_{j<m} E_D(m−j)𝒱_k E_b(j)·γ_k
#      （E_D = diag(e^{∓iϵ_d··})，E_b = e^{−i𝓗_bk t}，𝒱_k = diag(V_k,−V_k)）；
#   ⑤ 组装：G₁>(t_m) = M11·𝒜₁₁ + F0·𝒜₁₂ + C₁（M11 = 1+𝒢₁₁[Nτ]、F0 = 𝒢₂₁[Nτ]，
#      跳变反解）。
#
# 运行：JULIA_PROBE_LIBSTDCXX=0 julia --startup-file=no --project=. firstorder/bcs/mixedtime.jl
# =============================================================================
include(joinpath(@__DIR__, "..", "..", "secondorder", "bcs", "common.jl"))

"""
    bcs_mixed_naive(mdl, Nτ, Nt, T) -> G₁>[t_0..t_Nt]

BCS 混合一阶冻结核（naive 装配）。`G[m+1] ≈ G₁>(t_m)`（不含 −i）。
"""
function bcs_mixed_naive(mdl, Nτ, Nt, T)
    h = T / Nt
    δτ = mdl.β / Nτ
    dm = bcs_Δm(mdl.bath.spectrum; β=mdl.β, Nτ=Nτ, t=T, Nt=Nt, μ=mdl.μ, Δ=mdl.Δ)
    # ① τ 环（4 分量 Nambu，朴素矩形；数据取自单次 bcs_Δm 的 ξ 块）
    J = Matrix{Vector{ComplexF64}}(undef, 2, 2)
    K = Matrix{Vector{ComplexF64}}(undef, 2, 2)
    for (p, q, blk) in ((1, 1, dm.uu), (1, 2, dm.ud), (2, 1, dm.du), (2, 2, dm.dd))
        ξ = branch(blk; b1=:τ, b2=:τ)
        J[p, q], K[p, q] = ξ.ηⱼₖ, ξ.ηₖⱼ
    end
    uτ = [exp(-mdl.ϵd * δτ), exp(mdl.ϵd * δτ)]
    fτ = [exp(-mdl.ϵd * δτ / 2), exp(mdl.ϵd * δτ / 2)]
    M = zeros(ComplexF64, 4Nτ, 4Nτ)
    b = zeros(ComplexF64, 4Nτ)
    idx(i, j) = 2*(i-1) + j
    for m in 1:Nτ, i in 1:2, j in 1:2
        s = 4*(m-1) + idx(i, j)
        M[s, s] += 1.0
        if m ≥ 2
            M[s, 4*(m-2)+idx(i, j)] -= uτ[i]
        else
            M[s, 4*(Nτ-1)+idx(i, j)] += uτ[i]
            i == j && (b[s] = -uτ[i])
        end
        for k in 1:Nτ
            for q in 1:2                              # 𝚺̂ 反常槽位带 −1
                w = k < m ? J[i, q][m-k+1] : k > m ? K[i, q][k-m+1] :
                    J[i, q][1] + K[i, q][1]
                sgn = i == q ? 1.0 : -1.0
                M[s, 4*(k-1)+idx(q, j)] -= fτ[i] * sgn * w
            end
        end
    end
    x = M \ b
    𝒢 = [ComplexF64[x[4*(m-1)+idx(p, q)] for p in 1:2, q in 1:2] for m in 1:Nτ]
    M11 = 1 + 𝒢[Nτ][1, 1]
    F0 = 𝒢[Nτ][2, 1]
    # ② γ 初关联收缩（族 (1,1)；梯形 + 跳变端点延拓，无消峰）
    G0v = ComplexF64[-1-𝒢[Nτ][1, 1]  -𝒢[Nτ][1, 2];
                     -𝒢[Nτ][2, 1]    -1-𝒢[Nτ][2, 2]]
    γ = zeros(ComplexF64, 2, mdl.L)
    for k in 1:mdl.L
        ε, Δ, V = mdl.ws[k], mdl.Δ, mdl.Vs[k]
        Mk = ComplexF64[-ε -Δ; -conj(Δ) ε]
        PN = exp(mdl.β * Mk)
        s0 = V .* [G0v[1, 1], -G0v[2, 1]]
        sj = [V .* [𝒢[m][1, 1], -𝒢[m][2, 1]] for m in 1:Nτ]
        acc = zeros(ComplexF64, 2)
        Q0 = PN
        for m in 1:Nτ
            Q1 = exp(Mk * (mdl.β - m * δτ))
            acc += (δτ / 2) .* (Q0 * (m == 1 ? s0 : sj[m-1]) + Q1 * sj[m])
            Q0 = Q1
        end
        γ[:, k] .= -((Matrix{ComplexF64}(I, 2, 2) + PN) \ acc)
    end
    # ③ 实时核（Δm 约定）：正常 −(P+X)；反常用原始 uv/vu 加权 Δm 块
    #    Ŵ = X+P、k₁₂ = Ŵ_uv − conj(Ŵ_vu)（混合轮廓必需——组合块 cc/nn 的
    #    naive march 配对不收敛，见 secondorder/bcs/mixedtime.jl 机制说明）
    P1 = branch(dm.uu; b1=:+, b2=:+); X1 = branch(dm.uu; b1=:+, b2=:-)
    P2 = branch(dm.dd; b1=:+, b2=:+); X2 = branch(dm.dd; b1=:+, b2=:-)
    fu2, fuv, fvu, fv2 = IFD.get_all_fs(mdl.bath.spectrum, mdl.Δ)
    disp = ϵ -> sqrt(ϵ^2 + abs2(mdl.Δ))
    uv_m = fermionic_Δm(fuv, disp; β=mdl.β, Nτ=Nτ, t=T, Nt=Nt, μ=mdl.μ)
    vu_m = fermionic_Δm(fvu, disp; β=mdl.β, Nτ=Nτ, t=T, Nt=Nt, μ=mdl.μ)
    Puv = branch(uv_m; b1=:+, b2=:+); Xuv = branch(uv_m; b1=:+, b2=:-)
    Pvu = branch(vu_m; b1=:+, b2=:+); Xvu = branch(vu_m; b1=:+, b2=:-)
    K22(j, k) = [-(X1[j, k] + P1[j, k])   (Xuv[j, k] + Puv[j, k]) - conj(Xvu[j, k] + Pvu[j, k]);
                 (Xvu[j, k] + Pvu[j, k]) - conj(Xuv[j, k] + Puv[j, k])   -(X2[j, k] + P2[j, k])]
    Pph = Diagonal([exp(-im * mdl.ϵd * h), exp(im * mdl.ϵd * h)])
    # ④ A / C 通道（face-value 递推；C 带逐 cell 矩形 Duhamel 源）
    #    cellSrc(m) = −i·h·Σ_k 𝒱_k·e^{−i𝓗_bk t_m}·γ_k（bath 相位冻结在 cell 左端点，
    #    −i 因子同 secondorder Duhamel 约定；E_D 传播由 Pph 承担）
    A = [zeros(ComplexF64, 2, 2) for _ in 1:Nt+1]
    C = [zeros(ComplexF64, 2) for _ in 1:Nt+1]
    A[1] = Matrix{ComplexF64}(I, 2, 2)
    Ebs = [Matrix{ComplexF64}(I, 2, 2) for _ in 1:mdl.L]    # e^{−i𝓗_bk t_m}
    for m in 1:Nt
        sA = zeros(ComplexF64, 2, 2)
        sC = zeros(ComplexF64, 2)
        for k in 1:m-1
            Kk = K22(m+1, k+1)
            sA += Kk * A[k]
            sC += Kk * C[k]
        end
        cellSrc = zeros(ComplexF64, 2)
        for k in 1:mdl.L
            ε, Δ, V = mdl.ws[k], mdl.Δ, mdl.Vs[k]
            Hbk = ComplexF64[ε Δ; conj(Δ) -ε]
            𝒱 = ComplexF64[V 0; 0 -V]
            cellSrc += -im * h * (𝒱 * (Ebs[k] * γ[:, k]))
            Ebs[k] = exp(-im * h * Hbk) * Ebs[k]           # e^{−i𝓗_bk t_m}
        end
        A[m+1] = Pph * (A[m] - sA)
        C[m+1] = Pph * (C[m] - sC + cellSrc)
    end
    # ⑤ 组装
    return [M11 * A[m+1][1, 1] + F0 * A[m+1][1, 2] + C[m+1][1] for m in 0:Nt]
end

"""
    strict_mixed_bcs(mdl, N, T, order) -> G₁>[t_1..t_N]

严格 cell 链参照：G₁>(t_m) = [(st)^m·M₀]₁₁，st = 实时 Lie / Strang cell。
"""
function strict_mixed_bcs(mdl, N, T, order::Int)
    h = T / N
    𝓗 = bcs_bdg(mdl)
    n = mdl.L + 1
    hA = zeros(ComplexF64, 2n, 2n)
    hA[1, 1] = mdl.ϵd
    hA[n+1, n+1] = -mdl.ϵd
    eA = exp(-im * h / 2 * hA)
    eB = exp(-im * h * (𝓗 - hA))
    st = order == 1 ? exp(-im * h * hA) * eB : eA * eB * eA
    M0 = bcs_M0(mdl)
    return [(st^m * M0)[1, 1] for m in 1:N], h
end

# ------------------------------------------------------------------ #

end  # module

@testset "bcs_Δm：朴素混合装配（一阶）" begin
    mdl = FirstOrderBCSMixedtime.bcs_toulouse()
    gex = FirstOrderBCSMixedtime.exact_greater_eq(mdl)     # [S(t)M₀]₁₁（全平衡 G₁>）
    errs = Float64[]
    for N in ORDER_NS
        G = FirstOrderBCSMixedtime.bcs_mixed_naive(mdl, N, N, 2.0)  # G[m+1] ↔ G₁>(t_m)
        push!(errs, maximum(abs(G[m+1] - gex(m * 2.0 / N)) for m in frac_nodes(N)))
    end
    assert_order(errs, 0.75, 1.30)
end
