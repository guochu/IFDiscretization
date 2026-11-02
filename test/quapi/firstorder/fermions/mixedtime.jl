# =============================================================================
# test/quapi/firstorder/fermions/mixedtime.jl —— fermionic_Δm 朴素混合装配（frozen_mixed_naive）⇒ 全局一阶
#
# 求解器快照拷贝自 performance/firstorder/fermions/mixedtime.jl（main 与 PROGRAM_FILE 守卫已删，其余逐字未动；
# 相对 include 路径在 test/quapi 镜像树内原样解析）。外包 module 恢复
# performance「每脚本独立进程」的命名隔离，使本文件自包含、可任意顺序
# include；断言只经模块限定名访问求解器。与原件同步是人工责任——基准数值
# 由 archive_regression.jl 运行原件逐字节兜底，拷贝漂移则阶数断言自暴露。
# =============================================================================
module FirstOrderFermionsMixedtime
# =============================================================================
# firstorder/fermions/mixedtime.jl —— 费米混合轮廓：一阶冻结核 vs 严格 1/2 阶
# Trotter。对照：严格解（全平衡 G>）。输出 results/mixedtime.csv
#
#   · 一阶（朴素装配，face-value）：τ 环矩形（反周期双符号闭合、节点值、无梯
#     形、无跳变常数代入 φ₁）、实时 A 通道 face-value Δ = −(P+X)（Δm 约定）
#     full-step u 矩形源、C 通道连续卷积递推 + 朴素梯形 γ 提取（无消峰——
#     ε<0 模权重尖峰欠采样 → 一阶病态）；
#   · 全 K 侧改善（梯形 φ、半步相位 f、接触对角、跳变常数完整代入、消峰 γ
#     提取）= 直接二阶，见 secondorder/fermions/mixedtime.jl（firstorder 不重复）。
#
# 运行：JULIA_PROBE_LIBSTDCXX=0 julia --startup-file=no --project=. firstorder/fermions/mixedtime.jl
# =============================================================================
include(joinpath(@__DIR__, "..", "..", "secondorder", "fermions", "common.jl"))

"𝒦(D) = Σ_k V_k² e^{−iϵ_dD}(e^{iω_kD}−1)/(iω_k)（bath-only 单重卷积核，闭式）。"
function mathcalK(mdl, D)
    S = 0.0 + 0.0im
    for k in 1:mdl.L
        ω = mdl.ϵd - mdl.ws[k]
        S += mdl.αs[k] * exp(-im*mdl.ϵd*D) *
             (abs(ω) < 1e-10 ? im*D : (exp(im*ω*D) - 1) / (im*ω))
    end
    return S
end

"""
    frozen_mixed_naive(mdl, Nτ, Nt, T) -> G>[t_0..t_Nt]

费米混合一阶冻结核（朴素装配）。`G[m+1] ≈ G>(t_m)`。
"""
function frozen_mixed_naive(mdl, Nτ, Nt, T)
    h = T / Nt
    δτ = mdl.β / Nτ
    u = exp(-im * mdl.ϵd * h)
    f = exp(-im * mdl.ϵd * h / 2)
    uτ = exp(-mdl.ϵd * δτ)
    dm = fermionic_Δm(mdl.bath.spectrum; β=mdl.β, Nτ=Nτ, t=T, Nt=Nt, μ=mdl.μ)
    ξ = branch(dm; b1=:τ, b2=:τ)
    ηjk, ηkj = ξ.ηⱼₖ, ξ.ηₖⱼ
    d0τ = ηjk[1] + ηkj[1]
    P = branch(dm; b1=:+, b2=:+)
    Xr = branch(dm; b1=:+, b2=:-)
    Xc = branch(dm; b1=:+, b2=:τ)
    # ① τ 支环（朴素矩形：反周期双符号闭合、节点值、无梯形、无跳变常数代入 φ₁）
    M = zeros(ComplexF64, Nτ, Nτ)
    b = zeros(ComplexF64, Nτ)
    for m in 1:Nτ
        M[m, m] += 1.0
        if m ≥ 2
            M[m, m-1] -= uτ
        else
            b[m] -= uτ
            M[m, Nτ] += uτ
        end
        for k in 1:Nτ
            w = k < m ? ηjk[m-k+1] : k > m ? ηkj[k-m+1] : d0τ
            M[m, k] -= w                             # 矩形：节点值权重
        end
    end
    Gτ = M \ b
    n_d = -Gτ[Nτ]
    # γ[1,k] 朴素梯形提取（无消峰——一阶病态来源）
    γrow = zeros(ComplexF64, mdl.L)
    G0v = -1 - Gτ[Nτ]
    Gext = vcat(G0v, Gτ)
    for k in 1:mdl.L
        ε, V = mdl.ws[k], sqrt(mdl.αs[k])
        acc = 0.0 + 0.0im
        for j in 1:Nτ
            w0 = exp(-ε*(mdl.β - (j-1)*δτ))
            w1 = exp(-ε*(mdl.β - j*δτ))
            acc += δτ * (w0*Gext[j] + w1*Gext[j+1]) / 2
        end
        γrow[k] = V * acc / (1 + exp(-ε*mdl.β))
    end
    # ② 实时支 A（face-value Δ = −(P+X)、矩形源、接触对角隐式稳定化）
    Δ(j, k) = -(Xr[j, k] + P[j, k]); d0 = Δ(1, 1)
    A = zeros(ComplexF64, Nt + 1); A[1] = 1.0
    for m in 1:Nt
        s = 0.0 + 0.0im
        for k in 1:m-1
            s += Δ(m+1, k+1) * A[k]
        end
        A[m+1] = (u*A[m] - f*s - f*d0*A[m]/2) / (1 + f*d0/2)
    end
    # ③ C 通道：连续卷积递推 + 全历史 Duhamel 源（ζ 因式分解的实时侧因子）×朴素 γ
    C = zeros(ComplexF64, Nt + 1)
    for m in 1:Nt
        tm = m * h
        S = 0.0 + 0.0im
        for k in 1:mdl.L
            ωk = mdl.ϵd - mdl.ws[k]
            Jk = exp(-im*mdl.ϵd*tm) * (abs(ωk) < 1e-10 ? tm :
                 (exp(im*ωk*tm) - 1) / (im*ωk))
            S += -im * sqrt(mdl.αs[k]) * Jk * (-γrow[k])
        end
        mem = 0.0 + 0.0im
        for j in 1:m-1
            mem += mathcalK(mdl, (m-j)*h) * C[j+1]
        end
        C[m+1] = S - h * mem
    end
    return (-im) .* ((1 - n_d) .* A .+ C)
end

# ------------------------------------------------------------------ #

end  # module

@testset "fermionic_Δm：朴素混合装配（一阶）" begin
    mdl = FirstOrderFermionsMixedtime.fermi_toulouse()
    # 严格平衡 G>(t) = −i[e^{−iHt}(I − γᵀ)]₁₁（谱分解，仅参照用）
    ev = eigen(Hermitian(mdl.hfull))
    γ = ev.vectors * Diagonal([1 / (exp(mdl.β * λ) + 1) for λ in ev.values]) * ev.vectors'
    gex(t) = -im * (exp(-im * t * mdl.hfull) * (mdl.Id - transpose(γ)))[1, 1]
    errs = Float64[]
    for N in ORDER_NS
        G = FirstOrderFermionsMixedtime.frozen_mixed_naive(mdl, N, N, 2.0)  # G[m+1] ↔ G>(t_m)
        push!(errs, maximum(abs(G[m+1] - gex(m * 2.0 / N)) for m in frac_nodes(N)))
    end
    assert_order(errs, 0.75, 1.30)
end
