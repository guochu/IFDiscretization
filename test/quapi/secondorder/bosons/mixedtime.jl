# =============================================================================
# test/quapi/secondorder/bosons/mixedtime.jl —— bosonic_Δm + K/IF 拆分（weyl_k_mixed）⇒ 全局二阶
#
# 求解器快照拷贝自 performance/secondorder/bosons/mixedtime.jl（main 与 PROGRAM_FILE 守卫已删，其余逐字未动；include 行改直连 common；
# 相对 include 路径在 test/quapi 镜像树内原样解析）。外包 module 恢复
# performance「每脚本独立进程」的命名隔离，使本文件自包含、可任意顺序
# include；断言只经模块限定名访问求解器。与原件同步是人工责任——基准数值
# 由 archive_regression.jl 运行原件逐字节兜底，拷贝漂移则阶数断言自暴露。
# =============================================================================
module SecondOrderBosonsMixedtime
# =============================================================================
# secondorder/bosons/mixedtime.jl —— U1GTEMPO2 "冻结核不变 + K 侧二阶"的混合
# 时间复现（L 形轮廓：τ 环热制备 + 实时演化，玻色 Toulouse 平衡 G>(t)），
# 并输出画图数据 results/mixedtime.csv（Nτ=Nt 同步，dt=dτ）
#
# 精确单粒子分解：G>(t)/(−i) = (1+n_d)·A(t) + cross(t)，三个源通道共享同一
# 实时隐式梯形系统（realtime.jl 骨架）：
#   · 通道 1（retarded）：A_m —— 结点初始振幅，bosonic_Δm 的 ++/+- 块推迟组合，
#     梯形 φ + 半步相位 f + 接触对角（同实时三要素）；
#   · 通道 2（占据）：n_d = G^τ(β⁻)（ξ 块环端点，玻色 KMS）；
#   · 通道 3（初关联）：rhs_m = f·Σ_j ζ⁺ᵗ[m,j]·φ^τ_j —— (+:τ) 跨块 × τ 环解
#     梯形值（含双侧跳变），×吸收半步相位 f，块行取终止 cell（块为起始 cell
#     约定，行偏移 −1）。
#
# 运行：JULIA_PROBE_LIBSTDCXX=0 julia --startup-file=no --project=. secondorder/bosons/mixedtime.jl
# =============================================================================
include(joinpath(@__DIR__, "..", "..", "common", "common.jl"))  # （快照改动）直连 common，不再经 imagtime.jl 传递

"""
    weyl_k_mixed(mdl, Nτ, Nt, T) -> G>[t_0..t_Nt]

混合轮廓 K 侧二阶。`G[m+1] ≈ G>(t_m)`。
"""
function weyl_k_mixed(mdl, Nτ, Nt, T)
    h = T / Nt
    δτ = mdl.β / Nτ
    u = exp(-im * mdl.ϵd * h)          # 实时支相位
    f = exp(-im * mdl.ϵd * h / 2)
    uτ = exp(-mdl.ϵd * δτ)             # τ 支相位（实衰减，虚时无 i）
    fτ = exp(-mdl.ϵd * δτ / 2)
    mix = bosonic_Δm(mdl.bath.spectrum; β=mdl.β, Nτ=Nτ, t=T, Nt=Nt, μ=mdl.μ)
    P = branch(mix; b1=:+, b2=:+)
    Xr = branch(mix; b1=:+, b2=:-)
    Xc = branch(mix; b1=:+, b2=:τ)              # (Nt+1)×Nτ 跨块（起始 cell 约定）
    d0 = -(P[1, 1] + Xr[1, 1])
    # ① τ 支环（ξ 块 = bosonic_Δτ 同数据，逐位验证；四要素：梯形 φ、半步相位
    #    fτ、接触对角、跳变 G₀ = G_N+1 完整代入——自由项 b += u 与 φ₁ 左半 b += fτ·w/2）
    ξ = branch(mix; b1=:τ, b2=:τ)                # branch(:τ,:τ) = CM(−ξⱼₖ,−ξₖⱼ) = Δτ 条目
    ηjk, ηkj = ξ.ηⱼₖ, ξ.ηₖⱼ
    d0τ = ηjk[1] + ηkj[1]
    M = zeros(Float64, Nτ, Nτ)
    b = zeros(Float64, Nτ)
    for m in 1:Nτ
        M[m, m] += 1.0
        if m ≥ 2
            M[m, m-1] -= uτ
        else
            b[m] += uτ                          # −u·G₀ = −u·(G_N+1)：常数 +u
            M[m, Nτ] -= uτ
        end
        for k in 1:Nτ
            w = k < m ? ηjk[m-k+1] : k > m ? ηkj[k-m+1] : d0τ
            if k == 1                           # φ₁ 左半 = (G_N+1)/2
                M[m, Nτ] -= fτ * w * 0.5
                M[m, 1] -= fτ * w * 0.5
                b[m] += fτ * w * 0.5
            else
                M[m, k-1] -= fτ * w * 0.5
                M[m, k] -= fτ * w * 0.5
            end
        end
    end
    Gτ = M \ b
    n_d = Gτ[Nτ]
    Gτ0 = Gτ[Nτ] + 1
    φτ = [((j == 1 ? Gτ0 : Gτ[j-1]) + Gτ[j]) / 2 for j in 1:Nτ]
    # 实时双通道
    A = zeros(ComplexF64, Nt + 1); A[1] = 1.0
    C = zeros(ComplexF64, Nt + 1)
    for m in 1:Nt
        s1 = 0.0 + 0.0im
        s3 = 0.0 + 0.0im
        for k in 1:m-1
            Δ = -(P[m+1, k+1] + Xr[m+1, k+1])
            s1 += Δ * (A[k] + A[k+1]) / 2
            s3 += Δ * (C[k] + C[k+1]) / 2
        end
        rhs = (1 ≤ m ≤ Nt + 1) ? f * sum(Xc[m, j] * φτ[j] for j in 1:Nτ) : 0.0im
        A[m+1] = (u*A[m] - f*s1 - f*d0*A[m]/2) / (1 + f*d0/2)
        C[m+1] = (u*C[m] - f*s3 - f*d0*C[m]/2 + rhs) / (1 + f*d0/2)
    end
    return (-im) .* ((1 + n_d) .* A .+ C)
end

# ------------------------------------------------------------------ #

end  # module

@testset "bosonic_Δm：Weyl-K 混合装配（二阶）" begin
    mdl = SecondOrderBosonsMixedtime.boson_toulouse()
    GL = SecondOrderBosonsMixedtime.exact_greater_lesser(mdl)
    errs = Float64[]
    for N in ORDER_NS
        G = SecondOrderBosonsMixedtime.weyl_k_mixed(mdl, N, N, 2.0)  # G[m+1] ↔ G>(t_m)
        push!(errs, maximum(abs(G[m+1] - GL(m * 2.0 / N)[1]) for m in frac_nodes(N)))
    end
    assert_order(errs, 1.80, 2.20)
end
