# =============================================================================
# test/quapi/firstorder/bosons/mixedtime.jl —— bosonic_Δm 朴素混合装配（frozen_mixed_naive）⇒ 全局一阶
#
# 求解器快照拷贝自 performance/firstorder/bosons/mixedtime.jl（main 与 PROGRAM_FILE 守卫已删，其余逐字未动；
# 相对 include 路径在 test/quapi 镜像树内原样解析）。外包 module 恢复
# performance「每脚本独立进程」的命名隔离，使本文件自包含、可任意顺序
# include；断言只经模块限定名访问求解器。与原件同步是人工责任——基准数值
# 由 archive_regression.jl 运行原件逐字节兜底，拷贝漂移则阶数断言自暴露。
# =============================================================================
module FirstOrderBosonsMixedtime
# =============================================================================
# firstorder/bosons/mixedtime.jl —— 玻色混合轮廓：一阶冻结核 vs 严格 1/2 阶
# Trotter。对照：严格解（全平衡 G>）。输出 results/mixedtime.csv
#
#   · 一阶（E1 冻结核装配，face-value）：τ 支用正确环解作热背景（n_d、φτ
#     ——环是热背景输入），实时通道 = QuAPI 一阶装配 A = (A0⁻¹+K)⁻¹·源
#     （A0⁻¹ = I−uS⁻，K = face-value 因果下三角含接触对角；无半步 f、
#     无梯形），ζ 跨块朴素收缩（行 +1、G 节点值权重、无 ×f）；
#   · 全 K 侧改善（梯形 φ、半步相位 f、接触对角、跳变完整代入、跨块终止
#     cell + ×f）= 直接二阶，见 secondorder/bosons/mixedtime.jl（firstorder
#     不重复）。注：仅顶点移位单独抽出与其他朴素部分组合不自洽（实测误差
#     反而增大），故混合轮廓的一阶→二阶只有"打包改善"一条路径。
#
# 运行：JULIA_PROBE_LIBSTDCXX=0 julia --startup-file=no --project=. firstorder/bosons/mixedtime.jl
# =============================================================================
include(joinpath(@__DIR__, "..", "..", "common", "common.jl"))

"""
    frozen_mixed_naive(mdl, Nτ, Nt, T) -> G>[t_0..t_Nt]

玻色混合一阶冻结核（朴素 E1 装配）。`G[m+1] ≈ G>(t_m)`。
"""
function frozen_mixed_naive(mdl, Nτ, Nt, T)
    h = T / Nt
    δτ = mdl.β / Nτ
    u = exp(-im * mdl.ϵd * h)
    fτ = exp(-mdl.ϵd * δτ / 2)
    dm = bosonic_Δm(mdl.bath.spectrum; β=mdl.β, Nτ=Nτ, t=T, Nt=Nt, μ=mdl.μ)
    P = branch(dm; b1=:+, b2=:+)
    Xr = branch(dm; b1=:+, b2=:-)
    Xc = branch(dm; b1=:+, b2=:τ)
    d0 = -(P[1, 1] + Xr[1, 1])                  # 接触对角（冻结核本体条目）
    # ① τ 支热背景：正确环解（bosonic_Δτ 驱动——热背景是输入而非被改善对象）
    corr = bosonic_Δτ(mdl.bath.spectrum; β=mdl.β, N=Nτ, μ=mdl.μ)
    ηjk, ηkj = corr.data.ηⱼₖ, corr.data.ηₖⱼ
    d0τ = ηjk[1] + ηkj[1]
    M = zeros(Float64, Nτ, Nτ)
    b = zeros(Float64, Nτ)
    for m in 1:Nτ
        M[m, m] += 1.0
        if m ≥ 2
            M[m, m-1] -= exp(-mdl.ϵd * δτ)
        else
            b[m] += exp(-mdl.ϵd * δτ)
            M[m, Nτ] -= exp(-mdl.ϵd * δτ)
        end
        for k in 1:Nτ
            w = k < m ? ηjk[m-k+1] : k > m ? ηkj[k-m+1] : d0τ
            if k == 1
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
    # ② 实时支 A/C：E1 冻结核装配 A = (A0⁻¹+K)⁻¹·源（隐式、稳定）
    #    A0⁻¹ = I − u·S⁻（full-step 自由飞行）；K = face-value 因果下三角
    #    K[m,k] = −(P+X)[m−k+1]（lag m−k 直读、含对角接触 [1,1]）
    A0inv = Matrix{ComplexF64}(I, Nt + 1, Nt + 1)
    for j in 2:Nt+1; A0inv[j, j-1] = -u; end
    K = zeros(ComplexF64, Nt + 1, Nt + 1)
    for m in 1:Nt+1, k in 1:m
        K[m, k] = -(P[m-k+1, 1] + Xr[m-k+1, 1])
    end
    A = (A0inv + K) \ [1.0; zeros(ComplexF64, Nt)]          # A 通道（初值 a₀=1）
    # C 通道：同骨架 + ζ 跨块源（朴素：行 +1、G 节点值权重、无 ×f）
    Svec = zeros(ComplexF64, Nt + 1)
    for m in 1:Nt
        Svec[m+1] = sum(Xc[m+1, j] * Gτ[j] for j in 1:Nτ)
    end
    C = (A0inv + K) \ Svec
    return (-im) .* ((1 + n_d) .* A .+ C)
end

# ------------------------------------------------------------------ #

end  # module

@testset "bosonic_Δm：朴素混合装配（一阶）" begin
    mdl = FirstOrderBosonsMixedtime.boson_toulouse()
    GL = FirstOrderBosonsMixedtime.exact_greater_lesser(mdl)
    errs = Float64[]
    for N in ORDER_NS
        G = FirstOrderBosonsMixedtime.frozen_mixed_naive(mdl, N, N, 2.0)  # G[m+1] ↔ G>(t_m)
        push!(errs, maximum(abs(G[m+1] - GL(m * 2.0 / N)[1]) for m in frac_nodes(N)))
    end
    assert_order(errs, 0.75, 1.30)
end
