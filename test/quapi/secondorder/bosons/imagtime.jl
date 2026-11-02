# =============================================================================
# test/quapi/secondorder/bosons/imagtime.jl —— bosonic_Δτ + K/IF 拆分（weyl_k_imagtime）⇒ 全局二阶
#
# 求解器快照拷贝自 performance/secondorder/bosons/imagtime.jl（main 与 PROGRAM_FILE 守卫已删，其余逐字未动；
# 相对 include 路径在 test/quapi 镜像树内原样解析）。外包 module 恢复
# performance「每脚本独立进程」的命名隔离，使本文件自包含、可任意顺序
# include；断言只经模块限定名访问求解器。与原件同步是人工责任——基准数值
# 由 archive_regression.jl 运行原件逐字节兜底，拷贝漂移则阶数断言自暴露。
# =============================================================================
module SecondOrderBosonsImagtime
# =============================================================================
# secondorder/bosons/imagtime.jl —— U1GTEMPO2 "冻结核不变 + K 侧二阶"的虚时复现
# （玻色 Toulouse，Matsubara 环 G(τ)），并输出画图数据 results/imagtime.csv
#
# 方案（环形隐式梯形，冻结核 = bosonic_Δτ 条目原样、face-value cell 配对）：
#   G_m − u·G_{m−1} − f·Σ_k D[m,k]·φ_k = 0,  G₀ := G_N + 1（双侧跳变，c-number）
#   u = e^{−ϵ_d h}；f = e^{−ϵ_d h/2}；φ_k = (G_{k−1}+G_k)/2（环梯形，φ₁ 左半取
#   G_N+1）；D = 前向 ηⱼₖ / 环绕 ηₖⱼ / 对角等时和 d₀（接触全量对角，隐式）。
# 跳变 = [d,d†]=1（与耦合无关），完整代入自由项与 φ₁ 是环上二阶的决定性要素。
#
# 运行：JULIA_PROBE_LIBSTDCXX=0 julia --startup-file=no --project=. secondorder/bosons/imagtime.jl
# =============================================================================
include(joinpath(@__DIR__, "..", "..", "common", "common.jl"))

"""
    weyl_k_imagtime(mdl, N) -> (G[1..N], h)

虚时 K 侧二阶（Weyl-K 环）。`G[m] ≈ G(τ_m)`，`G[N] = G(β⁻)`。
"""
function weyl_k_imagtime(mdl, N)
    h = mdl.β / N
    u = exp(-mdl.ϵd * h)
    f = exp(-mdl.ϵd * h / 2)
    corr = bosonic_Δτ(mdl.bath.spectrum; β=mdl.β, N=N, μ=mdl.μ)
    ηjk, ηkj = corr.data.ηⱼₖ, corr.data.ηₖⱼ
    d0 = ηjk[1] + ηkj[1]                        # 等时（两序三角之和）
    M = zeros(Float64, N, N)
    b = zeros(Float64, N)
    for m in 1:N
        M[m, m] += 1.0
        if m ≥ 2
            M[m, m-1] -= u
        else
            b[m] += u                           # −u·G₀ = −u·(G_N+1)：常数 +u
            M[m, N] -= u
        end
        for k in 1:N
            w = k < m ? ηjk[m-k+1] : k > m ? ηkj[k-m+1] : d0
            if k == 1                           # φ₁ 左半 = (G_N+1)/2
                M[m, N] -= f * w * 0.5
                M[m, 1] -= f * w * 0.5
                b[m] += f * w * 0.5
            else
                M[m, k-1] -= f * w * 0.5
                M[m, k] -= f * w * 0.5
            end
        end
    end
    return M \ b, h
end

# ------------------------------------------------------------------ #

end  # module

@testset "bosonic_Δτ：Weyl-K 虚时环（二阶）" begin
    mdl = SecondOrderBosonsImagtime.boson_toulouse()
    Gex = SecondOrderBosonsImagtime.exact_Gτ(mdl)
    errs = Float64[]
    for N in ORDER_NS
        G, h = SecondOrderBosonsImagtime.weyl_k_imagtime(mdl, N)   # G[m] ↔ τ_m
        push!(errs, maximum(abs(G[m] - Gex(m * h)) for m in frac_nodes(N)))
    end
    assert_order(errs, 1.80, 2.20)
end
