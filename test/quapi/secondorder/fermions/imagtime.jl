# =============================================================================
# test/quapi/secondorder/fermions/imagtime.jl —— fermionic_Δτ + K/IF 拆分（weyl_k_imagtime）⇒ 全局二阶
#
# 求解器快照拷贝自 performance/secondorder/fermions/imagtime.jl（main 与 PROGRAM_FILE 守卫已删，其余逐字未动；
# 相对 include 路径在 test/quapi 镜像树内原样解析）。外包 module 恢复
# performance「每脚本独立进程」的命名隔离，使本文件自包含、可任意顺序
# include；断言只经模块限定名访问求解器。与原件同步是人工责任——基准数值
# 由 archive_regression.jl 运行原件逐字节兜底，拷贝漂移则阶数断言自暴露。
# =============================================================================
module SecondOrderFermionsImagtime
# =============================================================================
# secondorder/fermions/imagtime.jl —— 费米 Toulouse 虚时 K 侧二阶（Weyl-K 环）
# 输出画图数据 results/imagtime.csv
#
# 与玻色侧（../bosons/imagtime.jl）同骨架，两处费米特异性：
#   · 跳变：反周期 + δ 源 ⟹ G₀ = −1 − G_N（玻色为 G₀ = G_N + 1）；符号翻转
#     同时出现在右端常数与 G_N 列系数（φ₁ 左半 = (−1−G_N)/2）；
#   · 约定：解为 Matsubara greater 约定 G(τ) = −⟨d(τ)d†(0)⟩（V=0 极限
#     −(1−n̄)e^{−ϵτ} 逐点机器精度验证）；严格参照取谱求和同约定。
#
# 运行：JULIA_PROBE_LIBSTDCXX=0 julia --startup-file=no --project=. secondorder/fermions/imagtime.jl
# =============================================================================
include(joinpath(@__DIR__, "common.jl"))

"""
    weyl_k_imagtime(mdl, N) -> (G[1..N], h)

费米虚时 K 侧二阶（Weyl-K 环，greater 约定）。`G[m] ≈ G(τ_m)`，`G[N] = G(β⁻)`。
"""
function weyl_k_imagtime(mdl, N)
    h = mdl.β / N
    u = exp(-mdl.ϵd * h)
    f = exp(-mdl.ϵd * h / 2)
    corr = fermionic_Δτ(mdl.bath.spectrum; β=mdl.β, N=N, μ=mdl.μ)
    ηjk, ηkj = corr.data.ηⱼₖ, corr.data.ηₖⱼ
    d0 = ηjk[1] + ηkj[1]
    M = zeros(ComplexF64, N, N)
    b = zeros(ComplexF64, N)
    for m in 1:N
        M[m, m] += 1.0
        if m ≥ 2
            M[m, m-1] -= u
        else
            b[m] -= u                           # −u·G₀ = −u·(−1−G_N) = +u + u·G_N
            M[m, N] += u
        end
        for k in 1:N
            w = k < m ? ηjk[m-k+1] : k > m ? ηkj[k-m+1] : d0
            if k == 1                           # φ₁ = (−1−G_N+G₁)/2
                M[m, N] += f * w * 0.5
                M[m, 1] -= f * w * 0.5
                b[m] -= f * w * 0.5
            else
                M[m, k-1] -= f * w * 0.5
                M[m, k] -= f * w * 0.5
            end
        end
    end
    return M \ b, h
end

"严格参照（greater 约定谱求和）：G(τ_m) = −Σ|U₁ₚ|²e^{−λ_pτ}/(1+e^{−βλ_p})。"
function exact_Gτ(mdl, N)
    ev = eigen(Hermitian(mdl.hfull))
    h = mdl.β / N
    return [(-1)^0 * -sum(ev.vectors[1, p]^2 * exp(-ev.values[p] * m * h) /
                          (1 + exp(-mdl.β * ev.values[p])) for p in 1:length(ev.values))
            for m in 1:N], h
end

# ------------------------------------------------------------------ #

end  # module

@testset "fermionic_Δτ：Weyl-K 反周期环（二阶）" begin
    mdl = SecondOrderFermionsImagtime.fermi_toulouse()
    errs = Float64[]
    for N in ORDER_NS
        G, _ = SecondOrderFermionsImagtime.weyl_k_imagtime(mdl, N)  # greater 约定
        ex, _ = SecondOrderFermionsImagtime.exact_Gτ(mdl, N)
        push!(errs, maximum(abs(G[m] - ex[m]) for m in frac_nodes(N)))
    end
    assert_order(errs, 1.80, 2.20)
end
