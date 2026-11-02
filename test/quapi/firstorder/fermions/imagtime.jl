# =============================================================================
# test/quapi/firstorder/fermions/imagtime.jl —— fermionic_Δτ 朴素反周期环（frozen_first_order_ring）⇒ 全局一阶
#
# 求解器快照拷贝自 performance/firstorder/fermions/imagtime.jl（main 与 PROGRAM_FILE 守卫已删，其余逐字未动；
# 相对 include 路径在 test/quapi 镜像树内原样解析）。外包 module 恢复
# performance「每脚本独立进程」的命名隔离，使本文件自包含、可任意顺序
# include；断言只经模块限定名访问求解器。与原件同步是人工责任——基准数值
# 由 archive_regression.jl 运行原件逐字节兜底，拷贝漂移则阶数断言自暴露。
# =============================================================================
module FirstOrderFermionsImagtime
# =============================================================================
# firstorder/fermions/imagtime.jl —— 费米虚时：一阶冻结核环 vs 严格 Trotter 1/2 阶
# 对照：严格解 G(τ)（greater 约定谱求和）。输出 results/imagtime.csv
#
#   · 一阶（朴素矩形）：反周期双符号闭合、节点值权重、无梯形、无跳变常数
#     代入 φ₁；
#   · 严格参照：精确 cell 热环 (I+T^N)⁻¹（费米分母，对离散链精确）。
# 注：顶点移位是实时轮廓特有的自包含改善；虚时的二阶改善属 K 侧要素打包
# （梯形+半步+接触+跳变常数代入），见 secondorder/fermions/imagtime.jl。
# =============================================================================
include(joinpath(@__DIR__, "..", "..", "secondorder", "fermions", "common.jl"))

"""
    frozen_first_order_ring(mdl, N) -> (G[1..N], h)

费米虚时一阶冻结核环（朴素矩形）。`G[m] ≈ G(τ_m)`（greater 约定）。
"""
function frozen_first_order_ring(mdl, N)
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
        if m ≥ 2; M[m, m-1] -= u
        else; b[m] -= u; M[m, N] += u
        end
        for k in 1:N
            w = k < m ? ηjk[m-k+1] : k > m ? ηkj[k-m+1] : d0
            M[m, k] -= f * w                     # 矩形：节点值、无跳变常数代入
        end
    end
    return M \ b, h
end

"严格参照（greater 约定谱求和）与精确 cell 热环。"
function exact_refs(mdl, N)
    ev = eigen(Hermitian(mdl.hfull))
    h = mdl.β / N
    ex = [-sum(ev.vectors[1, p]^2 * exp(-ev.values[p] * m * h) /
               (1 + exp(-mdl.β * ev.values[p])) for p in 1:length(ev.values)) for m in 1:N]
    T = exp(-h * mdl.hfull)
    P = inv(mdl.Id + T^N)
    strict = [-(T^m * P)[1, 1] for m in 1:N]     # 精确 cell 热环（greater 约定）
    return ex, strict, h
end

# ------------------------------------------------------------------ #

end  # module

@testset "fermionic_Δτ：朴素反周期环（一阶）" begin
    mdl = FirstOrderFermionsImagtime.fermi_toulouse()
    errs = Float64[]
    for N in ORDER_NS
        G, _ = FirstOrderFermionsImagtime.frozen_first_order_ring(mdl, N)  # greater 约定
        ex, _ = FirstOrderFermionsImagtime.exact_refs(mdl, N)
        push!(errs, maximum(abs(G[m] - ex[m]) for m in frac_nodes(N)))
    end
    assert_order(errs, 0.75, 1.30)
end
