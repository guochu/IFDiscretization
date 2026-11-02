# =============================================================================
# test/quapi/firstorder/bosons/imagtime.jl —— bosonic_Δτ 朴素虚时环 ⇒ 全局一阶（求解器 frozen_dyson_Gτ 在 common 拷贝中）
#
# 求解器快照拷贝自 performance/firstorder/bosons/imagtime.jl（main 与 PROGRAM_FILE 守卫已删，其余逐字未动；
# 相对 include 路径在 test/quapi 镜像树内原样解析）。外包 module 恢复
# performance「每脚本独立进程」的命名隔离，使本文件自包含、可任意顺序
# include；断言只经模块限定名访问求解器。与原件同步是人工责任——基准数值
# 由 archive_regression.jl 运行原件逐字节兜底，拷贝漂移则阶数断言自暴露。
# =============================================================================
module FirstOrderBosonsImagtime
# =============================================================================
# firstorder/bosons/imagtime.jl —— 玻色虚时：一阶冻结核环 vs 严格 Trotter 1/2 阶
# 对照：严格解 G(τ)。输出 results/imagtime.csv
#
#   · 一阶（frozen_dyson_Gτ 同款）：bosonic_Δτ 条目 + 等时两序和，旧装配；
#   · 严格一阶/二阶 Trotter 热环参照（chain_Gτ）。
# 注：顶点移位是实时轮廓特有的自包含改善；虚时的二阶改善属 K 侧要素打包
# （梯形+半步+接触+跳变代入），见 secondorder/bosons/imagtime.jl。
# =============================================================================
include(joinpath(@__DIR__, "..", "..", "common", "trotter_frozen.jl"))


end  # module

@testset "bosonic_Δτ：朴素虚时环（一阶）" begin
    mdl = FirstOrderBosonsImagtime.boson_toulouse()
    Gex = FirstOrderBosonsImagtime.exact_Gτ(mdl)
    errs = Float64[]
    for N in ORDER_NS
        G, h = FirstOrderBosonsImagtime.frozen_dyson_Gτ(mdl, N)   # G[m] ↔ τ_m, m = 1..N−1
        push!(errs, maximum(abs(G[m] - Gex(m * h)) for m in frac_nodes(N)))
    end
    assert_order(errs, 0.75, 1.30)
end
