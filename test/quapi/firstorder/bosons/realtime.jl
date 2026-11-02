# =============================================================================
# test/quapi/firstorder/bosons/realtime.jl —— bosonic_Δt 朴素（face-value）实时装配 ⇒ 全局一阶（求解器 frozen_retarded_A 在 common 拷贝中）
#
# 求解器快照拷贝自 performance/firstorder/bosons/realtime.jl（main 与 PROGRAM_FILE 守卫已删，其余逐字未动；
# 相对 include 路径在 test/quapi 镜像树内原样解析）。外包 module 恢复
# performance「每脚本独立进程」的命名隔离，使本文件自包含、可任意顺序
# include；断言只经模块限定名访问求解器。与原件同步是人工责任——基准数值
# 由 archive_regression.jl 运行原件逐字节兜底，拷贝漂移则阶数断言自暴露。
# =============================================================================
module FirstOrderBosonsRealtime
# =============================================================================
# firstorder/bosons/realtime.jl —— 玻色实时：一阶冻结核装配 vs 顶点移位改善
# 对照：严格解 A(t) 与严格二阶 Trotter（Strang cell 链）。输出 results/realtime.csv
#
#   · 一阶（frozen_retarded_A 同款）：face-value lag 配对（不移位）+ 杂质相位
#     每步 u=e^{−iϵ_d h}：A₀⁻¹ = I−u·S⁻，K[j,k] = −(Δ⁺⁺+Δ⁺₋)[j,k]；
#   · 顶点移位改善：节点 lag d ↔ 条目 lag d−1（顶点内收半格）+ 三角并入次对角
#     (1−g₀)——同批 bosonic_Δt 条目，误差常数 ↓~12×（阶数不变，均一阶）。
# =============================================================================
include(joinpath(@__DIR__, "..", "..", "common", "common.jl"))

function vertex_shifted(mdl, N, T)
    h = T / N
    u = exp(-im * mdl.ϵd * h)
    corr = bosonic_Δt(mdl.bath.spectrum; β=mdl.β, N=N, t=T, μ=mdl.μ)
    P = branch(corr; b1=:+, b2=:+); X = branch(corr; b1=:+, b2=:-)
    g0 = -(P[1, 1] + X[1, 1])
    g = [-(P[d+1, 1] + X[d+1, 1]) for d in 1:N-1]
    M = Matrix{ComplexF64}(I, N, N); b = zeros(ComplexF64, N)
    for m in 1:N
        if m ≥ 2; M[m, m-1] -= u * (1 - g0); else; b[m] += u * (1 - g0); end
        for d in 2:m
            if m - d ≥ 1; M[m, m-d] += u * g[d-1]; else; b[m] -= u * g[d-1]; end
        end
    end
    return M \ b, h
end


end  # module

@testset "bosonic_Δt：朴素实时推迟装配（一阶）" begin
    mdl = FirstOrderBosonsRealtime.boson_toulouse()
    Aex = FirstOrderBosonsRealtime.exact_return_amplitude(mdl)
    errs = Float64[]
    for N in ORDER_NS
        A, h = FirstOrderBosonsRealtime.frozen_retarded_A(mdl, N, 2.0)  # A[j] ↔ t_{j−1}
        push!(errs, maximum(abs(A[m+1] - Aex(m * h)) for m in frac_nodes(N)))
    end
    assert_order(errs, 0.75, 1.30)
end
