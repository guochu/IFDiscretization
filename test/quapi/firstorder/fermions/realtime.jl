# =============================================================================
# test/quapi/firstorder/fermions/realtime.jl —— fermionic_Δt 朴素实时递推（frozen_first_order）⇒ 全局一阶
#
# 求解器快照拷贝自 performance/firstorder/fermions/realtime.jl（main 与 PROGRAM_FILE 守卫已删，其余逐字未动；
# 相对 include 路径在 test/quapi 镜像树内原样解析）。外包 module 恢复
# performance「每脚本独立进程」的命名隔离，使本文件自包含、可任意顺序
# include；断言只经模块限定名访问求解器。与原件同步是人工责任——基准数值
# 由 archive_regression.jl 运行原件逐字节兜底，拷贝漂移则阶数断言自暴露。
# =============================================================================
module FirstOrderFermionsRealtime
# =============================================================================
# firstorder/fermions/realtime.jl —— 费米实时：一阶冻结核装配 vs 顶点移位改善
# 对照：严格解 A(t) 与严格二阶 Trotter。输出 results/realtime.csv
#
#   · 一阶（face-value 配对，不移位）：K[j,k] = (Δ⁺₋−Δ⁺⁺)[j,k]（费米推迟
#     组合，lag j−k 直读）+ 杂质相位每步 u=e^{−iϵ_d h}；
#   · 顶点移位改善：节点 lag d ↔ 条目 lag d−1 + 三角并入次对角（同批条目，
#     误差常数下降，阶数不变）。
# =============================================================================
include(joinpath(@__DIR__, "..", "..", "secondorder", "fermions", "common.jl"))

function frozen_first_order(mdl, N, T)
    h = T / N
    u = exp(-im * mdl.ϵd * h)
    corr = fermionic_Δt(mdl.bath.spectrum; β=mdl.β, N=N, t=T, μ=mdl.μ)
    P = branch(corr; b1=:+, b2=:+); X = branch(corr; b1=:+, b2=:-)
    Δf(ℓ) = -(X[ℓ+1, 1] + P[ℓ+1, 1])
    A = zeros(ComplexF64, N + 1); A[1] = 1.0
    for m in 1:N
        s = 0.0 + 0.0im
        for d in 1:m-1
            s += Δf(d) * A[m-d]            # face-value：lag d ↔ 条目 d（不移位）
        end
        A[m+1] = u * (A[m] - s)
    end
    return A[2:end], h
end

function vertex_shifted(mdl, N, T)
    h = T / N
    u = exp(-im * mdl.ϵd * h)
    corr = fermionic_Δt(mdl.bath.spectrum; β=mdl.β, N=N, t=T, μ=mdl.μ)
    P = branch(corr; b1=:+, b2=:+); X = branch(corr; b1=:+, b2=:-)
    Δf(ℓ) = -(X[ℓ+1, 1] + P[ℓ+1, 1])
    g0 = Δf(0)
    g = [Δf(d) for d in 1:N-1]
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

@testset "fermionic_Δt：朴素实时递推（一阶）" begin
    mdl = FirstOrderFermionsRealtime.fermi_toulouse()
    Aex = FirstOrderFermionsRealtime.exact_return_amplitude(mdl)
    errs = Float64[]
    for N in ORDER_NS
        A, h = FirstOrderFermionsRealtime.frozen_first_order(mdl, N, 2.0)  # A[m] ↔ t_m
        push!(errs, maximum(abs(A[m] - Aex(m * h)) for m in frac_nodes(N)))
    end
    assert_order(errs, 0.75, 1.30)
end
