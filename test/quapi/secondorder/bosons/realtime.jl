# =============================================================================
# test/quapi/secondorder/bosons/realtime.jl —— bosonic_Δt + K/IF 拆分（weyl_k_realtime）⇒ 全局二阶
#
# 求解器快照拷贝自 performance/secondorder/bosons/realtime.jl（main 与 PROGRAM_FILE 守卫已删，其余逐字未动；
# 相对 include 路径在 test/quapi 镜像树内原样解析）。外包 module 恢复
# performance「每脚本独立进程」的命名隔离，使本文件自包含、可任意顺序
# include；断言只经模块限定名访问求解器。与原件同步是人工责任——基准数值
# 由 archive_regression.jl 运行原件逐字节兜底，拷贝漂移则阶数断言自暴露。
# =============================================================================
module SecondOrderBosonsRealtime
# =============================================================================
# secondorder/bosons/realtime.jl —— U1GTEMPO2 "冻结核不变 + K 侧二阶"的实时复现
# （玻色 Toulouse，A(t) = [e^{-iHt}]₁₁），并输出画图数据 results/realtime.csv
#
# 方案（隐式梯形，冻结核 = bosonic_Δt 条目原样、face-value cell 配对）：
#   A_j − u·A_{j−1} + f·Σ_{k<j} Δ[j,k]·φ_k + f·d₀·φ_j = 0,  A₀ = 1
#   u = e^{−iϵ_d h}；f = e^{−iϵ_d h/2}（半步相位）；φ_k = (A_{k−1}+A_k)/2（Weyl ½
#   源平均）；Δ = −(Δ⁺⁺+Δ⁺⁻)（推迟 cell 对积分）；d₀ = 等时三角（接触全量对角）。
# 机制：与严格 Strang 链相差 O(h²)，从而继承其二阶。
#
# 运行：JULIA_PROBE_LIBSTDCXX=0 julia --startup-file=no --project=. secondorder/bosons/realtime.jl
# =============================================================================
include(joinpath(@__DIR__, "..", "..", "common", "common.jl"))

"""
    weyl_k_realtime(mdl, N, T) -> (A[1..N], h)

实时 K 侧二阶（Weyl-K 隐式梯形）。`A[m] ≈ A(t_m)`。
"""
function weyl_k_realtime(mdl, N, T)
    h = T / N
    u = exp(-im * mdl.ϵd * h)
    f = exp(-im * mdl.ϵd * h / 2)
    corr = bosonic_Δt(mdl.bath.spectrum; β=mdl.β, N=N, t=T, μ=mdl.μ)
    P = branch(corr; b1=:+, b2=:+)
    X = branch(corr; b1=:+, b2=:-)
    d0 = -(P[1, 1] + X[1, 1])                   # 等时接触三角
    A = zeros(ComplexF64, N + 1)
    A[1] = 1.0
    for m in 1:N
        s = 0.0 + 0.0im
        for k in 1:m-1
            s += -(P[m+1, k+1] + X[m+1, k+1]) * (A[k] + A[k+1]) / 2
        end
        A[m+1] = (u * A[m] - f * s - f * d0 * A[m] / 2) / (1 + f * d0 / 2)
    end
    return A[2:end], h
end

# ------------------------------------------------------------------ #

end  # module

@testset "bosonic_Δt：Weyl-K 推迟递推（二阶）" begin
    mdl = SecondOrderBosonsRealtime.boson_toulouse()
    Aex = SecondOrderBosonsRealtime.exact_return_amplitude(mdl)
    errs = Float64[]
    for N in ORDER_NS
        A, h = SecondOrderBosonsRealtime.weyl_k_realtime(mdl, N, 2.0)  # A[m] ↔ t_m
        push!(errs, maximum(abs(A[m] - Aex(m * h)) for m in frac_nodes(N)))
    end
    assert_order(errs, 1.80, 2.20)
end
