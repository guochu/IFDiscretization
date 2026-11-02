# =============================================================================
# test/quapi/secondorder/fermions/realtime.jl —— fermionic_Δt + K/IF 拆分（weyl_k_realtime）⇒ 全局二阶
#
# 求解器快照拷贝自 performance/secondorder/fermions/realtime.jl（main 与 PROGRAM_FILE 守卫已删，其余逐字未动；
# 相对 include 路径在 test/quapi 镜像树内原样解析）。外包 module 恢复
# performance「每脚本独立进程」的命名隔离，使本文件自包含、可任意顺序
# include；断言只经模块限定名访问求解器。与原件同步是人工责任——基准数值
# 由 archive_regression.jl 运行原件逐字节兜底，拷贝漂移则阶数断言自暴露。
# =============================================================================
module SecondOrderFermionsRealtime
# =============================================================================
# secondorder/fermions/realtime.jl —— 费米 Toulouse 实时 K 侧二阶（Weyl-K）
# 输出画图数据 results/realtime.csv
#
# 与玻色侧（../bosons/realtime.jl）同骨架，唯核组合不同：
#   玻色推迟核 = −(Δ⁺⁺+Δ⁺₋)（热因子 (1+n)−n 相消）
#   费米推迟核 = −(Δ⁺⁺−Δ⁺₋) = Δ⁺₋−Δ⁺⁺（热因子 (1−n)+n 相消；Pauli 反映射：
#               Δ⁺⁺ = −(1−n_F)R、Δ⁺₋ = +n_F R，单模验尸实测）
#
# 运行：JULIA_PROBE_LIBSTDCXX=0 julia --startup-file=no --project=. secondorder/fermions/realtime.jl
# =============================================================================
include(joinpath(@__DIR__, "common.jl"))

"""
    weyl_k_realtime(mdl, N, T) -> (A[1..N], h)

费米实时 K 侧二阶（Weyl-K 隐式梯形，核 = X−P 推迟组合）。`A[m] ≈ A(t_m)`。
"""
function weyl_k_realtime(mdl, N, T)
    h = T / N
    u = exp(-im * mdl.ϵd * h)
    f = exp(-im * mdl.ϵd * h / 2)
    corr = fermionic_Δt(mdl.bath.spectrum; β=mdl.β, N=N, t=T, μ=mdl.μ)
    P = branch(corr; b1=:+, b2=:+)
    X = branch(corr; b1=:+, b2=:-)
    Δ(j, k) = -(X[j, k] + P[j, k])              # 费米推迟组合（统一约定；≡ 旧 X−P）
    d0 = Δ(1, 1)
    A = zeros(ComplexF64, N + 1)
    A[1] = 1.0
    for m in 1:N
        s = 0.0 + 0.0im
        for k in 1:m-1
            s += Δ(m+1, k+1) * (A[k] + A[k+1]) / 2
        end
        A[m+1] = (u*A[m] - f*s - f*d0*A[m]/2) / (1 + f*d0/2)
    end
    return A[2:end], h
end

# ------------------------------------------------------------------ #

# real_step 需在费米模型上工作（common.jl 的 boson 版本同构）
function real_step(mdl, order::Int, h::Real)
    eA(t) = exp(-im * t * mdl.hS)
    eB(t) = exp(-im * t * (mdl.hb + mdl.hi))
    s2(t) = eA(t / 2) * eB(t) * eA(t / 2)
    order == 1 && return eA(h) * eB(h)
    order == 2 && return s2(h)
    throw(ArgumentError("order 必须为 1/2"))
end

end  # module

@testset "fermionic_Δt：Weyl-K 推迟递推（二阶）" begin
    mdl = SecondOrderFermionsRealtime.fermi_toulouse()
    Aex = SecondOrderFermionsRealtime.exact_return_amplitude(mdl)
    errs = Float64[]
    for N in ORDER_NS
        A, h = SecondOrderFermionsRealtime.weyl_k_realtime(mdl, N, 2.0)  # A[m] ↔ t_m
        push!(errs, maximum(abs(A[m] - Aex(m * h)) for m in frac_nodes(N)))
    end
    assert_order(errs, 1.80, 2.20)
end
