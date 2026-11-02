# =============================================================================
# test/quapi/firstorder/bcs/imagtime.jl —— bcs_Δτ 朴素 Nambu Matsubara 环（bcs_tau_ring_naive）⇒ 全局一阶
#
# 求解器快照拷贝自 performance/firstorder/bcs/imagtime.jl（main 与 PROGRAM_FILE 守卫已删，其余逐字未动；
# 相对 include 路径在 test/quapi 镜像树内原样解析）。外包 module 恢复
# performance「每脚本独立进程」的命名隔离，使本文件自包含、可任意顺序
# include；断言只经模块限定名访问求解器。与原件同步是人工责任——基准数值
# 由 archive_regression.jl 运行原件逐字节兜底，拷贝漂移则阶数断言自暴露。
# =============================================================================
module FirstOrderBCSImagtime
# =============================================================================
# firstorder/bcs/imagtime.jl —— BCS 浴虚时：一阶冻结核 Nambu Matsubara 环
# vs 严格 BdG 环 vs 严格 Trotter 1/2。输出 results/imagtime.csv
#
# 模型与约定与 secondorder/bcs 完全一致（common.jl 复用）。观测对象：杂质
# Nambu 2×2 Matsubara 块 Ĝ(τ_m) = −[e^{−𝓗τ}M₀]_{imp,imp}（M₀ = ⟨ΦΦ†⟩ =
# (1+e^{−β𝓗})⁻¹），行 1 = [greater 𝒢₁, 反常 𝒢₂]（exact_Gτ_ring 同款）。
#
# 环方程（4 分量/节点：(i,j) = (1,1),(1,2),(2,1),(2,2)，行 i = 杂质 Nambu 行）：
#   自由相位（行 i）：uτ_i = e^{∓ϵ_d δτ}，fτ_i = e^{∓ϵ_d δτ/2}（行 2 = 空穴行）；
#   核 𝚺̂ 的 lag 权重（naive 矩形，同正常费米环）：k<m 用 ηⱼₖ lag、k>m 用
#   ηₖⱼ lag、对角 ηⱼₖ[1]+ηₖⱼ[1]；Nambu 槽位
#       𝚺̂ = [cn  −cc ; −nn  nc]        （反常槽位带 −1：IF 约定把负号吸进
#       Δ，Dyson 重排时反常自能多出 −1——符号由 exact_Gτ_ring 裁判扫描锁定）
#   卷积按行混排：[𝚺⊛Ĝ]_{ij} = Σ_k′ 𝚺̂_{i,k′} Ĝ_{k′,j}（行 2 分量参与行 1
#   方程——这是 2 分量切片不封闭的根源）；
#   跳变源只在 Nambu 对角分量：b[(i,i)] = −uτ_i（{Φ_i,Φ_i†}=1；反常
#   (1,2)/(2,1) 无等点反对易子，源为零）。
#   提取：Ĝ_11 = 𝒢₁、Ĝ_12 = 𝒢₂（行 1）。
#
#   Δ=0 退化：𝒢₂ ≡ 0 精确成立（已验证）。
#
# 运行：JULIA_PROBE_LIBSTDCXX=0 julia --startup-file=no --project=. firstorder/bcs/imagtime.jl
# =============================================================================
include(joinpath(@__DIR__, "..", "..", "secondorder", "bcs", "common.jl"))

"""
    bcs_tau_ring_naive(mdl, N) -> (M, b)

BCS Nambu Matsubara 一阶冻结核环（naive 矩形）。未知堆叠 s = 4(m−1)+idx(i,j)。
"""
function bcs_tau_ring_naive(mdl, N)
    δτ = mdl.β / N
    uτ = [exp(-mdl.ϵd * δτ), exp(mdl.ϵd * δτ)]
    fτ = [exp(-mdl.ϵd * δτ / 2), exp(mdl.ϵd * δτ / 2)]
    corr = bcs_Δτ(mdl.bath.spectrum; β=mdl.β, N=N, μ=mdl.μ, Δ=mdl.Δ)
    ηj(X) = X.data.ηⱼₖ
    ηk(X) = X.data.ηₖⱼ
    NJ(l) = [ηj(corr.uu)[l]  -ηj(corr.ud)[l];
             -ηj(corr.du)[l]  ηj(corr.nc)[l]]
    NK(l) = [ηk(corr.uu)[l]  -ηk(corr.ud)[l];
             -ηk(corr.du)[l]  ηk(corr.nc)[l]]
    Nd0 = NJ(1) + NK(1)
    M = zeros(ComplexF64, 4N, 4N)
    b = zeros(ComplexF64, 4N)
    idx(i, j) = 2(i - 1) + j
    for m in 1:N, i in 1:2, j in 1:2
        s = 4 * (m - 1) + idx(i, j)
        M[s, s] += 1.0
        if m ≥ 2
            M[s, 4*(m-2)+idx(i, j)] -= uτ[i]
        else
            M[s, 4*(N-1)+idx(i, j)] += uτ[i]           # 反周期回绕（per Nambu 行）
            i == j && (b[s] = -uτ[i])                  # 跳变源只在 Nambu 对角
        end
        for k in 1:N
            W = k < m ? NJ(m-k+1) : k > m ? NK(k-m+1) : Nd0
            for ip in 1:2                              # [Σ⊛Ĝ]_{ij} = Σ_ip Σ̂_{i,ip} Ĝ_{ip,j}
                M[s, 4*(k-1)+idx(ip, j)] -= fτ[i] * W[i, ip]
            end
        end
    end
    return M, b
end

"""
    strict_chain_bcs_tau(mdl, N, order) -> (Ĝ[1..N], h)

严格 Trotter 虚时链：T = e^{−h hA}e^{−h hB}（Lie）/ eA·eB·eA（Strang），
处置矩阵 P̃ = (I+Tᴺ)⁻¹，Ĝ(τ_m) = −[Tᵐ P̃]_{imp,imp}。
"""
function strict_chain_bcs_tau(mdl, N, order::Int)
    h = mdl.β / N
    𝓗 = bcs_bdg(mdl)
    n = mdl.L + 1
    hA = zeros(ComplexF64, 2n, 2n)
    hA[1, 1] = mdl.ϵd
    hA[n+1, n+1] = -mdl.ϵd
    hB = 𝓗 - hA
    eA = exp(-h / 2 * hA)
    eB = exp(-h * hB)
    T = order == 1 ? exp(-h * hA) * eB : eA * eB * eA
    P = inv(Matrix{ComplexF64}(I, 2n, 2n) + T^N)
    idx = [1, n+1]
    out = Vector{Matrix{ComplexF64}}(undef, N)
    Tm = Matrix{ComplexF64}(I, 2n, 2n)
    for m in 1:N
        Tm = Tm * T
        out[m] = -(Tm * P)[idx, idx]
    end
    return out, h
end

"严格 BdG 环：Ĝ(τ_m) = −[e^{−𝓗τ_m}M₀]_{imp,imp}（2×2 块，机器精度参照）。"
function exact_tau_block(mdl, N)
    V, λ, Vinv = bcs_eig(mdl)
    n = mdl.L + 1
    M0 = bcs_M0(mdl)
    h = mdl.β / N
    idx = [1, n+1]
    return [-(V * (Diagonal(exp.(-λ .* (m * h))) * (Vinv * M0)))[idx, idx] for m in 1:N], h
end

# ------------------------------------------------------------------ #

end  # module

@testset "bcs_Δτ：朴素 Nambu Matsubara 环（一阶）" begin
    mdl = FirstOrderBCSImagtime.bcs_toulouse()
    errs = Float64[]
    for N in ORDER_NS
        M, b = FirstOrderBCSImagtime.bcs_tau_ring_naive(mdl, N)
        G = M \ b
        ex, _ = FirstOrderBCSImagtime.exact_tau_block(mdl, N)   # ex[m] = 杂质 2×2 块
        e = 0.0
        for m in frac_nodes(N), i in 1:2, j in 1:2
            e = max(e, abs(G[4*(m-1) + 2*(i-1) + j] - ex[m][i, j]))
        end
        push!(errs, e)
    end
    assert_order(errs, 0.75, 1.30)
end
