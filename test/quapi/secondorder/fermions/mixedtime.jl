# =============================================================================
# test/quapi/secondorder/fermions/mixedtime.jl —— fermionic_Δm + K/IF 拆分（weyl_k_mixed）⇒ 全局二阶
#
# 求解器快照拷贝自 performance/secondorder/fermions/mixedtime.jl（main 与 PROGRAM_FILE 守卫已删，其余逐字未动；include 行改直连 common；
# 相对 include 路径在 test/quapi 镜像树内原样解析）。外包 module 恢复
# performance「每脚本独立进程」的命名隔离，使本文件自包含、可任意顺序
# include；断言只经模块限定名访问求解器。与原件同步是人工责任——基准数值
# 由 archive_regression.jl 运行原件逐字节兜底，拷贝漂移则阶数断言自暴露。
# =============================================================================
module SecondOrderFermionsMixedtime
# =============================================================================
# secondorder/fermions/mixedtime.jl —— 费米混合轮廓（L 形 Kadanoff-Baym）
# 全平衡 G>(t)：全 fermionic_Δm 冻结核 + K 侧直接二阶。输出 results/mixedtime.csv
#
# 参照（U1GTEMPO2 混合时间轴，本机实测，单模 β=2、T=0.8）：order=1 对全热
# 平衡 ED 阶 0.98、order=2 阶 2.00——混合目标 = τ 环热化的全平衡 G>（含 γ
# 非对角初关联），二阶由 K 侧修正达成、IF 不变。本文件在 QuAPI 冻结核框架
# 内实现同目标、同阶数。
#
# ============================ fermionic_Δm 块使用表 ============================
# 全部数据取自单次 fermionic_Δm 调用的三个块（+ 同数据的闭式权重）：
#   ① ξ 块（τ,τ）——τ 支环方程（weyl_k_imagtime 结构：梯形 φ、半步相位 f、
#      接触对角、反周期跳变 G₀ = −1−G(β⁻) 完整代入）→ 环解 G(τ_j)（二阶）；
#      n_d = −G(β⁻)（KMS）；γ[1,k] = 消峰 Boltzmann 收缩（见下，二阶）；
#   ② η 块（+,+）（+,−）——实时支 Weyl-K 递推的推迟组合。注意 Δm 约定：
#      branch(:+,:−) = −(Δt 的同块）⟹ 推迟组合 = −(P+X)（等价于 Δt 的 X−P，
#      数值已对拍：A 通道对 Aex 二阶一致）；
#   ③ ζ 块（+,τ）——实时支 C 通道的**全历史 Duhamel 源的构成因子**：
#      ζ⁺τ 每模严格 rank-1（单模 SVD σ₂/σ₁ = 9e-17）⟹ 实时侧因子 =
#      I_k(m)·e^{∓相位}（cell 对积分闭式，进入源），τ 侧因子 = Boltzmann
#      权重（进入 γ[1,k] 收缩）。即 ζ 块的物理内容（混合交连）以因式分解
#      形式完整进入源与初关联，而非 O(N²) 数值收缩。
#      恒等式数值证据：Σ_j ζ[m,j]·w_j（局部收缩）= γ[1,k]·I_k(m)·(−iV_k)
#      对全部 m 成立（比值常数、虚部波动 0.0）——因此 C 通道 = 同一条 Weyl-K
#      递推（η 块核，同 A 通道）+ 局部 Duhamel 源（ζ 块实时侧因子 × γ 行）。
#      注意递推配源的组合必须自洽：局部源配 η 块 Δ 递推（本文件，二阶）；
#      全历史源 J_k 配 𝒦 单重卷积递推（等价形式）。混配（局部源 + 𝒦，或全历
#      史源 + Δ）不自洽。
#
# ============================ 为什么 ζ 不能直接收缩 ============================
# bath 起点振幅 [U]₁ₐ 的正确记忆方程核是**单重卷积** 𝒦(D) =
# Σ_k V_k²e^{−iϵ_dD}(e^{iω_kD}−1)/(iω_k)（bath 起点振幅载波为 ε_a；杂质自
# 能核 Δ 是二重 cell 积分、载波 ϵ_d，不适用——实测：ζ 收缩驱动 Δ 递推 =
# 平地板/发散，𝒦 递推 + 全历史源 = 直接二阶）。C 通道源的全历史 Duhamel
# 积分 J_k(m) 与 γ[1,k] 收缩分别正是 ζ 块因式分解的实时侧因子与 τ 侧因子。
#
# 运行：JULIA_PROBE_LIBSTDCXX=0 julia --startup-file=no --project=. secondorder/fermions/mixedtime.jl
# =============================================================================
include(joinpath(@__DIR__, "common.jl"))  # （快照改动）直连 common，不再经 imagtime.jl 传递

"𝒦(D) = Σ_k V_k² e^{−iϵ_dD}(e^{iω_kD}−1)/(iω_k)（bath-only 单重卷积核，闭式）。"
function mathcalK(mdl, D)
    S = 0.0 + 0.0im
    for k in 1:mdl.L
        ω = mdl.ϵd - mdl.ws[k]
        S += mdl.αs[k] * exp(-im*mdl.ϵd*D) *
             (abs(ω) < 1e-10 ? im*D : (exp(im*ω*D) - 1) / (im*ω))
    end
    return S
end

"""
    weyl_k_mixed(mdl, Nτ, Nt, T) -> G>[t_0..t_Nt]

费米混合全 fermionic_Δm K 侧二阶（全平衡 G>）。`G[m+1] ≈ G>(t_m)`。
"""
function weyl_k_mixed(mdl, Nτ, Nt, T)
    h = T / Nt
    δτ = mdl.β / Nτ
    u = exp(-im * mdl.ϵd * h)          # 实时支相位
    f = exp(-im * mdl.ϵd * h / 2)
    uτ = exp(-mdl.ϵd * δτ)             # τ 支相位（实衰减！虚时无 i）
    fτ = exp(-mdl.ϵd * δτ / 2)
    dm = fermionic_Δm(mdl.bath.spectrum; β=mdl.β, Nτ=Nτ, t=T, Nt=Nt, μ=mdl.μ)
    # ① τ 支环（ξ 块；四要素同 imagtime.jl）
    ξ = branch(dm; b1=:τ, b2=:τ)
    ηjk, ηkj = ξ.ηⱼₖ, ξ.ηₖⱼ
    d0τ = ηjk[1] + ηkj[1]
    M = zeros(ComplexF64, Nτ, Nτ); b = zeros(ComplexF64, Nτ)
    for m in 1:Nτ
        M[m, m] += 1.0
        if m ≥ 2; M[m, m-1] -= uτ
        else; b[m] -= uτ; M[m, Nτ] += uτ; end
        for k in 1:Nτ
            w = k < m ? ηjk[m-k+1] : k > m ? ηkj[k-m+1] : d0τ
            if k == 1                       # φ₁ = (−1−G_N+G₁)/2：跳变完整代入
                M[m, Nτ] += fτ*w*0.5
                M[m, 1] -= fτ*w*0.5
                b[m] -= fτ*w*0.5
            else
                M[m, k-1] -= fτ*w*0.5
                M[m, k] -= fτ*w*0.5
            end
        end
    end
    Gτ = M \ b
    n_d = -Gτ[Nτ]
    G0v = -1 - Gτ[Nτ]
    Gext = vcat(G0v, Gτ)
    # γ[1,k] 消峰 Boltzmann 收缩（ζ 块 τ 侧因子的二阶离散）
    γrow = zeros(ComplexF64, mdl.L)
    for k in 1:mdl.L
        ε, V = mdl.ws[k], sqrt(mdl.αs[k])
        W_tot = abs(ε) < 1e-10 ? mdl.β : (1 - exp(-ε*mdl.β))/ε
        acc = 0.0 + 0.0im
        for j in 1:Nτ
            w0 = exp(-ε*(mdl.β - (j-1)*δτ))
            w1 = exp(-ε*(mdl.β - j*δτ))
            acc += δτ * (w0*(Gext[j] - G0v) + w1*(Gext[j+1] - G0v)) / 2
        end
        γrow[k] = V * (G0v * W_tot + acc) / (1 + exp(-ε*mdl.β))
    end
    # ② 实时支 A（η 块；Δm 约定推迟组合 −(P+X)）
    P = branch(dm; b1=:+, b2=:+)
    Xr = branch(dm; b1=:+, b2=:-)
    Xc = branch(dm; b1=:+, b2=:τ)
    Δ(j, k) = -(Xr[j, k] + P[j, k])
    d0 = Δ(1, 1)
    A = zeros(ComplexF64, Nt + 1); A[1] = 1.0
    for m in 1:Nt
        s = 0.0 + 0.0im
        for k in 1:m-1
            s += Δ(m+1, k+1) * (A[k] + A[k+1]) / 2
        end
        A[m+1] = (u*A[m] - f*s - f*d0*A[m]/2) / (1 + f*d0/2)
    end
    # ③ C 通道：同一条 Weyl-K 递推（η 块核）+ 局部 Duhamel 源
    #    S_m = Σ_k(−iV_k)·I_k(m)·(−γrow[k])，I_k(m) = ζ 块实时侧因子的 cell 积分
    #    （相位基准 e^{−iε_a t_{m−1}}·e^{−iϵ_d h}——错误的基准导致一阶病态）
    C = zeros(ComplexF64, Nt + 1)
    for m in 1:Nt
        s = 0.0 + 0.0im
        for k in 1:m-1
            s += Δ(m+1, k+1) * (C[k] + C[k+1]) / 2
        end
        S = 0.0 + 0.0im
        for k in 1:mdl.L
            εa, V = mdl.ws[k], sqrt(mdl.αs[k])
            ωd = mdl.ϵd - εa
            I = exp(-im*εa*(m-1)*h) * exp(-im*mdl.ϵd*h) *
                (abs(ωd) < 1e-10 ? h : (exp(im*ωd*h) - 1) / (im*ωd))
            S += -im * V * I * (-γrow[k])
        end
        C[m+1] = (u*C[m] - f*s - f*d0*C[m]/2 + S) / (1 + f*d0/2)
    end
    return (-im) .* ((1 - n_d) .* A .+ C)
end

# ------------------------------------------------------------------ #

end  # module

@testset "fermionic_Δm：Weyl-K 混合装配（二阶）" begin
    mdl = SecondOrderFermionsMixedtime.fermi_toulouse()
    ev = eigen(Hermitian(mdl.hfull))
    γ = ev.vectors * Diagonal([1 / (exp(mdl.β * λ) + 1) for λ in ev.values]) * ev.vectors'
    gex(t) = -im * (exp(-im * t * mdl.hfull) * (mdl.Id - transpose(γ)))[1, 1]
    errs = Float64[]
    for N in ORDER_NS
        G = SecondOrderFermionsMixedtime.weyl_k_mixed(mdl, N, N, 2.0)  # G[m+1] ↔ G>(t_m)
        push!(errs, maximum(abs(G[m+1] - gex(m * 2.0 / N)) for m in frac_nodes(N)))
    end
    assert_order(errs, 1.80, 2.20)
end
