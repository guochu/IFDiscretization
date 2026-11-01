# =============================================================================
# exact —— 玻色浴严格装配（每模闭式）
#
# ← QuAPI/exactif/test/exact_bosonic_api.jl 原样移植（kernel 未动）。
#
# 费米 exactif 严格装配的玻色对应：玻色浴无 Nambu 结构（对角谐振子库），
# 严格性落在**每模时间域闭式**上——GL-M 节点 (ω_k, w_k) 上
#     greater G_k>(t) = (1+n_k)e^{−iω_k t}·w_k     （n_k = boseeinstein(β,μ,ω_k)）
#     lesser  G_k<(t) = n_k e^{−iω_k t}·w_k
#     虚时   G_k(τ>0) = (1+n_k)e^{−ω_kτ}·w_k；玻色 KMS **周期**延拓
#             G(τ≤0) = +G(β+τ)（区别于费米的反周期 −）
#     混合   G₊τ(a,i) = Σ_k w_k(1+n_k)e^{−ω_kτᵢ}e^{−iω_kt_a}（有界闭式），
#             Gτ₊ = lesser 分支 Σ_k w_k n_k e^{−ω_k(β−τᵢ)}e^{−iω_kt_a}
# 谱求和 = GL-M 求积（M→∞ 收敛连续谱）；时间层机器精度（点值闭式）。
# μ 语义（保留两条路径——玻色 μ≠0 包内在用）：Δτ = spectrumshift
# （g 内 μ=0，要求 ω_k>0）；Δt/Δm = 热因子 boseeinstein(β,μ,ω_k)（要求
# ω_k>μ）。约束：谱带不得触及 ε=0（μ=0 的 1/ε 红外奇异）。
# 注：条目为点值闭式（exactif 族约定），与冻结核的梯形积分核**不同对象**
# （同 exactif 与 fermionic_* 冻结核的关系）。
#
# 接口调整（rev 4）：`M::Int=64` → `gfsolver::ToulouseGreenFunctionSolver
# = EigenSolver(64)`，节点数取 `_nbath(gfsolver)`；同 gfsolver 下与原版
# 逐位一致。
# =============================================================================

"GL-M 玻色模式：(ω_k, w_k = half·wGL·f(ω_k))——谱权重进求积权重。"
function _boson_modes(f::AbstractBoundedFunction, μ::Real, M::Int)
    lb, ub = lowerbound(f), upperbound(f)
    xs, ws = gauss_legendre(M)
    half = (ub - lb) / 2
    mid = (lb + ub) / 2
    ωs = [mid + half * x for x in xs]
    ws = [half * w * f(ω) for (ω, w) in zip(ωs, ws)]
    return ωs, ws
end

"""
    exact_bosonic_Δτ(f; β, N, μ=0, δτ=β/N, gfsolver=EigenSolver(64)) -> ImagCorrelationFunction

严格装配虚时核（点值 + 玻色 KMS 周期延拓；spectrumshift μ 路径）。
`gfsolver` 提供格林函数求解器（`EigenSolver(nbath)` 的 nbath = GL-M 节点数）。
"""
function exact_bosonic_Δτ(f0::AbstractBoundedFunction; β::Real, N::Int,
                          μ::Real=0, δτ::Real=β / N,
                          gfsolver::ToulouseGreenFunctionSolver=EigenSolver(64))
    β = convert(Float64, β)
    f = spectrumshift(f0, μ)
    ωs, ws = _boson_modes(f, 0.0, _nbath(gfsolver))
    G(τ) = sum(w * (1 + boseeinstein(β, 0, ω)) * exp(-ω * τ) for (ω, w) in zip(ωs, ws))
    ηⱼₖ = [G((d - 1) * δτ) for d in 1:N]            # d=1 → G(0⁺)
    ηₖⱼ = [G(β - (d - 1) * δτ) for d in 1:N]        # 周期延拓：G(−τ) = G(β−τ)
    return ImagCorrelationFunction(CorrelationMatrix(ηⱼₖ, ηₖⱼ))
end

"""
    exact_bosonic_Δt(f; β, N, t, μ=0, gfsolver=EigenSolver(64)) -> RealCorrelationFunction

严格装配实时核：greater/lesser 点值。存储对 = (G₊₋ = CM(−conj(G<), −G>),
G₋₊ = CM(−G>, −conj(G<)))——玻色交叉支的 ±conj 约定（玻色谱恒实）。
"""
function exact_bosonic_Δt(f::AbstractBoundedFunction; β::Real, N::Int, t::Real,
                          μ::Real=0, gfsolver::ToulouseGreenFunctionSolver=EigenSolver(64))
    β = convert(Float64, β)
    μ = convert(Float64, μ)
    δt = t / N
    ωs, ws = _boson_modes(f, μ, _nbath(gfsolver))
    Gt(τ, which) = sum(w * (which === :g ? (1 + boseeinstein(β, μ, ω)) :
                                 boseeinstein(β, μ, ω)) * exp(-im * ω * τ)
                       for (ω, w) in zip(ωs, ws))
    Ggr = [Gt(i * δt, :g) for i in 0:N]
    Gls = [Gt(i * δt, :l) for i in 0:N]
    return RealCorrelationFunction(CorrelationMatrix(.-conj.(Gls), .-Ggr),
                                   CorrelationMatrix(.-Ggr, .-conj.(Gls)))
end

"""
    exact_bosonic_Δm(f; β, Nτ, t, Nt, μ=0, δτ=β/Nτ, gfsolver=EigenSolver(64)) -> MixedCorrelationFunction

严格装配混合核：ττ = Δτ 环；Keldysh 四向量（同 Δt 布局）；τ-交叉闭式
G₊τ[a+1,i] = Σ w(1+n)e^{−ωτ}e^{−iωt}、Gτ₊ = lesser 分支（有界）。
"""
function exact_bosonic_Δm(f::AbstractBoundedFunction; β::Real, Nτ::Int, t::Real,
                          Nt::Int, μ::Real=0, δτ::Real=β / Nτ,
                          gfsolver::ToulouseGreenFunctionSolver=EigenSolver(64))
    β = convert(Float64, β)
    μ = convert(Float64, μ)
    δt = t / Nt
    ωs, ws = _boson_modes(f, μ, _nbath(gfsolver))
    Gt(τ, which) = sum(w * (which === :g ? (1 + boseeinstein(β, μ, ω)) :
                                 boseeinstein(β, μ, ω)) * exp(-im * ω * τ)
                       for (ω, w) in zip(ωs, ws))
    Ggr = [Gt(i * δt, :g) for i in 0:Nt]
    Gls = [Gt(i * δt, :l) for i in 0:Nt]
    G₊τ = [sum(w * (1 + boseeinstein(β, μ, ω)) * exp(-ω * ((i - 1) * δτ)) *
               exp(-im * ω * (a * δt)) for (ω, w) in zip(ωs, ws))
           for a in 0:Nt, i in 1:Nτ]
    Gτ₊ = [sum(w * boseeinstein(β, μ, ω) * exp(-ω * (β - (i - 1) * δτ)) *
               exp(-im * ω * (a * δt)) for (ω, w) in zip(ωs, ws))
           for i in 1:Nτ, a in 0:Nt]
    Gτ(τ) = sum(w * (1 + boseeinstein(β, μ, ω)) * exp(-ω * τ) for (ω, w) in zip(ωs, ws))
    ηⱼₖ = [Gτ((d - 1) * δτ) for d in 1:Nτ]
    ηₖⱼ = [Gτ(β - (d - 1) * δτ) for d in 1:Nτ]
    return MixedCorrelationFunction(CorrelationMatrix(.-conj.(Gls), .-Ggr),
                                    CorrelationMatrix(.-Ggr, .-conj.(Gls)),
                                    G₊τ, Gτ₊,
                                    CorrelationMatrix(ComplexF64.(ηⱼₖ),
                                                      ComplexF64.(ηₖⱼ)))
end
