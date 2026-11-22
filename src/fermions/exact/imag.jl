# =============================================================================
# exact/imag.jl —— exact_fermionic_Δτ：虚时反周期环严格 IF 核（谱闭式）
#
# 契约 = 纯耦合项严格核（族契约见 real.jl exact_fermionic_Δt docstring）：
# 杂质 kick 相位线性析出 Δ_IF(x) = x·Δ_IF(1)——x=1 的核即纯耦合项对象，
# Pfaffian 收缩与同阶直接 Trotter 演化逐位一致（imagtime 金标准）。
# =============================================================================
# ------------------------------ 虚时 ------------------------------ #

"""
    exact_fermionic_Δτ(f; β, N, μ=0, δτ=β/N, gfsolver=EDToulouseSolver(64))
        -> (ImagCorrelationFunction, normdelta)

ExactEqualDistantKernel 严格 IF 核（虚时反周期环）——**纯浴对象**（不含任何杂质信息），
元组返回：`(Δ_if, normdelta)`。

**谱闭式构造**（O(K³ + N·K + N²) 时间 / O(K² + N²) 内存；稠密 Schur 参照见
`test/exact/fermionic_imagtime.jl` 的 `dense_schur_Δτ`，O((KN)³)/O((KN)²) 勿用于
大规模）：Kronecker 分离（W 作用在模指标、反周期移位 S 作用在拷贝指标，
Sᴺ = −I）+ 平移不变 ⟹ 拷贝空间 N 点傅里叶块对角化
`(F⊗I)†(−I + kron(W,S))(F⊗I) = ⊕_f (ω_f·W − I)`（``ω_f = e^{i(2f+1)π/N}``）；
Schur 标量化 `Δ_full = (MM⁻¹)[imp,imp]⁻¹` 精确 Toeplitz（反周期频率下
双生成元——这正是核只存 ηⱼₖ/ηₖⱼ 两条向量的原因），由 N 个标量

```math
g_f = \\sum_i a2_i/(\\omega_f\\lambda_i - 1), \\qquad \\mu_f = 1/g_f
```

决定（h₀ 本征分解 W = V·diag(λ)·V†，a2_i = |V[1,i]|² 杂质谱权重；μ 已折进
`_quad_nodes` 的 Es；β 全部编码于 ω_fᴺ = −1 与 δτ = β/N）。核装配 =
Toeplitz 生成元求和（col 取 ω^{−d}、row 取 ω^{+d}，逐 d 复推）+ 骨架剥离
`Δ = Δ_full + I − S`（接触约定 ηₖⱼ[1] = 0；N=1 时 S[1,1] = −1 与对角同格
⟹ 对角 +2）。死模过滤（`wΓ > 0` + K ≥ 1 守卫，f≡0 抛 `ArgumentError`，
与 Δt/Δm 一致）。

严格性依据：杂质 kick 相位 x = e^{−δτ·ε_d} 从严格核线性析出，Δ_IF(x) =
x·Δ_IF(1)；x=1 时各阶劈分 cell 退化为同一纯浴演化 W = e^{−δτ·h₀} ⟹ 核与
order、ε_d 无关。装配契约（金标准，`test/exact/fermionic_imagtime.jl` T1–T3
数值钉死）：B = skeleton(x) + x·Δ_if（骨架 = 测度 −I + 动能 x·ā'ᵢ₊₁a'ᵢ +
反周期闭 −x·ā'₁a'_N），Pfaffian 收缩的 Z/n̄/G 与同阶直接 Trotter 演化逐位
一致（一阶全步踢与二阶 Strang 半步踢同核——对角 H_imp 下踢相位沿迹可对易、
总相位同为 xᴺ）。

**normdelta = det(BB) = [Π_{f,i}(ω_fλ_i − 1)]·[Π_f g_f]**（Schur 恒等式
det(MM) = det(BB)·det(Δ_full) 的闭式；成对树形累乘控舍入；Z 归一的必要
常数项）：`Z = normdelta · Pf(B)`。

存储约定同冻结核：CorrelationMatrix 双向量环（接触约定 ηⱼₖ[1] = 全量
对角，ηₖⱼ[1] = 0）。
"""
function exact_fermionic_Δτ(f::AbstractBoundedFunction; β::Real, N::Int,
                            μ::Real=0, δτ::Real=β / N,
                            gfsolver::ToulouseSolver=EDToulouseSolver(64))
    M = _nbath(gfsolver)
    Es, Vsq, wΓ = _quad_nodes(f, μ, M)
    keep = wΓ .> 0                      # 剔除谱为零区域的节点（死模不改核）
    εs = Es[keep]
    Vs = Vsq[keep]
    K = length(εs)
    K >= 1 || throw(ArgumentError("谱在支撑上的积分为零（或非正）"))
    n = K + 1
    h0 = zeros(ComplexF64, n, n)
    for k in 1:K
        h0[1, 1+k] = Vs[k]; h0[1+k, 1] = conj(Vs[k]); h0[1+k, 1+k] = εs[k]
    end
    ev = eigen(Hermitian(h0))
    λ = exp.(-δτ .* ev.values)                    # W = V·diag(λ)·V† 的谱
    a2 = abs2.(ev.vectors[1, :])                  # 杂质谱权重
    ωs = [cis((2f + 1) * π / N) for f in 0:N-1]   # 反周期移位谱：ω_f^N = −1
    g = [sum(a2[i] / (ωs[f] * λ[i] - 1) for i in eachindex(λ))
         for f in eachindex(ωs)]                  # 环格林函数频点采样
    # Δ_full 循环取向（F[k,f] = ω_f^{−(k−1)}/√N）：col = (1/N)Σ μ ω^{−d}、row = ω^{+d}，
    # 逐 d 复推（勿用 ω^d 逐次幂，漂移随 d 累积）
    col = zeros(ComplexF64, N)
    row = zeros(ComplexF64, N)
    for f in eachindex(ωs)
        μf = 1 / g[f]
        ωd = one(ComplexF64)
        for d in 0:N-1
            col[d+1] += μf * conj(ωd)
            row[d+1] += μf * ωd
            ωd *= ωs[f]
        end
    end
    col ./= N
    row ./= N
    # 骨架剥离 Δ = Δ_full + I − S：对角 +1（−S 对角零，N=1 时 S[1,1] = −1 ⟹ +2）、
    # 次对角 −1（S[k,k−1]）、(1,N) 反周期闭 +1（S[1,N] = −1）
    ηⱼₖ = Vector{ComplexF64}(undef, N)
    ηⱼₖ[1] = col[1] + (N == 1 ? 2.0 : 1.0)
    for d in 1:N-1
        ηⱼₖ[d+1] = col[d+1] - (d == 1 ? 1.0 : 0.0)
    end
    ηₖⱼ = vcat([zero(ComplexF64)],
                [row[d+1] + (d == N - 1 ? 1.0 : 0.0) for d in 1:N-1])
    # normdelta：det(MM)/det(Δ_full) 闭式；因子成对树形累乘控舍入（内联，无新增符号）
    facs = ComplexF64[ω * λi - 1 for ω in ωs for λi in λ]
    while length(facs) > 1
        half = length(facs) ÷ 2
        nxt = [facs[2j - 1] * facs[2j] for j in 1:half]
        2 * half < length(facs) && push!(nxt, facs[end])
        facs = nxt
    end
    # 必要常数项：纯浴环浴块行列式，装配 Z = normdelta · Pf(B)（金标准契约）
    return ImagCorrelationFunction(CorrelationMatrix(ηⱼₖ, ηₖⱼ)), facs[1] * prod(g)
end
