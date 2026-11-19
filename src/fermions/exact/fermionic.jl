# =============================================================================
# exact —— 费米浴严格 IF 核（ExactEqualDistantKernel）
#
# 纯浴对象契约（2026-11-18 换代）：三个入口签名与冻结核 fermionic_Δτ/Δt/Δm
# 一致，不含任何杂质信息（无 ε_d、无劈分阶数）——杂质 kick 相位经恒等式
# Δ_IF(x) = x·Δ_IF(1)（虚时，机器精度）或显式 kicks 接口（实时/混合，带踢
# Schur）析出到装配侧系统骨架，下游换核是机械替换。三轮廓装配级契约：
#   Δτ：反周期环单粒子矩阵 Schur（fermionic_imagtime.jl 金标准）；
#   Δt ：Keldysh 开线带踢 Schur（fermionic_realtime.jl 金标准，Z ≡ 1 机器）；
#   Δm ：Kadanoff–Baym 单环带踢 Schur（fermionic_mixedtime.jl 金标准，
#        Keldysh 段算符层面精确求和相消 ⟹ Z = τ 环乘积的精确迹）。
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

# ------------------------------ 实时 ------------------------------ #

"腿位（节点 j0 = 0..N；+ 支在前，支内 (a, ā) 相邻），1-based"
_exrt_legpos(j0::Int, br::Symbol, ā::Bool) = 4 * j0 + (br === :+ ? 0 : 2) + (ā ? 2 : 1)

"""
    exact_fermionic_Δt(f; β, N, t, μ=0, gfsolver=EDToulouseSolver(64))
        -> (ExactRealCorrelationFunction, normdelta)

ExactEqualDistantKernel 严格 IF 核（实时 Keldysh 轮廓），元组返回 `(核, 归一常数)`——
`exact_fermionic_Δt_cells` 的均匀网格便捷封装（cells = fill(t/N, N)，kicks = 1）。

装配契约（金标准，`test/exact/fermionic_realtime.jl`）：紧凑 (ā,a) 装配 =
skeleton（测度 −āa·(−1) + 轮廓闭合边 (ā⁻_N,a⁺_N)·(−1) + 真空 ρ₀ 常数）
+ data.Δ，**Z = normdelta · det(装配矩阵)**。kicks = 1（默认，纯浴）时
ε_d 的踢相位只在骨架链条目上（+ 支链 ×x、− 支链 ×x̄，x = e^{−iαε_d}）——
顶点移位离散 Γ(W+(x−1)P₁)^N：ε_d = 0 时 Z 恒 1（迹守恒）机器精确；
ε_d ≠ 0 时 Z 为实数但偏离 1 O(δt)（cell 复合矩阵非幺正的方案误差）。
**算符忠实装配**（Z ≡ 1 机器精确、观测量 = 门序列 Γ(D(x)·W) 的精确收缩）
须走 `exact_fermionic_Δt_cells` 的 `kicks` 接口（补零 pair 网格 + 带踢
Schur，同 GTEMPO NativeExact `exactif_pipeline` 的证明配方）。条目为
**有向指数系数**（无 conj 翻转）。
"""
function exact_fermionic_Δt(f::AbstractBoundedFunction; β::Real, N::Int, t::Real,
                            μ::Real=0, gfsolver::ToulouseSolver=EDToulouseSolver(64))
    return exact_fermionic_Δt_cells(f; β=β, cells=fill(Float64(t) / N, N), μ=μ,
                                    gfsolver=gfsolver)
end

"""
    exact_fermionic_Δt_cells(f; β, cells, μ=0, gfsolver=EDToulouseSolver(64), kicks)
        -> (ExactRealCorrelationFunction, normdelta)

实时 Keldysh 严格核的逐 cell 调度版：`cells[k]` = 第 k 个 Keldysh cell 的演化
长度（可取 0/负），`kicks[k]` = 第 k 个 gap 的杂质踢相位 x_k = e^{−iα_k·ε_d}
（默认全 1 = 纯浴对象）。腿点数 = length(cells)+1（节点 0..N，腿基同冻结核
冻结核实时序）。

构造（← QuAPI/src/exactfermionic.jl `exact_fermionic_Δt_cells` 移植 + 带踢
复合）：gap 复合单粒子矩阵 + 支 = D(x_k)·W_k（杂质输出行 ×x_k），− 支 = 其
伴随 W_k†D†（杂质输入列 ×x̄_k）；W = e^{−i·cell·h}（星形耦合，杂质槽 ε=0）
前向/伴随插入，bath 侧闭合边（(+,N)→(−,N) 费米迹负号）与热边
(+,0)→(−,0) n̄ₖ(1+rγ̄γ) 进 BB，Schur 消 bath 腿后减单位骨架 C0 得 IF-only Δ。

**normdelta = Πₖ n̄ₖ · det(BB)**（必要常数项）。装配契约（金标准
`test/exact/fermionic_realtime.jl`）：紧凑 (ā,a) 装配 = 测度 −1 + 单位链 +
闭合边 (ā⁻_N,a⁺_N)·(−1) + 真空 ρ₀ ⟹ Z = normdelta·det(装配)；
**n̄_vac = Z·(1 − A⁻¹[row(ā⁻_N), col(a⁺_N)])**（(order,N) ∈ {1,2}×{2,3}
四点 machine 钉定）。**kicks 为幺正门相位（|x_k|=1）时全因子幺正 ⟹
Z ≡ 1 机器精确（任意 ε_d、任意调度）**，观测量 = 门序列 Γ(D(x)·W) 的精确
收缩（补零 pair 网格调度下即 Strang 门链的机器精确实现：order 2 δt²、
order 1 δt¹，两阶真实分离）。BB 消元走块结构求解（O(N·K²·Limp + K³)，
稠密参照留在 `_exrt_strict_kernel_dense`）。
"""
function exact_fermionic_Δt_cells(f::AbstractBoundedFunction;
                                  β::Real, cells::AbstractVector{<:Real},
                                  μ::Real=0,
                                  gfsolver::ToulouseSolver=EDToulouseSolver(64),
                                  kicks::AbstractVector{<:Number}=ones(length(cells)))
    length(kicks) == length(cells) ||
        throw(DimensionMismatch("kicks 须与 cells 等长（每 gap 一个踢相位）"))
    return _exrt_strict_kernel(f, collect(Float64.(cells)), Float64(β), μ,
                               _nbath(gfsolver), ComplexF64.(kicks))
end

function _exrt_strict_kernel(f::AbstractBoundedFunction, cells::Vector{Float64},
                             β::Float64, μ::Real, M::Int,
                             kicks::Vector{ComplexF64}=ones(length(cells)))
    N = length(cells)
    N >= 1 || throw(ArgumentError("cell schedule must carry at least one cell"))
    Es, Vsq, wΓ = _quad_nodes(f, μ, M)
    keep = wΓ .> 0                      # 剔除谱为零区域的节点（死模不改配分）
    εs = Es[keep]
    Vs = Vsq[keep]
    K = length(εs)
    K >= 1 || throw(ArgumentError("谱在支撑上的积分为零（或非正）"))
    # bath+耦合 cell（杂质槽 ε=0）+ 带踢复合：+ 支 = D(x)·W（杂质输出行 ×x）、
    # − 支 = 伴随 W†D†（杂质输入列 ×x̄）
    _W(step) = begin
        h = zeros(ComplexF64, K + 1, K + 1)
        for k in 1:K
            h[1, k+1] = h[k+1, 1] = Vs[k]
            h[k+1, k+1] = εs[k]
        end
        exp(-1.0im * step .* h)
    end
    Ws = [_W(cell) for cell in cells]
    Limp = 4 * (N + 1)                  # 杂质腿（ā 行 = a 列）
    Nb = K * (2 * N + 2)                # bath 配对维数（每模式 2(N+1) 个 (γ̄,γ) 槽）
    bidx(k, br, j0) = (k - 1) * (2N + 2) + (br === :+ ? j0 : N + 1 + j0) + 1
    II = zeros(ComplexF64, Limp, Limp)
    IB = zeros(ComplexF64, Limp, Nb)
    BI = zeros(ComplexF64, Nb, Limp)
    for j0 in 0:N, br in (:+, :-)       # Berezin 测度 −āa
        II[_exrt_legpos(j0, br, true), _exrt_legpos(j0, br, false)] -= 1
    end
    for cell in 1:N
        W = Ws[cell]
        x = kicks[cell]
        # + 支：D(x)·W——杂质输出行 ×x
        II[_exrt_legpos(cell, :+, true), _exrt_legpos(cell - 1, :+, false)] += x * W[1, 1]
        for km in 1:K
            IB[_exrt_legpos(cell, :+, true), bidx(km, :+, cell - 1)] += x * W[1, km+1]
            BI[bidx(km, :+, cell), _exrt_legpos(cell - 1, :+, false)] += W[km+1, 1]
        end
        # − 支：伴随 W†D†——conj-转置 + 杂质输入列 ×x̄
        xc = conj(x)
        II[_exrt_legpos(cell - 1, :-, true), _exrt_legpos(cell, :-, false)] +=
            xc * conj(W[1, 1])
        for km in 1:K
            IB[_exrt_legpos(cell - 1, :-, true), bidx(km, :-, cell)] += conj(W[km+1, 1])
            BI[bidx(km, :-, cell - 1), _exrt_legpos(cell, :-, false)] +=
                xc * conj(W[1, km+1])
        end
    end
    # bath 闭合边（费米迹负号）与热边只进 BB——结构化求解吸收
    θ = [let nk = 1 / (exp(β * εs[k]) + 1); nk / (1 - nk) end for k in 1:K]
    Wb = [Ws[c][2:end, 2:end] for c in 1:N]        # 每 cell 的 bath K×K 块
    X, detBB = _exrt_schur_structured(Wb, θ, BI, K, N)
    Mt = II - IB * X
    C0 = zeros(ComplexF64, Limp, Limp)  # 单位动能骨架（x = 1）：测度 + 单位动能
    for j0 in 0:N, br in (:+, :-)
        C0[_exrt_legpos(j0, br, true), _exrt_legpos(j0, br, false)] -= 1
    end
    for cell in 1:N
        C0[_exrt_legpos(cell, :+, true), _exrt_legpos(cell - 1, :+, false)] += 1
        C0[_exrt_legpos(cell - 1, :-, true), _exrt_legpos(cell, :-, false)] += 1
    end
    Δ = Mt - C0
    basis = [GrassmannContourIndex(j0 + 1, br, 1, c)
             for j0 in 0:N for br in (:+, :-) for c in (false, true)]
    nbar = [1 / (exp(β * ε) + 1) for ε in εs]
    return _strict_real_from_legpos(basis, Δ), prod(nbar) * detBB
end

"""
BB 的块结构求解（分配函数内部；块行/列 = (branch, node)，每块 K×K）：

    BB = [B₊  C₀]   B₊ = 下双对角：diag −I、(c, c−1) 块 = Wb[c]
        [C₁  B₋]   B₋ = 上双对角：diag −I、(c−1, c) 块 = adjoint(Wb[c])
                   C₀[(+,0),(−,0)] = diag(θ)（热边 n̄/(1−n̄)）
                   C₁[(−,N),(+,N)] = −I（费米迹闭合边）

`BB·X = BI` 经仿射前推求解（P := X₊[N] 为参数：− 支后推、node-0 耦合、
+ 支前推，(I − A₊[N])·P = b₊[N] 一次 K×K 解后回代）；
**det(BB) = det(I_K − Φ·diag(θ)·Ψ)**，Φ = Wb[N]⋯Wb[1]（+ 支单融）、
Ψ = adj(Wb[1])⋯adj(Wb[N])（− 支逆单融）——两条双对角块的 (−1)^{K(N+1)}
因子相消。复杂度 O(N·K²·Limp + K³)，稠密路径 O((KN)³)。
"""
function _exrt_schur_structured(Wb::Vector{Matrix{ComplexF64}}, θ::Vector{Float64},
                                BI::Matrix{ComplexF64}, K::Int, N::Int)
    L = size(BI, 2)
    Θ = Diagonal(ComplexF64.(θ))
    stride = N + 1                      # bidx 的 +/− 支块行距
    Yp(j) = view(BI, (0:K-1) .* (2N+2) .+ (j+1), :)          # (+, j) 块行
    Ym(j) = view(BI, (0:K-1) .* (2N+2) .+ (stride+j+1), :)   # (−, j) 块行
    # − 支仿射后推：X₋[g] = A₋[g]·P + b₋[g]，X₋[N] = −Y₋[N] − P
    A₋ = Vector{Matrix{ComplexF64}}(undef, N + 1)
    b₋ = Vector{Matrix{ComplexF64}}(undef, N + 1)
    A₋[N+1] = -Matrix{ComplexF64}(I, K, K)
    b₋[N+1] = -Matrix{ComplexF64}(Ym(N))
    for c in N:-1:1                     # cell c：块 ((c−1), c) = adj(Wb[c])
        Cadj = adjoint(Wb[c])
        A₋[c] = Cadj * A₋[c+1]
        b₋[c] = Cadj * b₋[c+1] - Matrix{ComplexF64}(Ym(c - 1))
    end
    # node-0 热角耦合 + + 支前推
    A₊0 = Θ * A₋[1]
    b₊0 = Θ * b₋[1] - Matrix{ComplexF64}(Yp(0))
    A₊ = Vector{Matrix{ComplexF64}}(undef, N + 1)
    b₊ = Vector{Matrix{ComplexF64}}(undef, N + 1)
    A₊[1] = A₊0
    b₊[1] = b₊0
    for s in 2:(N + 1)                  # 槽位 s ↔ 节点 s−1；节点 s−1 行用 cell s−1
        A₊[s] = Wb[s-1] * A₊[s-1]
        b₊[s] = Wb[s-1] * b₊[s-1] - Matrix{ComplexF64}(Yp(s - 1))
    end
    # 一致性：(I − A₊[N])·P = b₊[N]
    P = (Matrix{ComplexF64}(I, K, K) - A₊[N+1]) \ b₊[N+1]
    # 回代装配 X（BI 的行布局）
    X = Matrix{ComplexF64}(undef, size(BI)...)
    Xm = [Matrix{ComplexF64}(Ym(N))]    # X₋ 节点缓存
    x₋ = -Matrix{ComplexF64}(Ym(N)) - P
    X[(0:K-1) .* (2N+2) .+ (stride+N+1), :] .= x₋
    for c in N:-1:1
        x₋ = adjoint(Wb[c]) * x₋ - Matrix{ComplexF64}(Ym(c - 1))
        X[(0:K-1) .* (2N+2) .+ (stride+c), :] .= x₋
    end
    x₊ = Θ * x₋ - Matrix{ComplexF64}(Yp(0))
    X[(0:K-1) .* (2N+2) .+ 1, :] .= x₊
    for s in 2:(N + 1)
        x₊ = Wb[s-1] * x₊ - Matrix{ComplexF64}(Yp(s - 1))
        X[(0:K-1) .* (2N+2) .+ s, :] .= x₊
    end
    # det(BB) = det(I_K − Φ·Θ·Ψ)
    Φ = Matrix{ComplexF64}(I, K, K)
    Ψ = Matrix{ComplexF64}(I, K, K)
    for c in 1:N
        Φ = Wb[c] * Φ
        Ψ = Ψ * adjoint(Wb[c])
    end
    detBB = det(Matrix{ComplexF64}(I, K, K) + Φ * Θ * Ψ)
    return X, detBB
end

# 稠密参照路径（机器等价性检验用；O((KN)³) 时间 + O((KN)²) 内存，勿用于大规模）
function _exrt_strict_kernel_dense(f::AbstractBoundedFunction, cells::Vector{Float64},
                                   β::Float64, μ::Real, M::Int,
                                   kicks::Vector{ComplexF64}=ones(length(cells)))
    N = length(cells)
    Es, Vsq, wΓ = _quad_nodes(f, μ, M)
    keep = wΓ .> 0
    εs = Es[keep]
    Vs = Vsq[keep]
    K = length(εs)
    _W(step) = begin
        h = zeros(ComplexF64, K + 1, K + 1)
        for k in 1:K
            h[1, k+1] = h[k+1, 1] = Vs[k]
            h[k+1, k+1] = εs[k]
        end
        exp(-1.0im * step .* h)
    end
    Ws = [_W(cell) for cell in cells]
    Limp = 4 * (N + 1)
    Nb = K * (2 * N + 2)
    bidx(k, br, j0) = (k - 1) * (2N + 2) + (br === :+ ? j0 : N + 1 + j0) + 1
    II = zeros(ComplexF64, Limp, Limp)
    IB = zeros(ComplexF64, Limp, Nb)
    BI = zeros(ComplexF64, Nb, Limp)
    BB = zeros(ComplexF64, Nb, Nb)
    for j0 in 0:N, br in (:+, :-)
        II[_exrt_legpos(j0, br, true), _exrt_legpos(j0, br, false)] -= 1
        for k in 1:K
            BB[bidx(k, br, j0), bidx(k, br, j0)] -= 1
        end
    end
    for cell in 1:N
        W = Ws[cell]
        x = kicks[cell]
        II[_exrt_legpos(cell, :+, true), _exrt_legpos(cell - 1, :+, false)] += x * W[1, 1]
        for km in 1:K
            IB[_exrt_legpos(cell, :+, true), bidx(km, :+, cell - 1)] += x * W[1, km+1]
            BI[bidx(km, :+, cell), _exrt_legpos(cell - 1, :+, false)] += W[km+1, 1]
            for kn in 1:K
                BB[bidx(km, :+, cell), bidx(kn, :+, cell - 1)] += W[km+1, kn+1]
            end
        end
        xc = conj(x)
        II[_exrt_legpos(cell - 1, :-, true), _exrt_legpos(cell, :-, false)] +=
            xc * conj(W[1, 1])
        for km in 1:K
            IB[_exrt_legpos(cell - 1, :-, true), bidx(km, :-, cell)] += conj(W[km+1, 1])
            BI[bidx(km, :-, cell - 1), _exrt_legpos(cell, :-, false)] +=
                xc * conj(W[1, km+1])
            for kn in 1:K
                BB[bidx(km, :-, cell - 1), bidx(kn, :-, cell)] += conj(W[kn+1, km+1])
            end
        end
    end
    for k in 1:K
        BB[bidx(k, :-, N), bidx(k, :+, N)] += -1
        nk = 1 / (exp(β * εs[k]) + 1)
        BB[bidx(k, :+, 0), bidx(k, :-, 0)] += nk / (1 - nk)
    end
    Mt = II - IB * (BB \ BI)
    C0 = zeros(ComplexF64, Limp, Limp)
    for j0 in 0:N, br in (:+, :-)
        C0[_exrt_legpos(j0, br, true), _exrt_legpos(j0, br, false)] -= 1
    end
    for cell in 1:N
        C0[_exrt_legpos(cell, :+, true), _exrt_legpos(cell - 1, :+, false)] += 1
        C0[_exrt_legpos(cell - 1, :-, true), _exrt_legpos(cell, :-, false)] += 1
    end
    Δ = Mt - C0
    basis = [GrassmannContourIndex(j0 + 1, br, 1, c)
             for j0 in 0:N for br in (:+, :-) for c in (false, true)]
    nbar = [1 / (exp(β * ε) + 1) for ε in εs]
    return _strict_real_from_legpos(basis, Δ), prod(nbar) * det(BB)
end

# 严格容器的分支块切片：腿基序（chain/legpos 同构）→ 各分支 (ā 行, a 列) 块
_strict_blocks_from_legpos(basis::Vector{GrassmannContourIndex}, Δ::Matrix{ComplexF64}) =
    (b1, b2) -> Δ[[i for (i, l) in enumerate(basis) if l.branch == b1 && l.conj],
                  [j for (j, l) in enumerate(basis) if l.branch == b2 && !l.conj]]
_strict_real_from_legpos(basis::Vector{GrassmannContourIndex}, Δ::Matrix{ComplexF64}) =
    (B = _strict_blocks_from_legpos(basis, Δ);
     ExactRealCorrelationFunction(B(:+, :+), B(:+, :-), B(:-, :+), B(:-, :-)))
_strict_mixed_from_legpos(basis::Vector{GrassmannContourIndex}, Δ::Matrix{ComplexF64}) =
    (B = _strict_blocks_from_legpos(basis, Δ);
     ExactMixedCorrelationFunction(B(:τ, :τ), B(:τ, :+), B(:τ, :-), B(:+,:τ), B(:-,:τ),
                                    B(:+, :+), B(:+, :-), B(:-, :+), B(:-, :-)))

# ------------------------------ 混合时间 ------------------------------ #

"""
    exact_fermionic_Δm(f; β, Nτ, t, Nt, μ=0, δτ=β/Nτ, gfsolver=EDToulouseSolver(64))
        -> (ExactMixedCorrelationFunction, normdelta)

ExactEqualDistantKernel 严格 IF 核（混合 Kadanoff–Baym L 轮廓，装配级配对核）——元组返回
`(核, 归一常数)`。轮廓**单环** τ₁→…→τ_{Nτ}→(+,1)→…→(+,Nt+1)→(−,Nt+1)→…→
(−,1)→τ₁：τ cells（含结合 cell τ_{Nτ}→+₁）用 W^τ = e^{−δτ·h}，实时 ± cells
用 W^t = e^{−iδt·h}（− 支共轭），(+,Nt+1)→(−,Nt+1) 闭合边带费米迹负号、
(−,1)→τ₁ 环回边权 1（热演化在 τ 段显式进行，无热边 r）。

腿基同冻结核（`contour_ordering` 混合序：τ 节点 1..Nτ 后接 Keldysh 节点
(+,jj)/(−,jj)，每节点 (a, ā) 相邻），可直接替换 `fermionic_Δm` 的 Δ。

kicks：`kicksτ[j]`（长度 Nτ：轮廓序 τ cells——j = 1..Nτ−1 为内部 cell
τⱼ→τⱼ₊₁、j = Nτ 为结合 cell τ_{Nτ}→+₁）与 `kickst[jj]`（长度 Nt：+ cells；
− cells 自动取共轭伴随）为各 cell 的杂质门相位——
带踢复合同实时版（+ 方向 D(x)·W 杂质输出行 ×x，− 方向 W†D† 杂质输入列
×x̄）。**Keldysh 段在算符层面精确求和相消（含任意幺正踢）：P₊P₋ = I**，
⟹ 全迹收缩 Z = τ 环踢乘积的精确迹（与离散化无关的机器恒等式）。

**normdelta = (−1)^{n(K+1)}·det(BB)/Z_bath**（n = Nτ + 2(Nt+1) 轮廓节点数；
折叠 Berezin 符号 + impurity 折叠，K=1/K=2 × 奇偶 n 四点机器钉定）。装配契约（金标准，`test/exact/fermionic_mixedtime.jl`）：
装配 = 测度 −1 + 单位链 + 两条闭合边（Keldysh 迹 −1、环回 +1）+ data.Δ ⟹
**Z = normdelta·det(装配) == Tr_Fock[τ 环踢乘积]/Z_bath（任意 kicks 机器精确）**。
BB 消元走环状块结构求解（单环仿射前推，O(n·K²·Limp + K³)；稠密参照
`_exrt_mixed_kernel_dense`）。
"""
function exact_fermionic_Δm(f::AbstractBoundedFunction; β::Real, Nτ::Int, t::Real,
                            Nt::Int, μ::Real=0, δτ::Real=β / Nτ,
                            gfsolver::ToulouseSolver=EDToulouseSolver(64),
                            kicksτ::AbstractVector{<:Number}=ones(Nτ),
                            kickst::AbstractVector{<:Number}=ones(Nt))
    Nτ >= 1 || throw(ArgumentError("Nτ 至少为 1"))
    Nt >= 1 || throw(ArgumentError("Nt 至少为 1"))
    length(kicksτ) == Nτ ||
        throw(DimensionMismatch("kicksτ 须长 Nτ（轮廓序 τ cells：Nτ−1 内部 + 结合）"))
    length(kickst) == Nt ||
        throw(DimensionMismatch("kickst 须长 Nt（+ cells）"))
    _mixed_kernel(f, β, Nτ, t, Nt, Float64(δτ), μ, _nbath(gfsolver),
                  ComplexF64.(kicksτ), ComplexF64.(kickst); dense=false)
end

# 装配本体（dense=false 结构化求解 / dense=true 稠密参照）
function _mixed_kernel(f::AbstractBoundedFunction, β::Real, Nτ::Int, t::Real, Nt::Int,
                       δτ::Float64, μ::Real, M::Int,
                       kicksτ::Vector{ComplexF64}, kickst::Vector{ComplexF64};
                       dense::Bool)
    Es, Vsq, wΓ = _quad_nodes(f, μ, M)
    keep = wΓ .> 0
    εs = Es[keep]
    Vs = Vsq[keep]
    K = length(εs)
    K >= 1 || throw(ArgumentError("谱在支撑上的积分为零（或非正）"))
    _W(step, sgn) = begin
        h = zeros(ComplexF64, K + 1, K + 1)
        for k in 1:K
            h[1, k+1] = h[k+1, 1] = Vs[k]
            h[k+1, k+1] = εs[k]
        end
        exp(sgn * step .* h)
    end
    Wτ = _W(δτ, -1.0 + 0.0im)                    # 虚时 cell（含结合 cell 共用）
    Wt = _W(t / Nt, -1.0im)                      # 实时 cell
    n = Nτ + 2 * (Nt + 1)                        # 轮廓节点数（链序）
    ipos(c, ā) = 2 * c - (ā ? 0 : 1)             # 杂质腿（a = 2c−1, ā = 2c）
    bidx(k, c) = (k - 1) * n + c                 # bath 配对槽（行 γ̄，列 γ）
    Limp = 2n
    Nb = K * n
    II = zeros(ComplexF64, Limp, Limp)
    IB = zeros(ComplexF64, Limp, Nb)
    BI = zeros(ComplexF64, Nb, Limp)
    BB = dense ? zeros(ComplexF64, Nb, Nb) : Matrix{ComplexF64}(undef, 0, 0)
    for c in 1:n                                 # Berezin 测度
        II[ipos(c, true), ipos(c, false)] -= 1
        dense && for k in 1:K
            BB[bidx(k, c), bidx(k, c)] -= 1
        end
    end
    # cell 边（dest ← src）：+ 方向 D(x)·W 杂质输出行 ×x；− 方向 W†D† 列 ×x̄
    function edge!(cdest, csrc, W, x)
        II[ipos(cdest, true), ipos(csrc, false)] += x * W[1, 1]
        dense && for km in 1:K, kn in 1:K
            BB[bidx(km, cdest), bidx(kn, csrc)] += W[km+1, kn+1]
        end
        for km in 1:K
            IB[ipos(cdest, true), bidx(km, csrc)] += x * W[1, km+1]
            BI[bidx(km, cdest), ipos(csrc, false)] += W[km+1, 1]
        end
    end
    function edge_adj!(cdest, csrc, W, x)
        # 伴随 W†D†：conj-转置 + 杂质输入列 ×x̄
        xc = conj(x)
        II[ipos(cdest, true), ipos(csrc, false)] += xc * conj(W[1, 1])
        dense && for km in 1:K, kn in 1:K
            BB[bidx(km, cdest), bidx(kn, csrc)] += conj(W[kn+1, km+1])
        end
        for km in 1:K
            IB[ipos(cdest, true), bidx(km, csrc)] += conj(W[km+1, 1])
            BI[bidx(km, cdest), ipos(csrc, false)] += xc * conj(W[1, km+1])
        end
    end
    for c in 2:Nτ                               # τ 内部 cells
        edge!(c, c - 1, Wτ, kicksτ[c - 1])
    end
    edge!(Nτ + 1, Nτ, Wτ, kicksτ[Nτ])           # 结合 cell（τ_Nτ → β ≡ +₁）
    for jj in 1:Nt                              # + cells（cp(jj) → cp(jj+1)）
        edge!(Nτ + 2jj + 1, Nτ + 2jj - 1, Wt, kickst[jj])
    end
    for jj in 1:Nt                              # − cells（cm(jj+1) → cm(jj)，后向）
        edge_adj!(Nτ + 2jj, Nτ + 2jj + 2, Wt, kickst[jj])
    end
    for k in 1:K                                # bath 闭合边与环回边
        dense && begin
            BB[bidx(k, Nτ + 2Nt + 2), bidx(k, Nτ + 2Nt + 1)] += -1   # 费米迹
            BB[bidx(k, 1), bidx(k, Nτ + 2)] += 1                     # (−,1)→τ₁
        end
    end
    # 杂质轮廓闭合边（Schur 前，每腿一次）：Keldysh 迹负号 + 环回单位
    II[ipos(Nτ + 2Nt + 2, true), ipos(Nτ + 2Nt + 1, false)] += -1
    II[ipos(1, true), ipos(Nτ + 2, false)] += 1
    # 消浴：结构化（环）或稠密参照（混合轮廓无热边——τ 段显式演化）
    Wbτ = Wτ[2:end, 2:end]
    Wbt = Wt[2:end, 2:end]
    if dense
        X = BB \ BI
        detBB = det(BB)
    else
        X, detBB = _exrt_schur_mixed_cyclic(Wbτ, Wbt, Nτ, Nt, BI, K)
    end
    Mt = II - IB * X
    # 单位骨架：测度 + 沿轮廓单位链 + 两条闭合边
    C0 = zeros(ComplexF64, Limp, Limp)
    for c in 1:n
        C0[ipos(c, true), ipos(c, false)] -= 1
    end
    C0[ipos(Nτ + 1, true), ipos(Nτ, false)] += 1
    C0[ipos(Nτ + 2Nt + 2, true), ipos(Nτ + 2Nt + 1, false)] += -1
    C0[ipos(1, true), ipos(Nτ + 2, false)] += 1
    for c in 2:Nτ
        C0[ipos(c, true), ipos(c - 1, false)] += 1
    end
    for jj in 1:Nt
        C0[ipos(Nτ + 2jj + 1, true), ipos(Nτ + 2jj - 1, false)] += 1
        C0[ipos(Nτ + 2jj, true), ipos(Nτ + 2jj + 2, false)] += 1
    end
    basis = GrassmannContourIndex[
        vcat([GrassmannContourIndex(j, :τ, 1, c) for j in 1:Nτ for c in (false, true)],
             [GrassmannContourIndex(jj, br, 1, c)
              for jj in 1:Nt+1 for br in (:+, :-) for c in (false, true)])...]
    Zb = prod(1 .+ exp.(-β .* εs))
    # (−1)^{n(K+1)}：折叠 Berezin 符号 (−1)^{nK}（bath 侧，K = 模式数）
    # × impurity 折叠 (−1)^n（本 det 收缩机器相对 QuAPI Zvalue 机器的约定差，
    # K=1/Nτ=3 vs Nτ=2 四点机器钉定）
    pref = (-1.0 + 0.0im)^(n * (K + 1)) * detBB / Zb
    return _strict_mixed_from_legpos(basis, Mt - C0), pref
end

# 稠密参照路径（机器等价性检验用）
function _exrt_mixed_kernel_dense(f::AbstractBoundedFunction, β::Real, Nτ::Int, t::Real,
                                  Nt::Int, δτ::Real, μ::Real, M::Int,
                                  kicksτ::Vector{ComplexF64}, kickst::Vector{ComplexF64})
    _mixed_kernel(f, β, Nτ, t, Nt, Float64(δτ), μ, M, kicksτ, kickst; dense=true)
end

"""
混合轮廓 BB 的环状块结构求解（分配函数内部）。

 bath 块沿轮廓构成**单环**（每行/列恰一个非对角块）：
    1→2→…→Nτ→Nτ+1→Nτ+3→…→n−1→n→n−2→…→Nτ+2→1
 边块依次 [Wbτ×(Nτ−1), Wbτ(结合), Wbt×Nt, −I(迹), adj(Wbt)×Nt(逆序), +I(环回)]，
 diag 全为 −I。环方向仿射前推解 BB·X = BI（O(n·K²·Limp + K³)）；
 **det(BB) = (−1)^{Kn}·det(I_K − Π)**，Π = 环方向边块有序积 B[n]⋯B[1]。
"""
function _exrt_schur_mixed_cyclic(Wbτ::Matrix{ComplexF64}, Wbt::Matrix{ComplexF64},
                                  Nτ::Int, Nt::Int, BI::Matrix{ComplexF64}, K::Int)
    n = Nτ + 2 * (Nt + 1)
    L = size(BI, 2)
    # 环节点序 + 边块（edge i：v[i] → v[i+1]，v 环回）
    cyc = collect(1:Nτ)
    for jj in 1:(Nt + 1)
        push!(cyc, Nτ + 2jj - 1)                # + 节点（奇段）
    end
    push!(cyc, n)                               # − 末节点（迹角入口）
    for jj in (Nt+1):-1:2
        push!(cyc, Nτ + 2jj - 2)                # − 节点（偶段，逆序）
    end
    @assert cyc[1] == 1 && length(cyc) == n
    edges = Vector{Matrix{ComplexF64}}(undef, n)
    for j in 2:Nτ
        edges[j-1] = Wbτ
    end
    edges[Nτ] = Wbτ                             # 结合 cell（Nτ → Nτ+1）
    for jj in 1:Nt                              # + 段：cyc[Nτ+jj] → cyc[Nτ+jj+1]
        edges[Nτ + jj] = Wbt
    end
    edges[Nτ + Nt + 1] = -Matrix{ComplexF64}(I, K, K)   # 迹角（+ 末 → − 末）
    for j in 1:Nt                                       # − 段（逆序 adj）
        edges[Nτ + Nt + 1 + j] = adjoint(Wbt)
    end
    edges[n] = Matrix{ComplexF64}(I, K, K)           # 环回（Nτ+2 → 1）
    # Y 视图（节点 → BI 块行）
    Yv(v) = view(BI, (0:K-1) .* n .+ v, :)
    # 仿射前推：X[v[i]] = A[i]·P + b[i]，P := X[1]
    A = Vector{Matrix{ComplexF64}}(undef, n)
    b = Vector{Matrix{ComplexF64}}(undef, n)
    A[1] = Matrix{ComplexF64}(I, K, K)
    b[1] = zeros(ComplexF64, K, L)
    for i in 2:n
        A[i] = edges[i-1] * A[i-1]
        b[i] = edges[i-1] * b[i-1] - Matrix{ComplexF64}(Yv(cyc[i]))
    end
    # 一致性（节点 1 方程）：X[1] = edges[n]·X[v[n]] − Y[1]
    P = (Matrix{ComplexF64}(I, K, K) - edges[n] * A[n]) \ (edges[n] * b[n] - Matrix{ComplexF64}(Yv(1)))
    # 回代装配 X（BI 行布局：节点 c 的块行 = (0:K-1)*n .+ c）
    X = Matrix{ComplexF64}(undef, size(BI)...)
    x = P
    X[(0:K-1) .* n .+ cyc[1], :] .= x
    for i in 2:n
        x = edges[i-1] * x - Matrix{ComplexF64}(Yv(cyc[i]))
        X[(0:K-1) .* n .+ cyc[i], :] .= x
    end
    # det(BB) = (−1)^{Kn}·det(I_K − Π)，Π = edges 沿环有序积 B[n]⋯B[1]
    Π = edges[n]
    for i in (n-1):-1:1
        Π = Π * edges[i]
    end
    detBB = (-1.0 + 0.0im)^(K * n) *
            det(Matrix{ComplexF64}(I, K, K) - Π)
    return X, detBB
end
