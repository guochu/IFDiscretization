# =============================================================================
# _kicks_helper.jl —— 带踢算符忠实核构造（金标准测试专用 helper）
#
# 2026-11-22 从 src/fermions/exact/fermionic.jl 移出：带踢核（核内携带杂质
# 踢相位，Z ≡ 1 机器、观测量 = 门序列的精确收缩）不属于 exact 核族的
# 「纯耦合项严格核」导出契约（见 real.jl exact_fermionic_Δt docstring），
# 但其 Z≡1 证书是底层 W/Schur 数学的最锐利锚——保留为金标准测试 helper。
# 本文件在测试作用域定义，引用 IFD._quad_nodes/IFD._nbath。
# =============================================================================

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
                               IFD._nbath(gfsolver), ComplexF64.(kicks))
end

function _exrt_strict_kernel(f::AbstractBoundedFunction, cells::Vector{Float64},
                             β::Float64, μ::Real, M::Int,
                             kicks::Vector{ComplexF64}=ones(length(cells)))
    N = length(cells)
    N >= 1 || throw(ArgumentError("cell schedule must carry at least one cell"))
    Es, Vsq, wΓ = IFD._quad_nodes(f, μ, M)
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
        II[IFD._exrt_legpos(j0, br, true), IFD._exrt_legpos(j0, br, false)] -= 1
    end
    for cell in 1:N
        W = Ws[cell]
        x = kicks[cell]
        # + 支：D(x)·W——杂质输出行 ×x
        II[IFD._exrt_legpos(cell, :+, true), IFD._exrt_legpos(cell - 1, :+, false)] += x * W[1, 1]
        for km in 1:K
            IB[IFD._exrt_legpos(cell, :+, true), bidx(km, :+, cell - 1)] += x * W[1, km+1]
            BI[bidx(km, :+, cell), IFD._exrt_legpos(cell - 1, :+, false)] += W[km+1, 1]
        end
        # − 支：伴随 W†D†——conj-转置 + 杂质输入列 ×x̄
        xc = conj(x)
        II[IFD._exrt_legpos(cell - 1, :-, true), IFD._exrt_legpos(cell, :-, false)] +=
            xc * conj(W[1, 1])
        for km in 1:K
            IB[IFD._exrt_legpos(cell - 1, :-, true), bidx(km, :-, cell)] += conj(W[km+1, 1])
            BI[bidx(km, :-, cell - 1), IFD._exrt_legpos(cell, :-, false)] +=
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
        C0[IFD._exrt_legpos(j0, br, true), IFD._exrt_legpos(j0, br, false)] -= 1
    end
    for cell in 1:N
        C0[IFD._exrt_legpos(cell, :+, true), IFD._exrt_legpos(cell - 1, :+, false)] += 1
        C0[IFD._exrt_legpos(cell - 1, :-, true), IFD._exrt_legpos(cell, :-, false)] += 1
    end
    Δ = Mt - C0
    basis = [GrassmannContourIndex(j0 + 1, br, 1, c)
             for j0 in 0:N for br in (:+, :-) for c in (false, true)]
    nbar = [1 / (exp(β * ε) + 1) for ε in εs]
    # 真空权重归一 Π(1−n̄ₖ)（← QuAPI exactfermionic.jl `prefactor` 约定逐字——
    # 勿换物理占据 Πn̄ₖ：差每模 e^{βεₖ}，对称谱下不可见、非对称离散谱 Z 证书
    # 直接破；与稠密参照路径同款注释）
    return IFD._strict_real_from_legpos(basis, Δ), prod(1 .- nbar) * detBB
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
    Es, Vsq, wΓ = IFD._quad_nodes(f, μ, M)
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
        II[IFD._exrt_legpos(j0, br, true), IFD._exrt_legpos(j0, br, false)] -= 1
        for k in 1:K
            BB[bidx(k, br, j0), bidx(k, br, j0)] -= 1
        end
    end
    for cell in 1:N
        W = Ws[cell]
        x = kicks[cell]
        II[IFD._exrt_legpos(cell, :+, true), IFD._exrt_legpos(cell - 1, :+, false)] += x * W[1, 1]
        for km in 1:K
            IB[IFD._exrt_legpos(cell, :+, true), bidx(km, :+, cell - 1)] += x * W[1, km+1]
            BI[bidx(km, :+, cell), IFD._exrt_legpos(cell - 1, :+, false)] += W[km+1, 1]
            for kn in 1:K
                BB[bidx(km, :+, cell), bidx(kn, :+, cell - 1)] += W[km+1, kn+1]
            end
        end
        xc = conj(x)
        II[IFD._exrt_legpos(cell - 1, :-, true), IFD._exrt_legpos(cell, :-, false)] +=
            xc * conj(W[1, 1])
        for km in 1:K
            IB[IFD._exrt_legpos(cell - 1, :-, true), bidx(km, :-, cell)] += conj(W[km+1, 1])
            BI[bidx(km, :-, cell - 1), IFD._exrt_legpos(cell, :-, false)] +=
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
        C0[IFD._exrt_legpos(j0, br, true), IFD._exrt_legpos(j0, br, false)] -= 1
    end
    for cell in 1:N
        C0[IFD._exrt_legpos(cell, :+, true), IFD._exrt_legpos(cell - 1, :+, false)] += 1
        C0[IFD._exrt_legpos(cell - 1, :-, true), IFD._exrt_legpos(cell, :-, false)] += 1
    end
    Δ = Mt - C0
    basis = [GrassmannContourIndex(j0 + 1, br, 1, c)
             for j0 in 0:N for br in (:+, :-) for c in (false, true)]
    nbar = [1 / (exp(β * ε) + 1) for ε in εs]
    # 真空权重归一 Π(1−n̄ₖ)（← QuAPI exactfermionic.jl `prefactor` 约定逐字——
    # Berezin 测度每 bath pair 的真空投影归一；勿换物理占据 Πn̄ₖ：那差每模
    # e^{βεₖ}，对称谱（Σεₖ=0，金标准半圆/T2 双线）下不可见、非对称离散谱
    # （DiracDelta/DiscreteSpectrum）Z 证书直接破（2026-11-21 定谳修复，
    # 三谱 + μ 平移 Z ≡ 1 机器验证）
    return IFD._strict_real_from_legpos(basis, Δ), prod(1 .- nbar) * det(BB)
end

"""
    exact_fermionic_Δm_kicks(f; β, Nτ, t, Nt, μ=0, δτ=β/Nτ,
                             gfsolver=EDToulouseSolver(64), kicksτ, kickst)
        -> (ExactMixedCorrelationFunction, normdelta)

混合轮廓带踢核（金标准测试专用，2026-11-22 契约换代自公共 API 移出）：
`kicksτ[j]`（长 Nτ：轮廓序 τ cells——j = 1..Nτ−1 内部 cell、j = Nτ 结合
cell）与 `kickst[jj]`（长 Nt：+ cells；− cells 自动取共轭伴随）为各 cell
的杂质门相位（+ 方向 D(x)·W 杂质输出行 ×x，− 方向 W†D† 杂质输入列 ×x̄）。
数学本体 = `IFD._mixed_kernel`（公共 `exact_fermionic_Δm` = 本构造
kicks ≡ 1 的纯耦合项退化）。**Keldysh 段在算符层面精确求和相消
（含任意幺正踢）：P₊P₋ = I** ⟹ 全迹收缩 Z = τ 环踢乘积的精确迹。
normdelta = (−1)^{n(K+1)}·det(BB)/Z_bath（带踢与纯核同式）。
"""
function exact_fermionic_Δm_kicks(f::AbstractBoundedFunction; β::Real, Nτ::Int,
                                  t::Real, Nt::Int, μ::Real=0, δτ::Real=β / Nτ,
                                  gfsolver::ToulouseSolver=IFD.EDToulouseSolver(64),
                                  kicksτ::AbstractVector{<:Number}=ones(Nτ),
                                  kickst::AbstractVector{<:Number}=ones(Nt))
    length(kicksτ) == Nτ ||
        throw(DimensionMismatch("kicksτ 须长 Nτ（轮廓序 τ cells：Nτ−1 内部 + 结合）"))
    length(kickst) == Nt ||
        throw(DimensionMismatch("kickst 须长 Nt（+ cells）"))
    IFD._mixed_kernel(f, β, Nτ, t, Nt, Float64(δτ), μ, IFD._nbath(gfsolver),
                      ComplexF64.(kicksτ), ComplexF64.(kickst); dense=false)
end
