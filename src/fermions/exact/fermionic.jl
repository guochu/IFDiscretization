# =============================================================================
# exact —— 费米浴严格 IF 核（ExactIF）
#
# 纯浴对象契约（2026-11-18 换代）：三个入口签名与冻结核 fermionic_Δτ/Δt/Δm
# 一致，不含任何杂质信息（无 ε_d、无劈分阶数）——杂质 kick 相位经恒等式
# Δ_IF(x) = x·Δ_IF(1)（机器精度）线性析出到装配侧系统骨架，下游换核是
# 机械替换。虚时核的严格性：反周期环单粒子矩阵的 Schur 消浴
#（test/exact/strict_kernel.jl 金标准：orders 1/2 × N=2..8 的 Z/n̄/G 与
# 同阶直接 Trotter 演化逐位一致）；实时/混合核当前为点值严格轮廓收缩
#（exactif 族约定，非装配配对核——虚时负对照定量：点值冒充配对核差
# O(0.1)；两轮廓的装配级严格核待后续步骤落地）。
# =============================================================================

# ------------------------------ 虚时 ------------------------------ #

"""
    exact_fermionic_Δτ(f; β, N, μ=0, δτ=β/N, gfsolver=EigenSolver(64))
        -> (ImagCorrelationFunction, normdelta)

ExactIF 严格 IF 核（虚时反周期环）——**纯浴对象**（不含任何杂质信息），
元组返回：`(Δ_if, normdelta)`。

严格性依据：杂质 kick 相位 x = e^{−δτ·ε_d} 从严格核线性析出，Δ_IF(x) =
x·Δ_IF(1)；x=1 时各阶劈分 cell 退化为同一纯浴演化 W = e^{−δτ·h₀} ⟹ 核与
order、ε_d 无关。构造 = 反周期环单粒子矩阵 M = −I + kron(W, S)（S 反周期
移位）的杂质通道 Schur 补 Δ_full = II − IB·BB⁻¹·BI，减去 x=1 骨架
（测度 −I + 动能 1·S）。装配契约（金标准，`test/exact/fermionic_imagtime.jl`
数值钉死）：B = skeleton(x) + x·Δ_if（骨架 = 测度 −I + 动能 x·ā'ᵢ₊₁a'ᵢ +
反周期闭 −x·ā'₁a'_N），Pfaffian 收缩的 Z/n̄/G 与同阶直接 Trotter 演化逐位
一致（一阶全步踢与二阶 Strang 半步踢同核——对角 H_imp 下踢相位沿迹可对易、
总相位同为 xᴺ）。

**normdelta = det(BB)**（纯浴环浴块行列式，Z 归一的必要常数项）：
`Z = normdelta · Pf(B)`。

存储约定同冻结核：CorrelationMatrix 双向量环（接触约定 ηⱼₖ[1] = 全量
对角，ηₖⱼ[1] = 0）。
"""
function exact_fermionic_Δτ(f::AbstractBoundedFunction; β::Real, N::Int,
                            μ::Real=0, δτ::Real=β / N,
                            gfsolver::ToulouseGreenFunctionSolver=EigenSolver(64))
    M = _nbath(gfsolver)
    Es, Vsq, _ = _quad_nodes(f, μ, M)
    n = M + 1
    h0 = zeros(ComplexF64, n, n)
    for k in 1:M
        h0[1, 1+k] = Vsq[k]; h0[1+k, 1] = conj(Vsq[k]); h0[1+k, 1+k] = Es[k]
    end
    W = exp(-δτ .* h0)                       # 纯浴 cell（x=1：各阶劈分同此 W）
    S = zeros(ComplexF64, N, N)
    for k in 2:N; S[k, k-1] = 1.0; end
    S[1, N] = -1.0                                    # 反周期扭转
    MM = -Matrix{ComplexF64}(I, n * N, n * N) + kron(W, S)
    II = MM[1:N, 1:N]
    IB = MM[1:N, N+1:end]
    BI = MM[N+1:end, 1:N]
    BB = MM[N+1:end, N+1:end]
    Δ_full = II - IB * (BB \ BI)
    Δ = Δ_full + Matrix{ComplexF64}(I, N, N) - S   # 减 x=1 骨架（−I + S）⟹ IF-only
    ηⱼₖ = [Δ[1 + d, 1] for d in 0:N-1]
    ηₖⱼ = vcat([zero(Δ[1, 1])], [Δ[1, 1 + d] for d in 1:N-1])
    # 必要常数项：纯浴环浴块行列式，装配 Z = normdelta · Pf(B)（金标准契约）
    return ImagCorrelationFunction(CorrelationMatrix(ηⱼₖ, ηₖⱼ)), det(BB)
end

# ------------------------------ 实时 ------------------------------ #

"腿位（节点 j0 = 0..N；+ 支在前，支内 (a, ā) 相邻），1-based"
_exrt_legpos(j0::Int, br::Symbol, ā::Bool) = 4 * j0 + (br === :+ ? 0 : 2) + (ā ? 2 : 1)

"""
    exact_fermionic_Δt(f; β, N, t, μ=0, gfsolver=EigenSolver(64))
        -> (GrassmannHybridizationMatrix, normdelta)

ExactIF 严格 IF 核（实时 Keldysh 轮廓），元组返回 `(核, 归一常数)`——
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
                            μ::Real=0, gfsolver::ToulouseGreenFunctionSolver=EigenSolver(64))
    return exact_fermionic_Δt_cells(f; β=β, cells=fill(Float64(t) / N, N), μ=μ,
                                    gfsolver=gfsolver)
end

"""
    exact_fermionic_Δt_cells(f; β, cells, μ=0, gfsolver=EigenSolver(64), kicks)
        -> (GrassmannHybridizationMatrix, normdelta)

实时 Keldysh 严格核的逐 cell 调度版：`cells[k]` = 第 k 个 Keldysh cell 的演化
长度（可取 0/负），`kicks[k]` = 第 k 个 gap 的杂质踢相位 x_k = e^{−iα_k·ε_d}
（默认全 1 = 纯浴对象）。腿点数 = length(cells)+1（节点 0..N，腿基同冻结核
GrassmannHybridizationMatrix(fermionic_Δt(...))）。

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
                                  gfsolver::ToulouseGreenFunctionSolver=EigenSolver(64),
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
    return GrassmannHybridizationMatrix(basis, Δ), prod(nbar) * detBB
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
    return GrassmannHybridizationMatrix(basis, Δ), prod(nbar) * det(BB)
end

# ------------------------------ 混合时间 ------------------------------ #

"""
    exact_fermionic_Δm(f; β, Nτ, t, Nt, μ=0, δτ=β/Nτ, gfsolver=EigenSolver(64))
        -> MixedCorrelationFunction

L 形（Kadanoff-Baym）轮廓的连续谱严格核。分支序：τ 支（0→β，Nτ 节点）
最前，其后 Keldysh 部分与实时版逐位相同。九块：
  G₊₋/G₋₊：与 exact_fermionic_Δt 的交叉块一致；
  Gττ：与 exact_fermionic_Δτ 一致（环 Toeplitz）；
  G₊τ (Nt+1)×Nτ、Gτ₊ Nτ×(Nt+1)：L 轮廓 τ-交叉收缩——τ 顶点的收缩向量
  = e^{−(β−τᵢ)h₀}e₁（W(t)W†(t)=I 使 Keldysh 闭环相消）；轮廓序 τ 最早
  ⟹ ⟨C_A A_τ⟩ = Σ n̄ c_a c̄_τ、⟨A_a C_τ⟩ = Σ (1−n̄) c̄_a c_τ。
  (−,τ)/(τ,−) 的 Keldysh 相对负号由 branch 规则承担（存 G₊τ/Gτ₊）。
"""
function exact_fermionic_Δm(f::AbstractBoundedFunction; β::Real, Nτ::Int, t::Real,
                            Nt::Int, μ::Real=0, δτ::Real=β / Nτ,
                            gfsolver::ToulouseGreenFunctionSolver=EigenSolver(64))
    δt = t / Nt
    M = _nbath(gfsolver)
    Es, Vsq, _ = _quad_nodes(f, μ, M)
    ev = _bath_eig(Es, Vsq)
    W(θ) = θ == 0 ? Matrix{ComplexF64}(I, M+1, M+1) :
           ev.vectors * Diagonal(exp.(-im * ev.values .* θ)) * ev.vectors'
    Wd(θ) = θ == 0 ? Matrix{ComplexF64}(I, M+1, M+1) :
            ev.vectors * Diagonal(exp.(im * ev.values .* θ)) * ev.vectors'
    n̄ = vcat([0.5], [1 / (1 + exp(β * e)) for e in Es])
    n1, e1 = Nt + 1, vcat([1.0], zeros(M))
    # 收缩向量：forward a、backward a 同 Δt
    cfwd = [Wd(t) * W(t - a * δt) * e1 for a in 0:Nt]
    cbwd = [Wd(a * δt) * e1 for a in 0:Nt]
    posf(a) = Nτ + a
    posb(a) = Nτ + Nt + (Nt - a)
    # τ-交叉块（本征基谱闭式，2 模多体 4 点逐位锚定 mx_fix12）：
    #   G₊τ[a+1,i] = Σₚ a2ₚ e^{−λₚτᵢ} e^{−iλₚt_a} fₚ   （粒子线到 C 端相位）
    #   Gτ₊[i,a+1] = −(G₊τ)ᵀ（μ=0 反对称）。
    #   注：V→0 时 G₊τ → ½（杂质自线）——系统腿结构，由调用方测度抵消
    #   （同 firstorder README §2.2 的 c-number 抵消簿记）。
    a2 = abs2.(ev.vectors[1, :])
    fk = [1 / (1 + exp(-β * ev.values[p])) for p in 1:M+1]
    G₊τ = [sum(a2[p] * exp(-ev.values[p] * ((i - 1) * δτ)) *
               exp(-im * ev.values[p] * (a * δt)) * fk[p] for p in 1:M+1)
           for a in 0:Nt, i in 1:Nτ]
    # μ=0 费米反对称 ζⱼₖ = −ζₖⱼᵀ（物理要求，docstring 语义）：
    Gτ₊ = μ == 0 ? Matrix(-transpose(G₊τ)) :
           [sum(a2[p] * exp(-ev.values[p] * ((i - 1) * δτ)) *
                exp(im * ev.values[p] * (a * δt)) * fk[p] for p in 1:M+1)
            for i in 1:Nτ, a in 0:Nt]
    # Keldysh 交叉块（与 Δt 同构造）
    function pair(v, w)
        cv, cw = v ≤ n1 ? cfwd[v] : cbwd[v - n1], w ≤ n1 ? cfwd[w] : cbwd[w - n1]
        pv = v ≤ n1 ? posf(v - 1) : posb(v - n1 - 1)
        pw = w ≤ n1 ? posf(w - 1) : posb(w - n1 - 1)
        if pv == pw
            return sum((1 - n̄[l]) * conj(cv[l]) * cw[l] for l in 2:M+1)
        elseif pv > pw
            return sum(n̄[l] * cv[l] * conj(cw[l]) for l in 2:M+1)
        else
            return sum((1 - n̄[l]) * conj(cv[l]) * cw[l] for l in 2:M+1)
        end
    end
    Gpm = [pair(n1 + j + 1, i + 1) for i in 0:Nt, j in 0:Nt]
    Gmp = [pair(j + 1, n1 + i + 1) for i in 0:Nt, j in 0:Nt]
    # ττ 块 = 环 Toeplitz（同 Δτ）
    a2 = abs2.(ev.vectors[1, :])
    G0(τ) = τ > 0 ?
            sum(a2[p] * exp(-ev.values[p] * τ) / (1 + exp(-β * ev.values[p])) for p in 1:M+1) :
            -sum(a2[p] * exp(-ev.values[p] * (τ + β)) / (1 + exp(-β * ev.values[p])) for p in 1:M+1)
    Gττ = CorrelationMatrix(ComplexF64[real(G0((d - 1) * δτ)) for d in 1:Nτ],
                            ComplexF64[real(G0(-(d - 1) * δτ)) for d in 1:Nτ])
    Gττ = CorrelationMatrix(ComplexF64.(Gττ.ηⱼₖ), ComplexF64.(Gττ.ηₖⱼ))
    return MixedCorrelationFunction(Gpm, Gmp, G₊τ, Gτ₊, Gττ)
end
