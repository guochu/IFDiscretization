# =============================================================================
# exact/trotter_orders.jl —— exact IF 的阶数严格对比：1/2/4/6 阶 Trotter
#
# 断言：exact IF 核（W(θ) = e^{−θh₀}，即 EigenSolver 传播的同一直积 cell）
# 在 order-k kick 装配下的 L 轮廓 Green 函数，与**同阶直接 Trotter 严格演化**
# （JW Fock 多体轮廓迹，同一批 order-k 单胞）在**全节点**（m = 0..Nτ 含 τ=0
# 与 β 顶点、a = 0..Nt 含两端点）逐位一致（机器精度）。
#
# ← QuAPI/exactif/test/mx_mb_anchor.jl 移植并推广：原版仅 order 1/2；此处把
#   单粒子装配与多体直接演化都推广到 4/6 阶（Suzuki 递归 S₄/S₆ = order-2
#   段的组合，p = θ/(4−4^{1/3})、θ/(4−4^{1/5})），并加非空性检查
#   （ε_d ≠ 0、有限 N 下各阶结果两两不同——防止恒等式空转）。
#
# 语义边界：ε_d 不入 exact IF 核（调用方 kick 簿记，exact API 文档约定）——
# 本测试即该调用方契约的定义级锚定：装配的 kick 结构（order-k）+ exact 核
# == 同阶直接演化。多体迹公式（Wick ⟹ 与单粒子装配逐位相等）：
#   G(m,a) = tr[Ŝ^{Nτ−m} d̂ Ŝ^m K̂^a d̂† K̂^{−a}] / tr[Ŝ^{Nτ}]
# =============================================================================

include(joinpath(@__DIR__, "_anchor_utils.jl"))

# 离散 2 模浴（Fock 8 维可承受）：杂质（位 1，ε=0）+ 星形耦合
const _ES = [-0.7, 1.3]
const _VS = [0.45, -0.6im]
const _BETA = 2.0
const _EPSD = 0.3
const _T = 0.6

"单粒子 h₀（杂质 + 2 浴模，3×3）"
function _sp_h()
    M = length(_ES)
    h = zeros(ComplexF64, M + 1, M + 1)
    for k in 1:M
        h[1, 1+k] = _VS[k]; h[1+k, 1] = conj(_VS[k]); h[1+k, 1+k] = _ES[k]
    end
    return h
end

# order-k 单胞递归（单粒子矩阵；imag=time=false 为实时核，true 为虚时核）
function _cell(order::Int, θ::Real, h::AbstractMatrix; imag::Bool)
    W = imag ? exp(-θ * h) : exp(-im * θ * h)
    Φ(α) = begin
        D = Matrix{ComplexF64}(I, size(h, 1), size(h, 1))
        D[1, 1] = imag ? exp(-α * _EPSD) : exp(-im * α * _EPSD)
        D
    end
    return order == 1 ? Φ(θ) * W :
           order == 2 ? Φ(θ / 2) * W * Φ(θ / 2) :
           order == 4 ? (p = θ / (4 - 4^(1 / 3));
                         _cell(2, p, h; imag=imag) * _cell(2, p, h; imag=imag) *
                         _cell(2, θ - 4p, h; imag=imag) *
                         _cell(2, p, h; imag=imag) * _cell(2, p, h; imag=imag)) :
                        (p = θ / (4 - 4^(1 / 5));
                         _cell(4, p, h; imag=imag) * _cell(4, p, h; imag=imag) *
                         _cell(4, θ - 4p, h; imag=imag) *
                         _cell(4, p, h; imag=imag) * _cell(4, p, h; imag=imag))
end

"单粒子装配：L 轮廓两点（exact IF 核 + order-k kick；全节点矩阵）"
function assembly_grid(order::Int, Nτ::Int, Nt::Int)
    h = _sp_h()
    n = size(h, 1)
    S = _cell(order, _BETA / Nτ, h; imag=true)
    K = _cell(order, _T / Nt, h; imag=false)
    PN = S^Nτ
    F = eigen(PN)
    ω = -real.(log.(F.values)) ./ _BETA             # 等效频率（保 kick 相位差）
    n̄P = F.vectors * Diagonal([1 / (1 + exp(_BETA * ω[j])) for j in 1:n]) / F.vectors
    F_P = I - n̄P
    e1 = vcat([1.0], zeros(n - 1))
    Spow = [Matrix{ComplexF64}(I, n, n)]
    Kpow = [Matrix{ComplexF64}(I, n, n)]
    for _ in 1:Nτ; push!(Spow, Spow[end] * S); end
    for _ in 1:Nt;  push!(Kpow, Kpow[end] * K); end
    G = zeros(Nτ + 1, Nt + 1)
    for m in 0:Nτ
        r = e1' * Spow[m+1] * F_P
        for a in 0:Nt
            G[m+1, a+1] = real(r * Kpow[a+1] * e1)
        end
    end
    return G
end

"多体直接演化：同一批 order-k 单胞的 JW Fock 轮廓迹（全节点）"
function mb_grid(order::Int, Nτ::Int, Nt::Int)
    h = _sp_h()
    G0 = mb_generator(h)                            # 8×8 多体生成元（JW）
    dim = 1 << size(h, 1)
    nd(J) = J & 1                                   # 杂质占据 = 位 0
    kτ(α) = Diagonal([exp(-α * _EPSD * nd(J)) for J in 0:dim-1])
    kt(α) = Diagonal([exp(-im * α * _EPSD * nd(J)) for J in 0:dim-1])
    S = _mb_cell(order, _BETA / Nτ, G0, kτ; imag=true)
    K = _mb_cell(order, _T / Nt, G0, kt; imag=false)
    Ki = inv(K)
    d̂ = zeros(ComplexF64, dim, dim)     # 杂质湮没（位 0：|jb,1⟩ → |jb,0⟩，无串符）
    dd = zeros(ComplexF64, dim, dim)
    for jb in 0:dim÷2-1
        d̂[2jb+1, 2jb+2] = 1.0
        dd[2jb+2, 2jb+1] = 1.0
    end
    Z = tr(S^Nτ)
    Spow = [Matrix{ComplexF64}(I, dim, dim)]
    Kpow = [Matrix{ComplexF64}(I, dim, dim)]; Kipow = [Matrix{ComplexF64}(I, dim, dim)]
    for _ in 1:Nτ; push!(Spow, Spow[end] * S); end
    for _ in 1:Nt;  push!(Kpow, Kpow[end] * K); push!(Kipow, Kipow[end] * Ki); end
    G = zeros(Nτ + 1, Nt + 1)
    for m in 0:Nτ
        pre = Spow[Nτ - m + 1] * d̂ * Spow[m+1]
        for a in 0:Nt
            G[m+1, a+1] = real(tr(pre * Kpow[a+1] * dd * Kipow[a+1]) / Z)
        end
    end
    return G
end

# order-k 多体单胞递归（多体核 = mb_generator 的指数；kick 为对角数算符相位）
function _mb_cell(order::Int, θ::Real, G0::AbstractMatrix, kick; imag::Bool)
    W = imag ? exp(-θ * G0) : exp(-im * θ * G0)
    return order == 1 ? kick(θ) * W :
           order == 2 ? kick(θ / 2) * W * kick(θ / 2) :
           order == 4 ? (p = θ / (4 - 4^(1 / 3));
                         _mb_cell(2, p, G0, kick; imag=imag) * _mb_cell(2, p, G0, kick; imag=imag) *
                         _mb_cell(2, θ - 4p, G0, kick; imag=imag) *
                         _mb_cell(2, p, G0, kick; imag=imag) * _mb_cell(2, p, G0, kick; imag=imag)) :
                        (p = θ / (4 - 4^(1 / 5));
                         _mb_cell(4, p, G0, kick; imag=imag) * _mb_cell(4, p, G0, kick; imag=imag) *
                         _mb_cell(4, θ - 4p, G0, kick; imag=imag) *
                         _mb_cell(4, p, G0, kick; imag=imag) * _mb_cell(4, p, G0, kick; imag=imag))
end

@testset "exact IF：1/2/4/6 阶 Trotter 严格对比（装配 vs 同阶直接演化）" begin
    for order in (1, 2, 4, 6), (Nτ, Nt) in ((5, 3), (8, 2))
        Gsp = assembly_grid(order, Nτ, Nt)
        Gmb = mb_grid(order, Nτ, Nt)
        d = maximum(abs.(Gsp - Gmb))
        @test d < 1e-12   # 全节点（m = 0..Nτ、a = 0..Nt）逐位
    end
    # 非空性：ε_d ≠ 0、有限 N 下各阶结果两两不同（恒等式不空转）
    G1 = assembly_grid(1, 5, 3)
    G2 = assembly_grid(2, 5, 3)
    G4 = assembly_grid(4, 5, 3)
    G6 = assembly_grid(6, 5, 3)
    @test maximum(abs.(G1 - G2)) > 1e-6
    @test maximum(abs.(G2 - G4)) > 1e-8
    @test maximum(abs.(G4 - G6)) > 1e-10
end
println("exact IF 1/2/4/6 阶 Trotter 严格对比通过 ✓")
