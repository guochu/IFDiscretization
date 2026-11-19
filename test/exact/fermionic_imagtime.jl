# =============================================================================
# exact/fermionic_imagtime.jl —— 费米虚时 ExactEqualDistantKernel 金标准
#
# 三层断言（严格性金标准，2026-11-18 起放弃 ED 主参照，改高效路线）：
#
#   T2 定义锚（小规模）：2 模 bath，直接 Trotter 的**单粒子环矩阵**演算
#      == 8 维 Fock ED（钉死约定：Z/n̄/G 的环提取式；ED 只在此处使用）。
#   T1 对齐（严格性，主断言）：包函数 `exact_fermionic_Δτ`（纯浴严格 IF 核，
#      无杂质信息）+ x·Δ 装配 == 直接 Trotter 单粒子演化。连续谱半圆 bath
#      经 GL-M 节点离散（与包内核内部离散精确同构），M=32、N ∈ {8,32,64}、
#      orders 1/2，Z/n̄/G 机器精度。直接路线规模 O((MN)³)，可做大做长。
#   T3 严格解比对（物理）：装配/直接两路线共同对照
#      · 离散模型严格解（h₀ 本征闭式，含 εd，无 Trotter 误差）：δτ² 收敛。
#        结构性事实：对角 H_imp 的 τ 环上 Lie ≡ Strang（踢相位沿闭合迹可
#        对易滑移，边界吸收相位）⟹ 无一阶/二阶分离，收敛从 δτ² 起步；
#        阶数分离断言属实时开线几何（fermionic_realtime.jl）。
#      · 连续模型严格解（ImpurityModelBase `fermionic_toulouse_Gτ_semicircular`
#        闭式）：最细 N 饱和于谱离散地板（GL-M 对半圆的求积偏差）。
#
# 装配契约：B = skeleton(x) + x·Δ_IF（x = e^{−δτ·ε_d}，ε_d ≠ 0 ⟹ x-标度
# 契约直接受检）；Z = det(BB)·Pf(B)，det(BB) = 纯浴环浴块行列式（本文件
# 内联构造）；n̄、G 从 B⁻¹ 元素提取。
# =============================================================================

# ---------------- 直接 Trotter：单粒子环矩阵演算（高效路线） ----------------

"""
    direct_trotter_sp(h0, εd, order, N, β) -> (Z, n̄, G)

无相互作用模型的 n 阶直接 Trotter 演化（单粒子环矩阵，O((nN)³)）：
环矩阵 A = −I + kron(Wcell, S)（Wcell = 踢相位×纯浴 cell；S 反周期移位），
Z = (−1)^{nN}·det(A)（反周期谱恒等式 det(−I+kron(C,S)) = (−1)^{nN}det(I+C^N)），
n̄ = 1 + row₁[1]、G(τ_m) = row₁[N−m+1]（row₁ = Aᵀ\e₁；约定经 T2 的 ED 锚定）。
"""
function direct_trotter_sp(h0::AbstractMatrix, εd::Real, order::Int, N::Int, β::Real)
    n = size(h0, 1)
    δτ = β / N
    W = exp(-δτ .* h0)
    D(α) = begin E = Matrix{ComplexF64}(I, n, n); E[1, 1] = exp(-α * εd); E end
    Wcell = order == 1 ? D(δτ) * W : D(δτ / 2) * W * D(δτ / 2)
    S = zeros(ComplexF64, N, N)
    for k in 2:N; S[k, k-1] = 1.0; end
    S[1, N] = -1.0                                    # 反周期扭转
    A = -Matrix{ComplexF64}(I, n * N, n * N) + kron(Wcell, S)
    Z = (-1)^(n * N) * det(A)
    row1 = transpose(A) \ vcat([1.0 + 0.0im], zeros(ComplexF64, n * N - 1))
    n̄ = 1 + real(row1[1])
    G = [1 - n̄; [real(row1[N - m + 1]) for m in 1:N-1]]
    return Z, n̄, G
end

"M 模 GL 离散的耦合单粒子 h₀（杂质位 1，ε=0；与包内核内部离散精确同构）"
function coupled_h0(f::AbstractBoundedFunction, M::Int; μ::Real=0.0)
    Es, Vsq, _ = IFD._quad_nodes(f, μ, M)
    n = M + 1
    h0 = zeros(ComplexF64, n, n)
    for k in 1:M
        h0[1, 1+k] = Vsq[k]; h0[1+k, 1] = conj(Vsq[k]); h0[1+k, 1+k] = Es[k]
    end
    return h0
end

"""
    strictif_assembly(f, εd, N, β, gfsolver) -> (Z, n̄, G)

严格 IF 路线装配：B = skeleton(x) + x·Δ_if。**无 order 参数（结构性事实，
非遗漏）**：τ 环上 Lie（每 cell 全步踢）与 Strang（每 cell 两端半步踢）的
骨架矩阵相同——踢相位住在顶点（入边半步 × 出边半步 = x），环无边界边，
每条边恰好承接相邻两半步的合并 ⟹ 两阶骨架同为 −I + x·S；纯浴核亦与阶数
无关（x=1 时各阶 cell 退化为同一 W）。故同一装配按阶分别对齐直接演化：
T1 逐 order 对比，T3 断言 direct(1) ≡ direct(2)（对角 H_imp 的规范恒等）。
"""
function strictif_assembly(f::AbstractBoundedFunction, εd::Real, N::Int, β::Real,
                           gfsolver)
    x = exp(-εd * β / N)
    Δbath, normdelta = exact_fermionic_Δτ(f; β=β, N=N, gfsolver=gfsolver)
    K1 = Matrix(Δbath.data)
    B = zeros(ComplexF64, 2N, 2N)
    for i in 1:N; B[2i, 2i-1] += -1.0; B[2i-1, 2i] -= -1.0; end
    for i in 1:N-1; B[2(i+1), 2i-1] += x; B[2i-1, 2(i+1)] -= x; end
    B[2, 2N-1] += -x; B[2N-1, 2] -= -x
    for i in 1:N, j in 1:N; B[2i, 2j-1] += x * K1[i, j]; B[2j-1, 2i] -= x * K1[i, j]; end
    Zpf = pfaffian(B)
    Binv = inv(B)
    n̄ = 1 - Binv[2, 1]
    G = [1 - n̄; [-Binv[2(N-m+1), 1] for m in 1:N-1]]
    return real(normdelta) * real(Zpf), n̄, G
end

# ================================= 测试 =================================

include(joinpath(@__DIR__, "..", "api", "_pfaffian_utils.jl"))
const _pf = pfaffian
include(joinpath(@__DIR__, "_anchor_utils.jl"))

const β = 2.0
const εd = 0.3
const t_half = 1.0

@testset "费米虚时 ExactEqualDistantKernel 金标准（exact_fermionic_Δτ vs 直接 Trotter vs 严格解）" begin

    # ---- T2 定义锚：2 模 bath，单粒子直接路线 == Fock ED（钉约定） ----
    @testset "T2 定义锚：单粒子直接路线 == Fock ED（2 模）" begin
        Es2 = [-0.7, 1.3]; Vs2 = [0.45, 0.6]
        mid = (Es2[1] + Es2[2]) / 2
        half = (Es2[2] - Es2[1]) * sqrt(3) / 2
        f2 = bounded(ω -> (abs(ω - (mid - half / sqrt(3))) < 1e-6 ? Vs2[1]^2 :
                            abs(ω - (mid + half / sqrt(3))) < 1e-6 ? Vs2[2]^2 : 0.0) / half,
                     mid - half, mid + half)
        h0sp = coupled_h0(f2, 2)
        G0 = mb_generator(h0sp)
        dim = 2^(3)
        function fock_ed(order, N)
            δτ = β / N
            kτ(α) = Diagonal([exp(-α * εd * (J & 1)) for J in 0:dim-1])
            Scell = order == 1 ? kτ(δτ) * exp(-δτ .* G0) :
                    kτ(δτ / 2) * exp(-δτ .* G0) * kτ(δτ / 2)
            d = zeros(ComplexF64, dim, dim); dd = zeros(ComplexF64, dim, dim)
            for jb in 0:dim÷2-1; d[2jb+1, 2jb+2] = 1.0; dd[2jb+2, 2jb+1] = 1.0; end
            Z = real(tr(Scell^N))
            n̄ = real(tr(dd * d * Scell^N) / Z)
            Spow = [Matrix{ComplexF64}(I, dim, dim)]
            for _ in 1:N; push!(Spow, Spow[end] * Scell); end
            G = [1 - n̄; [real(tr(Spow[N-m+1] * d * Spow[m+1] * dd) / Z) for m in 1:N-1]]
            Z, n̄, G
        end
        for order in (1, 2), N in (2, 4)
            Zed, ned, Ged = fock_ed(order, N)
            Zd, nd, Gd = direct_trotter_sp(h0sp, εd, order, N, β)
            @test abs(real(Zd) - Zed) < 1e-10
            @test abs(nd - ned) < 1e-12
            @test maximum(abs.(Gd .- Ged)) < 1e-12
        end
    end

    # ---- T1 对齐：包核装配 == 直接 Trotter（连续谱 GL-32 离散，规模做大） ----
    @testset "T1 对齐：exact_fermionic_Δτ 装配 == 直接 Trotter（半圆 GL-32，N≤64）" begin
        f = semicircular(t_half)
        h0 = coupled_h0(f, 32)
        for N in (8, 32, 64), order in (1, 2)
            Zd, nd, Gd = direct_trotter_sp(h0, εd, order, N, β)
            Zs, ns, Gs = strictif_assembly(f, εd, N, β, IFD.EDToulouseSolver(32))
            # 严格 IF 路线的 Z = normdelta·Pf(B)（归一随核对象）；直接路线
            # Z = (−1)^{nN}det(A)。Z 随模式数指数大（~2ⁿᴹ 量级），断言用相对容差
            Zass = Zs
            worst = max(abs(Zass - real(Zd)) / abs(real(Zd)), abs(ns - nd),
                        maximum(abs.(Gs .- Gd)))
            worst > 1e-8 && println("  ✗ T1 order=$order N=$N: worst=$worst")
            @test abs(Zass - real(Zd)) < 1e-10 * abs(real(Zd))
            @test abs(ns - nd) < 1e-10
            @test maximum(abs.(Gs .- Gd)) < 1e-10
        end
    end

    # ---- T3 严格解比对：收敛到离散模型严格解，饱和于谱离散地板 ----
    # 结构性事实（本 testset 显式钉死）：对角 H_imp 的 τ 环上，一阶 Lie 与
    # 二阶 Strang 劈分**严格恒等**——踢相位全对易，闭合迹上边界相位滑动是
    # 规范变换（ED 金标准的"阶数无关性"已机器精度验证）⟹ 环可观测量的
    # 劈分误差从 O(δτ) 全局项开始就是纯规范，收敛从 δτ² 起步。
    # "二阶比一阶更准"的阶数分离因此属于实时开线几何（fermionic_realtime.jl），
    # 虚时无此分离。本段断言：装配/直接两路线随 N 二阶收敛到离散模型严格解
    # （h₀ 本征闭式），并饱和于连续严格解（ImpurityModelBase 闭式）的谱地板。
    @testset "T3 严格解比对：二阶收敛 + 谱地板饱和（半圆 GL-48，N≤64）" begin
        f = semicircular(t_half)
        M = 48
        h0 = coupled_h0(f, M)
        # 离散模型严格解（h₀ 本征闭式，无 Trotter 误差）。注意：模型哈密顿量
        # 的杂质 onsite = εd（包核的 h₀ 依 IF 分割约定取 ε_d = 0，εd 经踢进入
        # 离散演化；此处是非 Trotter 的**模型**精确解，必须含 εd）
        hf = copy(h0)
        hf[1, 1] = εd
        ev = eigen(Hermitian(hf))
        a2 = abs2.(ev.vectors[1, :])
        Gdisc(τ) = sum(a2[p] * exp(-ev.values[p] * τ) / (1 + exp(-β * ev.values[p]))
                       for p in 1:M+1)
        # 连续模型严格解（ImpurityModelBase 闭式）
        Gcont(τ) = fermionic_toulouse_Gτ_semicircular(τ; β=β, ϵ_d=εd, t=t_half)
        floor_G = maximum(abs(Gdisc(τ) - Gcont(τ)) for τ in (β / 4, β / 2))
        println("  谱离散地板（GL-48 vs 连续闭式，τ=β/4,β/2）= $(round(floor_G, sigdigits=3))")
        err = Float64[]
        for N in (4, 8, 16, 32, 64)
            # 阶数不变性：Lie/Strang 环可观测量恒等（结构事实，容差 = 舍入级）
            _, n1, G1 = direct_trotter_sp(h0, εd, 1, N, β)
            _, n2, G2 = direct_trotter_sp(h0, εd, 2, N, β)
            @test abs(n1 - n2) < 1e-12
            @test maximum(abs.(G1 .- G2)) < 1e-12
            Zs1, ns1, Gs1 = strictif_assembly(f, εd, N, β, IFD.EDToulouseSolver(M))
            @test abs(ns1 - n1) < 1e-10
            @test maximum(abs.(Gs1 .- G1)) < 1e-10
            e = maximum(abs(G1[m+1] - Gdisc(τ)) for (m, τ) in ((N ÷ 4, β / 4), (N ÷ 2, β / 2)))
            push!(err, e)
            # 对连续严格解的偏差（含地板）
            ec = maximum(abs(G1[m+1] - Gcont(τ)) for (m, τ) in ((N ÷ 4, β / 4), (N ÷ 2, β / 2)))
            println("  N=$N:  err_vs_disc=$e  err_vs_cont=$ec")
        end
        # 二阶收敛斜率（对离散严格解，N ∈ {8,16,32} 对数斜率 ≈ 2）
        s = [log2(err[i] / err[i+1]) for i in 2:3]
        println("  收敛斜率 = $(round.(s, digits=2))（δτ² ⟹ 2.0）")
        @test all(1.75 .< s .< 2.25)
        @test err[end] < err[end-1] < err[end-2]               # 单调收敛
        # 连续严格解比对：最细 N 饱和于地板（含 GL 求积偏差）
        ec64 = maximum(abs(direct_trotter_sp(h0, εd, 2, 64, β)[3][m+1] - Gcont(τ))
                       for (m, τ) in ((16, β / 4), (32, β / 2)))
        @test ec64 < 1e-5
        @test ec64 >= floor_G * 0.999
    end

end
println("费米虚时 ExactEqualDistantKernel 金标准通过 ✓（T2 ED 锚 + T1 对齐 + T3 严格解）")
