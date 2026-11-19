# =============================================================================
# exact/fermionic_mixedtime.jl —— 费米混合 ExactEqualDistantKernel 金标准
#
# 契约（2026-11-19 装配级换代，← QuAPI/src/exactfermionic.jl `exact_fermionic_Δm`
# 母本移植 + 带踢复合 + 环状结构化求解）：
#
#   轮廓 = 单环 τ₁→…→τ_{Nτ}→(+,1)→…→(+,Nt+1)→(−,Nt+1)→…→(−,1)→τ₁；τ cells
#   （含结合 cell τ_{Nτ}→+₁）用 W^τ = e^{−δτh}、实时 ± cells 用 W^t = e^{−iδth}
#   （− 支伴随）；(+,Nt+1)→(−,Nt+1) 闭合边费米迹负号、(−,1)→τ₁ 环回边权 1。
#   腿基同冻结核（contour_ordering 混合序），返回
#   (GrassmannHybridizationMatrix, normdelta)，normdelta = (−1)^{n(K+1)}·det(BB)/Zb
#   （(−1)^{nK} 折叠 Berezin 符号 × impurity 折叠 (−1)^n——本 det 收缩机器相对
#   QuAPI Zvalue 机器的约定差，K=1/K=2 × 奇偶 n 四点机器钉定）。
#
#   **Keldysh 段在算符层面精确求和相消（含任意幺正 kicks）：P₊P₋ = I**
#   ⟹ 全迹收缩 Z = τ 环踢乘积的精确迹（与 Keldysh 段离散化无关的机器恒等式）。
#   BB bath 块沿轮廓构成单环 ⟹ 环状结构求解 O(n·K²·Limp + K³)，
#   det(BB) = (−1)^{Kn}·det(I_K − Π)（稠密参照机器等价）。
#
# 断言层次：
#   Z  证书（εd=0） Z == Tr_Fock[e^{−β·h₀-二次量子化}]/Z_bath 机器精确
#   Z  证书（kicks） Z == τ 环踢乘积 ED 迹 /Z_bath 机器精确（Keldysh 段独立）
#   T1 对齐         大 Nτ/Nt 机器精度（同调度 ED 链）
#   结构等价        结构化 vs 稠密参照（Δ 与 normdelta 机器等价）
#   腿基契约        basis == contour_ordering 混合序（可替换冻结核）
# =============================================================================

using Test, LinearAlgebra
using ImpurityModelBase
using IFDiscretization
import IFDiscretization as IFD

include(joinpath(@__DIR__, "_anchor_utils.jl"))

"""
    compact_delta(kern, Nτ, Nt) -> Matrix

测试侧稠密重建：ExactMixedCorrelationFunction 的九块 → 轮廓链序紧凑矩阵
（n×n，行 = ā 腿、列 = a 腿；ifmatrix 层删除后由测试侧承担装配）。
"""
function compact_delta(kern, Nτ::Int, Nt::Int)
    n = Nτ + 2 * (Nt + 1)
    pos(br, j) = br === :τ ? j : (br === :+ ? Nτ + 2j - 1 : Nτ + 2j)
    chain = [(br, j) for j in 1:Nτ for br in (:τ,)]
    append!(chain, [(br, j) for j in 1:Nt+1 for br in (:+, :-)])
    A = zeros(ComplexF64, n, n)
    for (c1, (b1, j1)) in enumerate(chain), (c2, (b2, j2)) in enumerate(chain)
        A[c1, c2] = branch(kern, b1, b2)[j1, j2]
    end
    return A
end

"紧凑 (ā 行, a 列) 装配：测度 −1 + 单位链 + 两条闭合边 + 真空 ρ₀ 常数"
function mixed_assemble(kΔ::Matrix{ComplexF64}, Nτ::Int, Nt::Int)
    n = Nτ + 2 * (Nt + 1)
    A = copy(kΔ)
    for c in 1:n
        A[c, c] -= 1
    end
    A[Nτ+1, Nτ] += 1                                  # 结合 cell（τ_Nτ → +₁）
    A[n, n-1] += -1                                   # Keldysh 迹闭合
    A[1, Nτ+2] += 1                                   # (−,1)→τ₁ 环回
    for c in 2:Nτ
        A[c, c-1] += 1
    end
    for jj in 1:Nt
        A[Nτ+2jj+1, Nτ+2jj-1] += 1                    # + 链
        A[Nτ+2jj, Nτ+2jj+2] += 1                      # − 链
    end
    return A
end

"IF 路线端到端：(Z, normdelta)；kicksτ 长度 Nτ（轮廓序），kickst 长度 Nt"
function mixed_route(f, Nτ::Int, Nt::Int, β::Real, t::Real, M::Int;
                     kicksτ=ones(Nτ), kickst=ones(Nt))
    kern, ndlt = exact_fermionic_Δm(f; β=β, Nτ=Nτ, t=t, Nt=Nt,
                                    gfsolver=IFD.EDToulouseSolver(M),
                                    kicksτ=ComplexF64.(kicksτ), kickst=ComplexF64.(kickst))
    Z = ndlt * det(mixed_assemble(compact_delta(kern, Nτ, Nt), Nτ, Nt))
    return kern, real(Z)
end

"ED 参照：τ 环踢乘积的 Fock 迹（Keldysh 段精确相消，不进入参照）；
impurity = mode 1 = bit 0（mb_generator/JW 约定）"
function tau_ring_ed(f, Nτ::Int, β::Real, εd::Real, nmodes::Int;
                     kicksτ=ones(Nτ))
    Es, Vsq, _ = IFD._quad_nodes(f, 0.0, nmodes)
    δτ = β / Nτ
    h0 = zeros(ComplexF64, nmodes + 1, nmodes + 1)
    for k in 1:nmodes
        h0[1, 1+k] = h0[1+k, 1] = Vsq[k]
        h0[1+k, 1+k] = Es[k]
    end
    dim = 1 << (nmodes + 1)
    U = Matrix{ComplexF64}(I, dim, dim)
    for j in 1:Nτ
        D = Diagonal([exp(-δτ * εd * (J & 1)) for J in 0:dim-1])
        U = D * exp(-δτ .* mb_generator(h0)) * U
    end
    return tr(U)
end

const β = 2.0
const t_T = 0.6
const εd = 0.3

@testset "费米混合 ExactEqualDistantKernel 金标准（单环带踢 Schur，Keldysh 段精确相消）" begin

    f2 = bounded(ω -> 1.0, -1.0, 1.0)                  # 2 模离散参照
    f1 = bounded(ω -> 1.0, -1.0, 1.0)                  # 1 模（prefactor 符号钉定）

    # ---- Z 证书（εd = 0）：Z == Tr_Fock[e^{−β·h₀}]/Z_bath（奇偶 n × K=1/K=2） ----
    @testset "Z 证书 εd=0：Tr[(B^τ)^Nτ]/Zb（K=1/K=2 × 奇偶 n）" begin
        for (f, M, tag) in ((f1, 1, "K=1"), (f2, 2, "K=2"))
            # Zref 经 tau_ring_ed 内部构造（h₀ 与核内 GL 求积精确同源）；
            # εd=0 时 τ 环乘积 = e^{−β·g}，迹与环长无关
            Zref = real(tau_ring_ed(f, 4, β, 0.0, M)) /
                   prod(1 .+ exp.(-β .* IFD._quad_nodes(f, 0.0, M)[1]))
            for (Nτ, Nt) in ((2, 2), (3, 2), (2, 3), (3, 3))
                _, Z = mixed_route(f, Nτ, Nt, β, t_T, M)
                @test abs(Z - Zref) < 1e-10
            end
        end
    end

    # ---- Z 证书（kicks）：Z == τ 环踢乘积 ED 迹 /Zb（εd≠0，Keldysh 段独立） ----
    @testset "Z 证书 kicks：τ 环踢乘积（εd≠0，奇偶 n）" begin
        for (f, M) in ((f1, 1), (f2, 2)), (Nτ, Nt) in ((2, 2), (3, 3), (2, 3))
            δτ = β / Nτ
            δt = t_T / Nt
            kicksτ = [exp(-δτ * εd) for _ in 1:Nτ]
            kickst = [exp(-im * δt * εd) for _ in 1:Nt]
            _, Z = mixed_route(f, Nτ, Nt, β, t_T, M; kicksτ=kicksτ, kickst=kickst)
            Zed = real(tau_ring_ed(f, Nτ, β, εd, M; kicksτ=kicksτ)) /
                  prod(1 .+ exp.(-β .* IFD._quad_nodes(f, 0.0, M)[1]))
            @test abs(Z - Zed) < 1e-10
        end
        # 非均匀复相位 kicks（核与 |x|=1 相位值无关，约定更严的检查）
        Nτ, Nt = 3, 2
        δτ = β / Nτ
        kicksτ = [exp(-im * 0.7 * j) for j in 1:Nτ]
        kickst = [exp(im * 0.4 * j) for j in 1:Nt]
        _, Z = mixed_route(f2, Nτ, Nt, β, t_T, 2; kicksτ=kicksτ, kickst=kickst)
        Es, Vs, _ = IFD._quad_nodes(f2, 0.0, 2)
        h0 = zeros(ComplexF64, 3, 3)
        for k in 1:2
            h0[1, 1+k] = h0[1+k, 1] = Vs[k]
            h0[1+k, 1+k] = Es[k]
        end
        dim = 8
        U = Matrix{ComplexF64}(I, dim, dim)
        for j in 1:Nτ
            D = Diagonal([exp(-im * 0.7 * j * (J & 1)) for J in 0:dim-1])
            U = D * exp(-δτ .* mb_generator(h0)) * U
        end
        Zed = real(tr(U)) / prod(1 .+ exp.(-β .* Es))
        @test abs(Z - Zed) < 1e-10
    end

    # ---- T1 对齐：大 Nτ/Nt 机器精度（同调度 ED） ----
    @testset "T1 对齐：大 Nτ/Nt（≤8）" begin
        for (Nτ, Nt) in ((8, 8), (8, 3))
            δτ = β / Nτ
            kicksτ = [exp(-δτ * εd) for _ in 1:Nτ]
            kickst = [exp(-im * (t_T / Nt) * εd) for _ in 1:Nt]
            _, Z = mixed_route(f2, Nτ, Nt, β, t_T, 2; kicksτ=kicksτ, kickst=kickst)
            Zed = real(tau_ring_ed(f2, Nτ, β, εd, 2; kicksτ=kicksτ)) /
                  prod(1 .+ exp.(-β .* IFD._quad_nodes(f2, 0.0, 2)[1]))
            @test abs(Z - Zed) < 1e-9
        end
    end

    # ---- 结构化 vs 稠密参照（机器等价） ----
    @testset "结构化 vs 稠密参照" begin
        for (Nτ, Nt) in ((2, 2), (3, 2), (2, 3), (3, 3)), withkicks in (false, true)
            δτ = β / Nτ
            δt = t_T / Nt
            kicksτ = withkicks ? ComplexF64.(exp(-δτ * εd) for _ in 1:Nτ) :
                     ones(ComplexF64, Nτ)
            kickst = withkicks ? ComplexF64.(exp(-im * δt * εd) for _ in 1:Nt) :
                     ones(ComplexF64, Nt)
            k1, n1 = exact_fermionic_Δm(f2; β=β, Nτ=Nτ, t=t_T, Nt=Nt,
                                        gfsolver=IFD.EDToulouseSolver(2),
                                        kicksτ=kicksτ, kickst=kickst)
            k2, n2 = IFD._exrt_mixed_kernel_dense(f2, β, Nτ, t_T, Nt, δτ, 0.0, 2,
                                                  kicksτ, kickst)
            @test k1 ≈ k2 atol=1e-12 rtol=1e-9   # 结构化/稠密舍入级差异
            @test abs(n1 - n2) < 1e-9 * abs(n2)
        end
    end

    # ---- 块尺寸契约：节点序同冻结核 contour_ordering 混合序（可替换冻结核） ----
    @testset "块尺寸契约：混合 contour 序" begin
        kern, _ = exact_fermionic_Δm(f2; β=β, Nτ=3, t=t_T, Nt=2,
                                     gfsolver=IFD.EDToulouseSolver(2))
        @test size(kern.Gττ) == (3, 3)
        @test all(m -> size(m) == (3, 3), (kern.Gτ₊, kern.Gτ₋, kern.G₊τ, kern.G₋τ))
        @test all(m -> size(m) == (3, 3),
                  (kern.G₊₊, kern.G₊₋, kern.G₋₊, kern.G₋₋))
        @test branch(kern, :τ, :+) === kern.Gτ₊   # branch = 字段直取（无损）
    end

end
println("费米混合 ExactEqualDistantKernel 金标准通过 ✓（Z 证书 ×2 + T1 + 结构等价 + 腿基契约）")
