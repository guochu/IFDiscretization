# =============================================================================
# exact/strict_kernel.jl —— ExactIF 严格离散核的金标准检验（费米虚时）
#
# 金标准（用户定义）：严格离散 hybridization 核 K_bath（纯浴对象）+ 杂质
# 动能相位 x 的装配（B = −I + x·K_bath）得到的观测量（Z、n̄、G(τ_m)）
# 必须与同阶**直接 Trotter 演化**（JW Fock 多体链，同一批踢位排布）逐位一致。
#
# 被测对象 = 包输出：strict_fermionic_Δτ(f; β, N)（ExactIF 严格核）。
# 装配契约：B[ā_i, a_j] = −δ_ij + x·K_bath[i,j]，x = e^{−δτ·ε_d}；
# n̄ 与 G 由 B⁻¹ 元素提取（环节点反向序映射，见 strictkernel.jl 契约）。
#
# 阶数：对角 H_imp 下核与 x 排布方式有关。orders 1/2 覆盖一阶（全步踢）
# 与二阶（Strang 半步踢）——各自与同阶直接演化逐位一致。
# orders 4/6 的 Suzuki fractal 在虚时中含负步长（θ−4p < 0），
# exp(+|θ|·h₀) 数值爆炸——不适用于虚时核（实时过程张量不受此限）。
#
# 模型：2 模离散浴（Fock 8 维可承受直接演化），经 GL-2 节点技巧精确表示
# 为有界谱（GL-2 节点恰好落在离散模上 ⟹ 包内 ExactIF 核 == 离散链严格核）。
# =============================================================================
using Test, LinearAlgebra
using ImpurityModelBase
using IFDiscretization
import IFDiscretization as IFD
include(joinpath(@__DIR__, "..", "api", "_pfaffian_utils.jl"))
const _pf = pfaffian
include(joinpath(@__DIR__, "_anchor_utils.jl"))

const β = 2.0
const εd = 0.3
const Es = [-0.7, 1.3]
const V2s = [0.45^2, 0.6^2]
const M = length(Es)
const Vs = sqrt.(V2s)
const mid = (Es[1] + Es[2]) / 2
const half = (Es[2] - Es[1]) * sqrt(3) / 2
const fspec = bounded(ω -> (abs(ω - (mid - half / sqrt(3))) < 1e-6 ? V2s[1] :
                            abs(ω - (mid + half / sqrt(3))) < 1e-6 ? V2s[2] : 0.0) / half,
                      mid - half, mid + half)
const h0 = begin
    h = zeros(ComplexF64, M + 1, M + 1)
    for k in 1:M; h[1, 1+k] = h[1+k, 1] = Vs[k]; h[1+k, 1+k] = Es[k]; end
    h
end
const G0 = mb_generator(h0)
const dim = 2^(M + 1)

"n 阶直接 Trotter 演化（JW Fock 多体链）：Z、n̄、G(τ_m)"
function fock_ref(order::Int, N::Int)
    δτ = β / N
    kτ(α) = Diagonal([exp(-α * εd * (J & 1)) for J in 0:dim-1])
    function cell(order, θ)
        order == 1 && return kτ(θ) * exp(-θ .* G0)
        order == 2 && return kτ(θ/2) * exp(-θ .* G0) * kτ(θ/2)
        sub = order - 2
        p = θ / (4 - 4^(1/(order - 1)))
        r = cell(sub, p); r = cell(sub, p) * r; r = cell(sub, θ - 4p) * r
        r = cell(sub, p) * r; cell(sub, p) * r
    end
    Scell = cell(order, δτ)
    d = zeros(ComplexF64, dim, dim); dd = zeros(ComplexF64, dim, dim)
    for jb in 0:dim÷2-1; d[2jb+1, 2jb+2] = 1.0; dd[2jb+2, 2jb+1] = 1.0; end
    Z = real(tr(Scell^N))
    n̄ = real(tr(dd * d * Scell^N) / Z)
    Spow = [Matrix{ComplexF64}(I, dim, dim)]
    for _ in 1:N; push!(Spow, Spow[end] * Scell); end
    G = [1 - n̄; [real(tr(Spow[N-m+1] * d * Spow[m+1] * dd) / Z) for m in 1:N-1]]
    return Z, n̄, G
end

"装配契约：B = Δ 单体（测度/着装动能/Schur 杂交全部内含于核）；返回 Z（Pf）、n̄、G"
function pf_assembly(kernel, N)
    Δ = kernel.K
    B = zeros(ComplexF64, 2N, 2N)
    for i in 1:N, j in 1:N
        B[2i, 2j-1] += Δ[i, j]
        B[2j-1, 2i] -= Δ[i, j]
    end
    Z = pfaffian(B)
    Binv = inv(B)
    n̄ = 1 - Binv[2, 1]
    G = [1 - n̄; [-Binv[2(N-m+1), 1] for m in 1:N-1]]
    return Z, n̄, G
end

@testset "ExactIF 严格离散核金标准（费米虚时）" begin

    @testset "平移不变性（反周期环双向量结构）" begin
        for N in (3, 4, 7)
            kernel = strict_fermionic_Δτ(fspec; β=β, N=N, gfsolver=IFD.EigenSolver(2))
            K = kernel.K
            devF = maximum(abs(K[j+d, j] - K[1+d, 1]) for d in 0:N-1 for j in 1:N-d)
            devB = maximum(abs(K[i, i+d] - K[1, 1+d]) for d in 1:N-1 for i in 1:N-d)
            @test devF < 1e-13
            @test devB < 1e-13
        end
    end

    @testset "金标准：Z/n̄/G vs 同阶直接演化（orders 1/2 × N=1..8）" begin
        for order in (1, 2), N in 1:8
            x = exp(-εd * β / N)
            kernel = strict_fermionic_Δτ(fspec; β=β, N=N, order=order,
                                         imp_phase=x, gfsolver=IFD.EigenSolver(2))
            Zpf, n̄pf, Gpf = pf_assembly(kernel, N)
            Zfk, n̄fk, Gfk = fock_ref(order, N)
            @test abs(real(kernel.detBB * Zpf) - Zfk) < 1e-10
            @test abs(n̄pf - n̄fk) < 1e-12
            @test maximum(abs.(real.(Gpf) .- Gfk)) < 1e-12
        end
    end

    @testset "Z 符号公式一致性（c(N) 不随 order 变化）" begin
        for N in 1:8
            signs = Set{Int}()
            for order in (1, 2)
                kernel = strict_fermionic_Δτ(fspec; β=β, N=N, order=order,
                                             imp_phase=exp(-εd*β/N), gfsolver=IFD.EigenSolver(2))
                Zpf, _, _ = pf_assembly(kernel, N)
                Zfk, _, _ = fock_ref(order, N)
                push!(signs, round(Int, sign(real(kernel.detBB * Zpf) / Zfk)))
            end
            @test length(signs) == 1   # 同一 N 下各 order 同号（核与阶数无关）
        end
    end

    @testset "负对照：点值格林函数 ≠ 装配配对核" begin
        # pointwise_fermionic_Gτ 的条目是 H_hyb 精确演化的连续轮廓两点收缩
        #（格林函数点值，MPS-IF 列递归消费对象）——不是装配级配对核。
        # 对照：同一 Pfaffian 装配（B = 核条目单体），严格核逐位、
        # 点值显著偏离——定量暴露"格林函数点值 ≠ 配对核"。
        N = 3
        x = exp(-εd * β / N)
        _, n̄fk, Gfk = fock_ref(1, N)

        # 严格核（正对照，应逐位）
        kernel = strict_fermionic_Δτ(fspec; β=β, N=N, order=1,
                                     imp_phase=x, gfsolver=IFD.EigenSolver(2))
        B_s = zeros(ComplexF64, 2N, 2N)
        for i in 1:N, j in 1:N; B_s[2i, 2j-1] += kernel.K[i,j]; B_s[2j-1, 2i] -= kernel.K[i,j]; end
        Z_s = pfaffian(B_s); Binv_s = inv(B_s)
        n̄s = 1 - Binv_s[2, 1]
        Gs = [1 - n̄s; [-Binv_s[2(N-m+1), 1] for m in 1:N-1]]
        @test abs(n̄s - n̄fk) < 1e-12
        @test maximum(abs.(real.(Gs) .- Gfk)) < 1e-12

        # 点值（负对照，应偏离）
        pw = pointwise_fermionic_Gτ(fspec; β=β, N=N, gfsolver=IFD.EigenSolver(2))
        Δpw = Matrix(pw.data)
        B_pw = zeros(ComplexF64, 2N, 2N)
        for i in 1:N, j in 1:N; B_pw[2i, 2j-1] += Δpw[i,j]; B_pw[2j-1, 2i] -= Δpw[i,j]; end
        Z_pw = pfaffian(B_pw); Binv_pw = inv(B_pw)
        n̄pw = 1 - Binv_pw[2, 1]
        Gpw = [1 - n̄pw; [-Binv_pw[2(N-m+1), 1] for m in 1:N-1]]
        err_pw_n = abs(n̄pw - n̄fk)
        err_pw_G = maximum(abs.(real.(Gpw) .- Gfk))
        @test err_pw_n > 1e-2
        @test err_pw_G > 1e-2
        println("  负对照: err_n̄ = $(round(err_pw_n, sigdigits=3)), " *
                "err_G = $(round(err_pw_G, sigdigits=3))（点值 ≠ 配对核）")
    end

    @testset "对角 H_imp 的阶数无关性" begin
        # 对角 H_imp 的 kick 相位沿迹可对易滑动 ⟹ orders 1/2 给出相同的
        # n̄ 和 G（劈分阶数对观测量无影响）。非对角 H_imp 下此等式破坏。
        for N in (2, 4)
            _, n1, g1 = fock_ref(1, N)
            _, n2, g2 = fock_ref(2, N)
            @test abs(n1 - n2) < 1e-12
            @test maximum(abs.(g1 .- g2)) < 1e-12
        end
    end

end
println("ExactIF 严格离散核金标准检验通过 ✓")
