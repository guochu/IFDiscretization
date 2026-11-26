# =============================================================================
# exact/bcs_realtime.jl —— BCS 实时 ExactEqualDistantKernel 金标准（粒子字典 FormB 链）
#
# exact_bcs_Δt（粒子字典 FormB 链，2026-11-26 换代）的包内锚定阶梯：
#
#   T0  闭式 cell == Fock 精确提取：二次型 (u, v, w)、|c₀|²，且 Fock 重叠确为 Gaussian
#       （全单项式 P == c₀·exp(Q)）；热边闭式 == 4×4 矩阵指数。
#   T0c `_formb_chain_schur`（块三对角 + 角块 Woodbury）== 稠密求解（随机块）。
#   T0d 结构化装配 == **独立** Fock 稠密参照（_bcs_physical_ref.jl：Fock cell + Fock 热边
#       + 稠密 B\Cᵀ）：四块 × 四支对 + normdelta 机器精度。
#   T8  ED 端到端证书（冻结核字面放置语义 = GTEMPO AnomalousPairing 四键消费，最锐利）：
#       世界线消费（_worldline.jl）的 Z ≡ 1、n̄↑、n̄↓、⟨d↓d↑⟩、⟨d†↑d†↓⟩ == Fock ED
#       门序列链（4·4^K 维，_bcs_ed_chain.jl）：一般谱 × Δ ∈ {0, 实, 复} × orders {1, 2}
#       × N × ε_d——闭式 cell / 热边 / 结构化 Schur / ν 闭式 / 四块读出一并终裁。
#   T1  Δ=0 退化（全部谱）：uu == dd == exact_fermionic_Δt（谱 ε）逐位；ud/du ≡ 0。
#   T2  normdelta：Δ=0 时 ν = ν_ferm²；V→0 极限 ν → 1（核平凡）。
#   T3  结构：Δ≠0 Pf(G) 非零 / ud,du Δ-连续 / 复 Δ。
#   T4  守卫：μ≠0 / 零谱 / 相干态图表奇异（|c₀|² → 0）/ N < 1 / 离散谱直取。
#   T5  参照自检：退役 raw 收缩闭式（_bcs_raw_reference.jl）== 16 维 Fock ED 轮廓收缩。
#   T6  分派：Δt(BCSBath; scheme=ExactEqualDistantKernel(s)) == 直连 [1] 逐位。
#   T7  Pfaffian 主元交换回归：_formb_pfaffian 触发主元交换 == 定义式。
#
# 消费契约 = 冻结核字面放置语义（两味皆物理粒子；见 src/bcs/exact/realtime.jl 头注）。
# 铁律：参照侧（ED 链 / Fock 稠密参照 / raw 参照）禁任何谱恒等式加速；全套件单线程。
# =============================================================================

using Random: MersenneTwister
include(joinpath(@__DIR__, "_bcs_ed_utils.jl"))
include(joinpath(@__DIR__, "_bcs_raw_reference.jl"))

const β = 2.0
const δt_B = 0.3

@testset "BCS 实时 ExactEqualDistantKernel 金标准（粒子字典 FormB 链）" begin

    f1 = bounded(ω -> 0.15, -3.0, 3.0)                        # GL-1（单节点 ε=0）
    f2 = bounded(ω -> 1.0, -1.0, 1.0)                         # GL-2（对称）
    fd = DiracDelta(ω=0.8, α=0.4)                             # 非对称单模
    f3 = DiscreteSpectrum([-1.3, 0.2, 0.8], [0.2025, 0.11, 0.36])   # 三线（含负能量）
    fa = bounded(ω -> exp(-(ω - 0.3)^2 / 0.8), -2.0, 2.0)    # 非对称 GL-3
    Δs = (0.0, 0.3, 0.2 + 0.15im)
    solver(M) = IFD.EDToulouseSolver(M)
    bcs_kernel(f, M, N, Δv; t=N * δt_B) = exact_bcs_Δt(f; β=β, N=N, t=t, Δ=Δv, gfsolver=solver(M))
    blkerr(x, y) = maximum(maximum(abs.(branch(x, b1, b2) - branch(y, b1, b2)))
                           for b1 in (:+, :-), b2 in (:+, :-))
    kernerr(x, y) = max(blkerr(x.uu, y.uu), blkerr(x.dd, y.dd), blkerr(x.ud, y.ud), blkerr(x.du, y.du))
    spectra = [(f1, 1), (f2, 2), (fd, 1), (f3, 3), (fa, 3)]

    # ---- T0 闭式 cell / 热边 == Fock 精确提取 ----
    @testset "T0 闭式 cell/热边 == Fock 精确提取（u,v,w,|c₀|²；Fock 重叠为 Gaussian）" begin
        for (f, M) in spectra, Δv in Δs, dtv in (0.3, -0.3, 1.1)
            εs, Vs = ed_bath_modes(f, solver(M))
            c0, u, v, w, resid = bcs_cell_fock(εs, Vs, Δv, dtv)
            uc, vc, wc, c0sq = IFD._bcs_cell(εs, Vs, Δv, dtv)
            @test resid < 1e-11                                   # Fock 重叠 = c₀·exp(Q) 全单项式
            @test maximum(abs.(u - uc)) < 1e-11
            @test maximum(abs.(v - vc)) < 1e-11
            @test maximum(abs.(w - wc)) < 1e-11
            @test abs(c0sq - abs(c0)^2) < 1e-11
        end
        for ε in (-1.3, 0.0, 0.8), Δv in (0.0, 0.3, 0.2 + 0.15im), βv in (0.5, 2.0, 8.0, 40.0)
            gref = bcs_pair_gibbs_fock(ε, Δv, βv)
            @test maximum(abs.(IFD._bcs_pair_gibbs(ε, Δv, βv) - gref)) < 1e-12 * maximum(abs.(gref))
            @test real(IFD._bcs_pair_gibbs(ε, Δv, βv)[1, 1]) > 0          # 真空权重恒正（无抵消）
        end
    end

    # ---- T0c 块三对角 + 角块 Schur == 稠密（随机块） ----
    @testset "T0c _formb_chain_schur：块三对角 + 角块 Woodbury == 稠密求解" begin
        rng = MersenneTwister(20261126)
        for (Q, nb, m) in ((3, 1, 2), (4, 2, 3), (6, 3, 5)), _ in 1:3
            nb4 = 2nb
            asym(n) = (R = randn(rng, ComplexF64, n, n); R - transpose(R))
            D = [begin
                     Dq = asym(nb4)
                     for i in 1:nb
                         Dq[i, nb+i] -= 4; Dq[nb+i, i] += 4       # 测度主导
                     end
                     Dq
                 end for _ in 1:Q]
            ℓ = [randn(rng, ComplexF64, nb, nb) .* 0.3 for _ in 1:Q]
            R = randn(rng, ComplexF64, nb, nb) .* 0.2
            rws = collect(1:nb)
            cls = collect((Q - 1) * nb4 + nb + 1:Q * nb4)
            n = nb4 * Q
            B = zeros(ComplexF64, n, n)
            for q in 1:Q
                B[(q-1)*nb4+1:q*nb4, (q-1)*nb4+1:q*nb4] .= D[q]
            end
            for q in 2:Q
                Lq = [zeros(ComplexF64, nb, nb) ℓ[q]; zeros(ComplexF64, nb, nb4)]   # (out,in) 象限
                B[(q-1)*nb4+1:q*nb4, (q-2)*nb4+1:(q-1)*nb4] .= Lq
                B[(q-2)*nb4+1:(q-1)*nb4, (q-1)*nb4+1:q*nb4] .= -transpose(Lq)
            end
            for a in 1:nb, b in 1:nb
                B[rws[a], cls[b]] += R[a, b]; B[cls[b], rws[a]] -= R[a, b]
            end
            Ct = randn(rng, ComplexF64, n, m)
            X, ldet = IFD._formb_chain_schur(D, ℓ, R, rws, cls, Ct)
            @test maximum(abs.(B * X - Ct)) < 1e-10 * max(1.0, maximum(abs.(Ct)))
            @test abs(ldet - logabsdet(B)[1]) < 1e-10
        end
    end

    # ---- T0d 结构化装配 == 独立 Fock 稠密参照 ----
    @testset "T0d exact_bcs_Δt == 独立 Fock 稠密参照（四块 + normdelta）" begin
        for (f, M) in spectra, Δv in Δs, N in (1, 2, 3)
            εs, Vs = ed_bath_modes(f, solver(M))
            kern, nd = bcs_kernel(f, M, N, Δv)
            rk, rnd, _, _ = bcs_chain_dense_ref(εs, Vs, Δv, β, N, δt_B; fock=true)
            @test kernerr(kern, rk) < 1e-11
            @test abs(nd - rnd) < 1e-11 * max(1.0, abs(rnd))
        end
    end

    # ---- T8 ED 端到端证书（冻结核字面放置语义）：世界线消费 == Fock ED 门序列链 ----
    @testset "T8 ED 端到端证书（冻结核字面放置语义）：Z≡1、n̄↑、n̄↓、反常关联 == ED" begin
        for (f, M) in spectra, Δv in Δs, order in (1, 2), N in (1, 2, 3), εd in (0.3, -0.5, 0.8)
            cells, links = ed_native_schedule(order, N, δt_B)
            kern, nd = bcs_kernel(f, M, N, Δv)
            Z, nu, ndn, a1, a2 = bcs_worldline_particle(kern, nd, N, links, εd)
            εs, Vs = ed_bath_modes(f, solver(M))
            _, nue, nde, a1e, a2e = bcs_ed_chain(εs, Vs, Δv, β, cells, links, εd)
            @test abs(Z - 1) < 1e-11
            @test abs(nu - nue) < 1e-11
            @test abs(ndn - nde) < 1e-11
            @test abs(a1 - a1e) < 1e-11
            @test abs(a2 - a2e) < 1e-11
        end
    end

    # ---- T1 Δ=0 退化（全部谱）：uu == dd == 费米核逐位、ud/du ≡ 0 ----
    @testset "T1 Δ=0 退化：uu == dd == 费米核（谱 ε）逐位、ud/du ≡ 0" begin
        for (f, M) in spectra, N in (1, 2, 4)
            kern, nd = bcs_kernel(f, M, N, 0.0)
            fkern, fnd = exact_fermionic_Δt(f; β=β, N=N, t=N * δt_B, μ=0, gfsolver=solver(M))
            @test blkerr(kern.uu, fkern) < 1e-12
            @test blkerr(kern.dd, fkern) < 1e-12
            @test maximum(maximum(abs.(branch(kern.ud, b1, b2))) for b1 in (:+, :-), b2 in (:+, :-)) == 0.0
            @test maximum(maximum(abs.(branch(kern.du, b1, b2))) for b1 in (:+, :-), b2 in (:+, :-)) == 0.0
            @test abs(nd - fnd^2) < 1e-11 * max(1.0, abs(fnd^2))     # T2：normdelta = 费米²
        end
    end

    # ---- T2 normdelta：V→0 极限 ν → 1（核平凡；Δ≠0、cell 真空振幅 ≠ 1） ----
    @testset "T2 normdelta：V→0 极限 ν → 1（Δ≠0）" begin
        for N in (1, 3), Δv in (0.3, 0.2 + 0.15im)
            f = DiracDelta(ω=0.8, α=1e-6)
            εs, Vs = ed_bath_modes(f, solver(1))
            kern, nd = bcs_kernel(f, 1, N, Δv)
            _, rnd, G, _ = bcs_chain_dense_ref(εs, Vs, Δv, β, N, δt_B; fock=false)
            @test abs(nd - rnd) < 1e-11 * max(1.0, abs(rnd))
            ν = rnd / pf_std(G)
            @test abs(ν - 1) < 1e-6
        end
    end

    # ---- T3 结构：Δ≠0 Pf 非零 / Δ-连续 / 复 Δ ----
    @testset "T3 结构：Pf 非零、Δ-连续、复 Δ" begin
        for Δv in (0.3, 0.2 + 0.15im)
            kern, nd = bcs_kernel(fd, 2, 3, Δv; t=0.6)
            @test abs(nd) > 1e-12
            @test all(isfinite, branch(kern.uu, :+, :+))
            @test all(isfinite, branch(kern.ud, :+, :-))
            @test all(isfinite, branch(kern.du, :-, :+))
            @test all(isfinite, branch(kern.dd, :-, :-))
        end
        kern1, _ = bcs_kernel(fd, 2, 3, 0.3; t=0.6)
        kern2, _ = bcs_kernel(fd, 2, 3, 0.3 + 1e-7; t=0.6)
        @test blkerr(kern1.ud, kern2.ud) < 1e-5
        @test blkerr(kern1.du, kern2.du) < 1e-5
        # 小 Δ 下反常交叉块幅度随 Δ 缩（连续归零的粗界）
        kern3, _ = bcs_kernel(fd, 2, 3, 0.001; t=0.6)
        @test maximum(abs.(branch(kern3.ud, :+, :+))) < 0.05
        # 点 1（ρ₀ 端）置零，核支撑 = 点 2..N+1（与费米核同构）
        for blk in (kern1.uu, kern1.dd, kern1.ud, kern1.du), b1 in (:+, :-), b2 in (:+, :-)
            Bk = branch(blk, b1, b2)
            @test count(!iszero, Bk[1, :]) + count(!iszero, Bk[:, 1]) == 0
            @test any(!iszero, Bk[end, :]) || any(!iszero, Bk[:, end])
        end
    end

    # ---- T4 守卫：μ≠0 / 零谱 / N < 1 / 相干态图表奇异 / 离散谱直取 ----
    @testset "T4 守卫" begin
        @test_throws ArgumentError exact_bcs_Δt(f1; β=β, N=2, t=0.6, Δ=0.3, μ=0.5,
                                                gfsolver=solver(2))
        @test_throws ArgumentError exact_bcs_Δt(bounded(ω -> 0.0, -1.0, 1.0); β=β, N=2,
                                                t=0.6, Δ=0.3, gfsolver=solver(4))
        @test_throws ArgumentError exact_bcs_Δt(f1; β=β, N=0, t=0.6, Δ=0.3, gfsolver=solver(2))
        # 相干态图表奇异：纯配对 cell 的真空振幅 cos(|Δ|·δt) 在 |Δ|·δt = π/2 处过零
        @test_throws ArgumentError IFD._bcs_cell([0.0], [0.0], 1.0, π / 2)
        # 离散谱直取（不抛 GL 求积错误）
        kern, nd = exact_bcs_Δt(DiracDelta(ω=0.5, α=0.3); β=β, N=2, t=0.6, Δ=0.3,
                                gfsolver=solver(8))
        @test abs(nd) > 1e-12
    end

    # ---- T5 参照自检：退役 raw 收缩闭式 == 16 维 Fock ED（参照面自证） ----
    @testset "T5 raw 参照自检：raw 闭式 == Fock ED（1 模，实/复 Δ）" begin
        Wb = 3.0
        fr = bounded(ω -> 0.15, -Wb, Wb)
        edmake = bcs_ed_factory(fr, β)
        N, t = 3, 0.6
        for Δv in (0.3, 0.2 + 0.15im)
            ed, opH, ρ = edmake(Δv; Nt=N, t=t, Nτ=1)
            ex = raw_reference_bcs_Δt(fr; β=β, N=N, t=t, Δ=Δv, gfsolver=solver(1))
            worst = 0.0
            for (blk, p, q) in bcs_blocks(ex), b1 in (:+, :-), b2 in (:+, :-)
                B = branch(blk, b1, b2)
                for i in 0:N, j in 0:N
                    worst = max(worst, abs(B[i+1, j+1] - ed((b1, i), (b2, j), p, q)))
                end
            end
            @test worst < 1e-12
        end
    end

    # ---- T6 分派：统一入口 == 直连 [1] 逐位 ----
    @testset "T6 分派：Δt(BCSBath; scheme=Exact) == 直连" begin
        bC = BCSBath(f2, β=β, μ=0.0, Δ=0.4)
        e = ExactEqualDistantKernel(solver(8))
        x = Δt(bC, N=3, t=0.6, scheme=e)
        y = exact_bcs_Δt(f2; β=β, N=3, t=0.6, Δ=0.4, gfsolver=solver(8))
        @test x isa BCSCorrelationFunction{ExactRealCorrelationFunction}
        @test x.uu == y[1].uu && x.dd == y[1].dd && x.ud == y[1].ud && x.du == y[1].du
        # QuAPI 分派不受影响
        q = Δt(bC, N=3, t=0.6)
        @test q isa BCSCorrelationFunction
    end

    # ---- T7 Pfaffian 主元交换回归（2026-11-25 修复：列交换空操作 + 无符号追踪） ----
    @testset "T7 Pfaffian 主元交换回归：触发交换 == 定义式" begin
        pf_ref(A::Matrix{ComplexF64}) = begin          # 定义式（首行展开）
            n = size(A, 1)
            n == 0 && return one(ComplexF64)
            sum((-1.0)^j * A[1, j] * pf_ref(A[[i for i in 2:n if i != j], [i for i in 2:n if i != j]])
                for j in 2:n)
        end
        sign_of(n) = (-1.0)^(n ÷ 2)                     # 包内约定 = (−1)^{n/2}·Pf
        A4 = zeros(ComplexF64, 4, 4)                    # A[2,1] = 0 ⟹ 首步即触发交换
        A4[3, 1] = 1; A4[4, 2] = 2; A4[4, 3] = 3
        A4 = A4 - transpose(A4)
        @test IFD._formb_pfaffian(A4) ≈ sign_of(4) * pf_ref(A4) atol=1e-14
        @test pf_ref(A4) ≈ -2 atol=1e-14
        for n in (6, 8, 10), trial in 1:8
            R = randn(ComplexF64, n, n); A = R - transpose(R)
            A[2, 1] = 0; A[1, 2] = 0                     # 人造零主元
            A[n, n-1] = 0; A[n-1, n] = 0
            @test IFD._formb_pfaffian(A) ≈ sign_of(n) * pf_ref(A) atol=1e-10 * (1 + abs(pf_ref(A)))
        end
    end

end
println("BCS 实时金标准通过 ✓（T0 闭式 cell/热边 == Fock + T0c 结构化 Schur + T0d 独立 Fock 稠密参照 + T8 ED 端到端证书（冻结核字面放置语义）+ T1 Δ=0 退化 + T2 ν 锚 + T3/T4 结构守卫 + T5 raw 参照自检 + T6 分派 + T7 Pfaffian 回归）")
