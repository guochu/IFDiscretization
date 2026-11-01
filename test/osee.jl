println("------------------------------------")
println("|   GaussianStates extension: OSEE  |")
println("------------------------------------")

# ← QuAPI/test/osee.jl 移植：只保留「GaussianStates extension」核心测试集
#   （corr → 高斯态构造 + OSEE，purify/解析锚，经 IFDiscretization 的
#   ifmatrix 层）。其余测试集（玻色 OSEE ↔ BosonicPT d→∞、稠密 IF 张量 vs
#   快速 OSEE、稠密费米 IF、AbstractContourIndex 访问、稠密 IF 乘法/margin、
#   稠密玻色 ADT/PT 矩阵元）以 QuAPI.Dense 稠密张量为锚/对象——随 Dense
#   留在 QuAPI/test/osee.jl（与本扩展共用同一构造器代码，过渡期由 QuAPI
#   侧继续验证 BCS 路径）。BCS 分支的 GrassmannIF SVD 交叉检查同此。

# 测试内小工具（只依赖导出接口，避免调用未导出内部函数）：
# 2n×2n 辛形式 Ω = ⊕ₖ [[0,1],[−1,0]]
function symplectic_form(n::Int)
    Ω = zeros(2n, 2n)
    for k in 1:n
        Ω[2k-1, 2k] = 1.0
        Ω[2k, 2k-1] = -1.0
    end
    return Ω
end

@testset "GaussianStates extension" begin
    # ---------------- Δ 提取约定（经构造器间接验证） ----------------
    # 新约定（GaussianStates v7：态直接由 corr 构造）：扩展读的 Δ 与
    # GrassmannHybridizationMatrix（顶点移位）同一 —— 独立性体现在
    # "不调用扩展自身的 Δ 提取"；约定本身的正确性由下方解析锚、
    # QuAPI 侧的稠密 SVD 对照与 performance/fermionicG4.jl 的 V⁴ 平台检验背书。
    function delta_ref(corr)
        C = Matrix(GrassmannHybridizationMatrix(corr).Δ)
        n = size(C, 1) ÷ 2
        return [C[2i, 2j-1] for i in 1:n, j in 1:n]
    end

    corrτ = fermionic_Δτ(semicircular(), β=5.0, N=8, μ=0.3)
    Δτ_ref = delta_ref(corrτ)

    corrt = fermionic_Δt(semicircular(), β=2.0, t=1.0, N=3, μ=0.2)
    Δt_ref = delta_ref(corrt)

    corrm = fermionic_Δm(semicircular(), β=2.0, Nτ=4, t=1.0, Nt=3, μ=0.2)
    Δm_ref = delta_ref(corrm)

    # ---------------- 费米：FermionicGaussianState(corr) → 系数态 OSEE ----------
    for (corr, Δref) in ((corrτ, Δτ_ref), (corrt, Δt_ref), (corrm, Δm_ref))
        s = FermionicGaussianState(corr)
        n = size(Δref, 1)
        @test nmodes(s) == 2n                        # doubled：模 1..n = ket、n+1..2n = bra
        # 约定：corr 入口 = 腿矩阵 [ā_p, a_q] 位放 Δ 的矩阵级入口（与旧
        # W 块测试同一约定声明：W[ā_p, a_q] = Δ_pq ⟺ C[2p, 2q−1] = Δ_pq）
        C = zeros(eltype(Δref), 2n, 2n)
        for i in 1:n, j in 1:n
            C[2i, 2j-1] = Δref[i, j]
        end
        sref = FermionicGaussianState(GrassmannHybridizationMatrix(basis(corr), C))
        @test covariance(sref) ≈ covariance(s) atol = 1e-13
        Γ = covariance(s)
        @test Γ isa Matrix{Float64}                # 任意向量 Majorana 协方差恒实
        @test Γ ≈ -transpose(Γ)
        @test maximum(abs, Γ^2 + I) < 1e-10        # 纯态
        @test ispure(s)
        @test entropy(s, collect(1:2n)) ≈ 0.0 atol = 1e-10   # 无 bipartition
        # 互补对称（纯态）：前缀割 = 余下割
        k = div(n, 2)
        @test entropy(s, vcat(1:k, n+1:n+k)) ≈
              entropy(s, vcat(k+1:n, n+k+1:2n)) atol = 1e-10
        @test_throws BoundsError entropy(s, [0])
        @test_throws ArgumentError entropy(s, [1, 1])
    end

    # 跨 site 单配对解析（矩阵级，验证 Δ → 态 → entropy 的约定链）：
    # Δ[ā₂, a₁] = d 只配对 ā₂–a₁ ⟹ site 割 1|2 劈开该配对，两割同熵
    d = 0.7 + 0.2im
    b2 = [GrassmannContourIndex(j, :τ, 1, c) for j in 1:2 for c in (false, true)]
    Cd = zeros(ComplexF64, 4, 4)
    Cd[4, 1] = d                                     # C[ā₂, a₁]
    s2 = FermionicGaussianState(GrassmannHybridizationMatrix(b2, Cd))
    ε = abs(d)^2 / (1 + abs(d)^2)
    h2(p) = -p * log(p) - (1 - p) * log(1 - p)
    @test entropy(s2, [1, 3]) ≈ h2(ε) atol = 1e-12
    @test entropy(s2, [2, 4]) ≈ h2(ε) atol = 1e-12
    @test entropy(s2, [1, 3]; α=2) ≈ -log(ε^2 + (1 - ε)^2) atol = 1e-12

    # ---------------- 玻色：BosonicGaussianState(corr) → 系数态 OSEE ----------
    # 玻色 bath 谱须全正（BE 分布要求 ε > μ），semicircular 含负能量不适用
    f0b = spectrum(ε -> exp(-ε), lb=0.5, ub=8.0)
    corr_b = bosonic_Δτ(f0b, β=2.0, N=4)
    sb = BosonicGaussianState(corr_b)
    # 约定：Δ 与 BosonicHybridizationMatrix（contour_ordering 行序）同源
    Δb = real.([branch(corr_b; b1=:τ, b2=:τ)[i, j] for i in 1:4, j in 1:4])
    hb = BosonicHybridizationMatrix([BosonicContourIndex(j, :τ) for j in 1:4], Δb)
    @test covariance(BosonicGaussianState(hb)) ≈ covariance(sb) atol = 1e-13
    # 显式 BosonicHybridizationMatrix 入口：split=false 与 correlation 入口一致；
    # split=true 的 BosonicSplitDelta(M1, M2) 是等时次序拆分，不是配对振幅，
    # 必须报错而不是静默拆成两个矩阵
    @test covariance(BosonicGaussianState(BosonicHybridizationMatrix(corr_b))) ≈
          covariance(sb) atol = 1e-13
    hsplit = BosonicHybridizationMatrix(corr_b; split=true)
    @test !(hsplit.Δ isa Tuple)                    # split 结果为专用类型，非裸元组
    @test_throws ArgumentError BosonicGaussianState(hsplit)
    @test_throws MethodError BosonicGaussianState(hsplit.Δ...)   # 不可 splat
    Γb = covariance(sb)
    @test Γb isa Matrix{Float64} && Γb ≈ transpose(Γb)
    Ω4 = symplectic_form(8)                         # 8 模（16×16）的辛形式
    ν = abs.(eigvals(Ω4 * Γb))
    @test maximum(abs.(ν .- 0.5)) < 1e-10           # 纯态：辛本征值全 ½
    @test ispure(sb)
    @test entropy(sb) ≈ 0.0 atol = 1e-10            # 纯态整态熵为 0（v7：entropy
    @test entropy(sb, collect(1:8)) ≈ 0.0 atol = 1e-10  # 对任意 GaussianState 有定义）
    # site p ↔ 模式对 [p, n+p]；纯态互补对称
    @test entropy(sb, vcat([1, 2], 4 .+ [1, 2])) ≈
          entropy(sb, vcat(3:4, 7:8)) atol = 1e-10
    @test_throws BoundsError entropy(sb, [0])
    @test_throws ArgumentError entropy(sb, [1, 1])
    # 复 Δ（实时 contour）报错
    @test_throws ArgumentError BosonicGaussianState(corrt)

    # 玻色解析（矩阵级）：Δ = diag(λ₁,λ₂) ⟹ M = diag(t₁,t₂)（t_k = e^{−λ_k}），
    # 系数态 = 两个独立双模压缩对 (c₁d₁)、(c₂d₂)；单模割劈开一对 ⟹ 单模热熵
    #（site 割 [p, n+p] 此时含完整配对对，熵为 0——交叉配对下的 site 割
    # 解析由费米单配对锚覆盖）
    λ1, λ2 = 1.1, 0.75
    hd = BosonicHybridizationMatrix([BosonicContourIndex(j, :τ) for j in 1:2],
                                    [λ1 0.0; 0.0 λ2])
    sd = BosonicGaussianState(hd)
    t1, t2 = exp(-λ1), exp(-λ2)
    n̄(x) = x^2 / (1 - x^2)
    SvN(x) = (n̄(x) + 1) * log(n̄(x) + 1) - n̄(x) * log(n̄(x))
    @test entropy(sd, [1]) ≈ SvN(t1) atol = 1e-12
    @test entropy(sd, [2]) ≈ SvN(t2) atol = 1e-12
    @test entropy(sd, [1, 2]) ≈ SvN(t1) + SvN(t2) atol = 1e-12   # ket 半空间
    @test entropy(sd, collect(1:4)) ≈ 0.0 atol = 1e-12
    @test entropy(sd, [1]; α=2) ≈ log((1 + t1^2) / (1 - t1^2)) atol = 1e-12
    # 旋转不变性：对称 Δ' = VΛVᵀ ⟹ M' = V diag(t) Vᵀ 奇异值不变，
    # ket|bra 割的熵只依赖 M 的奇异值（每个奇异值一对跨割双模压缩）
    θ = 0.37
    V = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    hr = BosonicHybridizationMatrix([BosonicContourIndex(j, :τ) for j in 1:2],
                                    V * [λ1 0.0; 0.0 λ2] * transpose(V))
    sr = BosonicGaussianState(hr)
    @test entropy(sr, [1, 2]) ≈ SvN(t1) + SvN(t2) atol = 1e-12
    @test entropy(sr, collect(1:4)) ≈ 0.0 atol = 1e-12
    @test ispure(sr)
end
