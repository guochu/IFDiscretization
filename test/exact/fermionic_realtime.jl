# =============================================================================
# exact/fermionic_realtime.jl —— 费米实时 ExactEqualDistantKernel 金标准
#
# 装配契约（2026-11-18 定谳，test 头注历史：εd≠0 装配 bug 三轮排查——
# IB 行缩放（Z 出虚部）→ 链条目顶点移位（Z 实但偏离 1）→ 带踢 Schur 定谳）：
#
#   主契约 = 补零 pair 网格 + 带踢 Schur（GTEMPO NativeExact `exactif_pipeline`
#   的证明配方）：`exact_fermionic_Δt_cells` 的 `kicks` 接口提供每 gap 幺正门
#   相位 x_g = e^{−iα_g·ε_d}；+ 支 gap 复合矩阵 = D(x)·W（杂质输出行 ×x），
#   − 支 = 伴随 W†D†（杂质输入列 ×x̄）。全因子幺正 ⟹
#   **Z = normdelta · det(装配) ≡ 1 机器精确（任意 ε_d、任意调度、任意阶数）**，
#   观测量 = 门序列 Γ(D·W) 的精确收缩。order 2 调度 = Strang 门链（半步踢两端）
#   ⟹ n̄ 对连续演化 δt² 收敛；order 1 的踢锚定在 gap 上 ⟹ δt¹（两阶在 n̄ 上
#   真实分离——pair 网格的踢位置被 cells 锚定，τ 环式循环等价不成立）。
#
#   装配（紧凑 (ā,a) 子块）：A = data.Δ[ā 行, a 列] + 测度 −1 + 单位链 + 闭合边
#   (ā⁻_N, a⁺_N)·(−1) + 真空 ρ₀ 常数；**n̄_vac = Z·(1 − A⁻¹[row(ā⁻_N), col(a⁺_N)])**
#   （(order,N) ∈ {1,2}×{2,3} 四点 machine 钉定；Z·invA = ρ[1,1] 通道）。
#
#   uniform API（exact_fermionic_Δt，无 kicks）：ε_d = 0 时同一公式机器锚定；
#   ε_d ≠ 0 时为顶点移位离散 Γ(W+(x−1)P₁)^N——Z 实数但偏离 1 O(δt)
#   （cell 复合矩阵非幺正），n̄ 通道为多 Wick 配对（非单元素），金标准不钉。
#
# 断言层次：
#   Z  证书   Z ≡ 1 机器（任意 ε_d/调度/谱；最锐利的严格性证书）
#   T2 定义锚 Fock ED（2 模 bath，dim=8）门序列链 vs IF 装配 n̄：机器精度
#   T1 对齐   大 N（≤16）IF vs ED 链：机器精度（同调度两实现）
#   T3 物理   n̄(N) 对连续极限的阶数标度（order 2 ⟹ δt²、order 1 ⟹ δt¹）
# =============================================================================

include(joinpath(@__DIR__, "_anchor_utils.jl"))

# ---------------- 调度表：补零 pair 网格（NativeExact trotter_factors 同款） ----------------

"""
    pair_grid(order, N, δt) -> (gates, cells)

每 gap 一个 Trotter 因子的补零表：`gates[g]` = gap g 的踢时长（0 = 恒等门），
`cells[g]` = gap g 的二次型 cell 时长（0 = 恒等 cell）；腿点数 = length+1。
order 2 = Strang 门链（半步踢两端）。order 4 的 Suzuki 回文表未含（实时
严格核金标准钉 orders 1/2；order 4 走 GTEMPO NativeExact 侧）。
"""
function pair_grid(order::Int, N::Int, δt::Real)
    N >= 1 || throw(ArgumentError("N 至少为 1"))
    if order == 1
        gates = fill(Float64(δt), N)
        cells = fill(Float64(δt), N)
        cells[end] = 0.0                     # 末 gap 仅恒等门（开放端在末时刻）
    elseif order == 2
        np = 2N + 1
        gates = zeros(np); cells = zeros(np)
        gates[1:2:end] .= vcat(δt / 2, fill(δt, N - 1), δt / 2)
        cells[2:2:end] .= δt
    else
        throw(ArgumentError("unsupported order $order (use 1 or 2)"))
    end
    return gates, cells
end

# ---------------- 紧凑装配 + n̄ 读出 ----------------

"""
    full_delta(kern) -> Matrix

测试侧稠密重建：ExactRealCorrelationFunction 的四块 → legpos 全矩阵
（4(N+1)²，行 = ā 腿、列 = a 腿；ifmatrix 层删除后由测试侧承担装配）。
"""
function full_delta(kern)
    n = size(kern, 1)
    lp(j0, br, ā) = 4 * j0 + (br === :+ ? 0 : 2) + (ā ? 2 : 1)
    Δ = zeros(ComplexF64, 4n, 4n)
    for j1 in 0:n-1, b1 in (:+, :-), j2 in 0:n-1, b2 in (:+, :-)
        Δ[lp(j1, b1, true), lp(j2, b2, false)] = branch(kern, b1, b2)[j1+1, j2+1]
    end
    return Δ
end

"""
    assemble(kΔ, npoints) -> A

紧凑 (ā,a) 装配（契约见头注）：行 = ā 腿、列 = a 腿的 2(npoints)² 子块，
+ 测度 −1 + 单位链 + 闭合边。真空 ρ₀ = 常数（无腿项）。
`kΔ` 为核的 legpos 空间 Δ 矩阵（节点 0..npoints−1）。
"""
function assemble(kΔ::Matrix{ComplexF64}, npairs::Int)
    lp(j0, br, ā) = 4 * j0 + (br === :+ ? 0 : 2) + (ā ? 2 : 1)
    ix(j0, br) = 2 * j0 + (br === :+ ? 1 : 2)
    n = npairs + 1
    A = kΔ[[lp(j0, br, true) for j0 in 0:npairs for br in (:+, :)],
           [lp(j0, br, false) for j0 in 0:npairs for br in (:+, :)]]
    for i in 1:2n
        A[i, i] -= 1
    end
    for g in 1:npairs
        A[ix(g, :+), ix(g-1, :+)] += 1
        A[ix(g-1, :-), ix(g, :-)] += 1
    end
    A[ix(npairs, :-), ix(npairs, :+)] += -1
    return A
end

"""真空 ρ₀ 的占据数：n̄ = Z·(1 − A⁻¹[row(ā⁻_N), col(a⁺_N)])（头注公式）。"""
function nbar_vacuum(A::Matrix{ComplexF64}, Z::Number, npairs::Int)
    ix(j0, br) = 2 * j0 + (br === :+ ? 1 : 2)
    return Z * (1 - inv(A)[ix(npairs, :-), ix(npairs, :+)])
end

"""IF 路线端到端：pair 网格调度 → 带踢核 → 装配 → (Z, n̄)。"""
function if_route(f, order::Int, N::Int, t::Real, β::Real, εd::Real, M::Int)
    δt = t / N
    gates, cells = pair_grid(order, N, δt)
    kicks = [exp(-im * g * εd) for g in gates]
    kernel, ndlt = exact_fermionic_Δt_cells(f; β=β, cells=cells, μ=0.0,
                                            gfsolver=IFD.EDToulouseSolver(M), kicks=kicks)
    A = assemble(full_delta(kernel), length(gates))
    Z = ndlt * det(A)
    return real(Z), real(nbar_vacuum(A, Z, length(gates)))
end

# ---------------- Fock ED 参照（门序列链，impurity = mode 1 = bit 0） ----------------

"""
    ed_chain(f, order, N, t, β, εd, nmodes; ρ0v=[1 0; 0 0]) -> (Z, n̄)

门序列的 Fock ED 参照：U = ∏_g Γ(D(gate_g))·Γ(W(cell_g))（gap 序），
Γ(W) = exp(−i·cell·mb_generator(h₀))（JW，mb_generator = 测试侧锚工具），
ρ₀ = ρ0v⊗ρ_bath（真空/占据/混合对角均可），n̄ 与 Z 皆为迹比值。
"""
function ed_chain(f::AbstractBoundedFunction, order::Int, N::Int, t::Real, β::Real,
                  εd::Real, nmodes::Int; ρ0v=ComplexF64[1 0; 0 0])
    δt = t / N
    gates, cells = pair_grid(order, N, δt)
    Es, Vsq, _ = IFD._quad_nodes(f, 0.0, nmodes)
    h0 = zeros(ComplexF64, nmodes + 1, nmodes + 1)
    for k in 1:nmodes
        h0[1, 1+k] = h0[1+k, 1] = Vsq[k]
        h0[1+k, 1+k] = Es[k]
    end
    dim = 1 << (nmodes + 1)
    nb = [1 / (1 + exp(β * e)) for e in Es]
    ρbath = zeros(ComplexF64, dim ÷ 2, dim ÷ 2)
    for b in 0:(dim ÷ 2 - 1)
        ρbath[b+1, b+1] = prod((1 - nb[k], nb[k])[(b >> (k - 1)) & 1 + 1] for k in 1:nmodes)
    end
    ρfull = kron(ρbath, ρ0v)
    Gmb = mb_generator(h0)
    U = Matrix{ComplexF64}(I, dim, dim)
    for g in eachindex(gates)
        D = Diagonal([exp(-im * gates[g] * εd * (J & 1)) for J in 0:dim-1])
        C = cells[g] == 0 ? Matrix{ComplexF64}(I, dim, dim) :
            exp(-im * cells[g] .* Gmb)
        U = D * C * U
    end
    R = U * ρfull * U'
    T = reshape(R, 2, dim ÷ 2, 2, dim ÷ 2)
    # n̄ = Tr[n̂·Uρ₀U†]/Tr[Uρ₀U†]（分母用同一 ρ₀——非幺正调度下迹不守恒）
    Z = sum(T[a, B, a, B] for a in 1:2 for B in 1:(dim ÷ 2))
    n̄ = sum(T[2, B, 2, B] for B in 1:(dim ÷ 2)) / Z
    return real(Z), real(n̄)
end

# ================================= 测试 =================================
const β = 2.0
const t_T = 0.8
const εd = 0.3

@testset "费米实时 ExactEqualDistantKernel 金标准（带踢 Schur 契约 vs 门序列 ED）" begin

    f2 = bounded(ω -> 1.0, -1.0, 1.0)                  # 2 模离散参照
    f16 = semicircular(1.0)                            # 连续谱（GL-16）

    # ---- Z 证书：全因子幺正 ⟹ Z ≡ 1 机器（任意 ε_d / 调度 / 谱） ----
    @testset "Z 证书：Z ≡ 1（任意 εd、orders 1/2）" begin
        for (f, M, tag) in ((f2, 2, "2模"), (f16, 16, "GL-16"))
            for order in (1, 2), N in (2, 3), εdv in (0.0, 0.3, -0.5)
                z, _ = if_route(f, order, N, t_T, β, εdv, M)
                @test abs(z - 1) < 1e-12
            end
        end
    end

    # ---- T2 定义锚：IF 装配 n̄ == 门序列 ED（机器精度） ----
    @testset "T2 定义锚：装配 n̄ == ED 门序列链（2 模，εd=0/0.3）" begin
        for order in (1, 2), N in (2, 3, 4), εdv in (0.0, εd)
            _, nif = if_route(f2, order, N, t_T, β, εdv, 2)
            _, ned = ed_chain(f2, order, N, t_T, β, εdv, 2)
            @test abs(nif - ned) < 1e-11
        end
        # 一般（含非对角）ρ₀ 的通道机器不在本文件契约内（顶点移位实验的教训：
        # 占据/混合通道为多 Wick 配对结构，单元素公式不成立）——
        # 见 test 头注与 debug 记录；GTEMPO 侧热态 ρ₀ 消费走其已验证管线。
    end

    # ---- T1 对齐：大 N 机器精度（同调度两实现） ----
    @testset "T1 对齐：大 N（≤16）IF vs ED 链" begin
        for order in (1, 2), N in (8, 16)
            _, nif = if_route(f2, order, N, t_T, β, εd, 2)
            _, ned = ed_chain(f2, order, N, t_T, β, εd, 2)
            @test abs(nif - ned) < 1e-10
        end
    end

    # ---- T3 物理：n̄(N) 对连续极限的阶数标度 ----
    # 结构事实：pair 网格的踢锚定在 gap 上（τ 环式循环等价不成立）⟹
    # order 2（Strang 门链）δt²、order 1 δt¹——两阶在 n̄ 上真实分离。
    @testset "T3 标度：order 2 δt²、order 1 δt¹（ED N=128 为连续代理）" begin
        Nref = 128
        _, ncont = ed_chain(f2, 2, Nref, t_T, β, εd, 2)
        for (order, slope_lo, slope_hi) in ((2, 1.75, 2.25), (1, 0.8, 1.2))
            Ns = (4, 8, 16, 32)
            err = Float64[]
            for N in Ns
                _, nif = if_route(f2, order, N, t_T, β, εd, 2)
                push!(err, abs(nif - ncont))
            end
            s = [log2(err[i] / err[i+1]) for i in 1:length(err)-1]
            println("  order=$order: n̄ 误差 = $(round.(err, sigdigits=3))  斜率 = $(round.(s, digits=2))")
            @test all(slope_lo .< s .< slope_hi)
            @test err[end] < err[1]
        end
    end

    # ---- uniform API：εd=0 机器锚 + εd≠0 Z 实数性质 ----
    @testset "uniform API（exact_fermionic_Δt）：εd=0 机器锚" begin
        for N in (3, 4)
            kernel, ndlt = exact_fermionic_Δt(f2; β=β, N=N, t=t_T,
                                              gfsolver=IFD.EDToulouseSolver(2))
            A = assemble(full_delta(kernel), N)
            Z = ndlt * det(A)
            @test abs(Z - 1) < 1e-12
            ix(j0, br) = 2 * j0 + (br === :+ ? 1 : 2)
            nif = Z * (1 - inv(A)[ix(N, :-), ix(N, :+)])
            # ED 对照 = W^N 链（uniform 顶点移位在 εd=0 的算符；pair 网格
            # order-1 链为 W^{N−1}·D 型——调度不同，勿混用）
            Es2, Vs2, _ = IFD._quad_nodes(f2, 0.0, 2)
            h0 = zeros(ComplexF64, 3, 3)
            for k in 1:2
                h0[1, 1+k] = h0[1+k, 1] = Vs2[k]
                h0[1+k, 1+k] = Es2[k]
            end
            dim = 8
            nbv = [1 / (1 + exp(β * e)) for e in Es2]
            ρbath = zeros(ComplexF64, 4, 4)
            for b in 0:3
                ρbath[b+1, b+1] = prod((1 - nbv[k], nbv[k])[(b >> (k - 1)) & 1 + 1]
                                       for k in 1:2)
            end
            U = exp(-im * (t_T / N) .* mb_generator(h0))
            for _ in 2:N
                U = exp(-im * (t_T / N) .* mb_generator(h0)) * U
            end
            R = U * kron(ρbath, ComplexF64[1 0; 0 0]) * U'
            T = reshape(R, 2, 4, 2, 4)
            ned = real(sum(T[2, B, 2, B] for B in 1:4))
            @test abs(nif - ned) < 1e-11
        end
        # εd≠0：Z 实数（虚部机器零）但偏离 1 O(δt)——顶点移位 cell 非幺正的方案性质
        for N in (3, 4)
            kernel, ndlt = exact_fermionic_Δt(f2; β=β, N=N, t=t_T,
                                              gfsolver=IFD.EDToulouseSolver(2))
            A = assemble(full_delta(kernel), N)
            for g in 1:N
                x = exp(-im * (t_T / N) * εd)
                lp(j0, br, ā) = 4 * j0 + (br === :+ ? 0 : 2) + (ā ? 2 : 1)
                ix(j0, br) = 2 * j0 + (br === :+ ? 1 : 2)
                # 顶点移位链骨架：单位链 1 → x（替换而非叠加）
                A[ix(g, :+), ix(g-1, :+)] += x - 1
                A[ix(g-1, :-), ix(g, :-)] += conj(x) - 1
            end
            Z = ndlt * det(A)
            @test abs(imag(Z)) < 1e-12
            @test 0.9 < abs(Z) < 1.1
        end
    end

    # ---- 结构化求解 vs 稠密参照（机器等价） ----
    @testset "结构化 Schur vs 稠密参照（机器等价）" begin
        for (order, N, εdv, M) in ((1, 3, 0.3, 3), (2, 2, 0.0, 4), (2, 3, -0.4, 3))
            δt = t_T / N
            gates, cells = pair_grid(order, N, δt)
            kicks = [exp(-im * g * εdv) for g in gates]
            k1 = IFD._exrt_strict_kernel(f16, cells, β, 0.0, M, kicks)
            k2 = IFD._exrt_strict_kernel_dense(f16, cells, β, 0.0, M, kicks)
            @test k1[1] ≈ k2[1] atol=1e-12 rtol=1e-9
            @test abs(k1[2] - k2[2]) < 1e-9 * abs(k2[2])
        end
    end

end
println("费米实时 ExactEqualDistantKernel 金标准通过 ✓（Z 证书 + T2 ED 锚 + T1 对齐 + T3 标度）")
