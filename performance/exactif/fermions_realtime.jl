# =============================================================================
# exactif/fermions_realtime.jl —— ExactIF 严格离散 IF 的金标准误差标度（实时）
#
# 金标准：严格离散 IF（过程张量 if_path_values，纯浴对象、不含 H_imp）
# + 对角 kick 收缩（ε_d/U 只在收缩侧）⇒ ρ_imp 与同劈分直接 n 阶 Trotter
# 演化（trotter_exact，多体直乘）逐位一致。
#
# 误差标度：err(N) = max|ρ_contract − ρ_direct|。ExactIF 的 err(N) 是
# 机器精度平线（1e-14 量级，不随 δt 增长）——这就是"严格"的定量含义：
# 离散 IF 相对同一劈分直接演化零误差，Trotter 误差只存在于"离散 vs 连续"
# 的另一轴上（由 δt 扫描的连续参照另测，属冻结核/连续极限对比）。
#
# CSV: results/fermions_realtime_scaling.csv
# 运行：julia --startup-file=no --project=. performance/exactif/fermions_realtime.jl
# =============================================================================
using IFDiscretization, ImpurityModelBase, LinearAlgebra, Printf

const β = 2.0
const Δt = 0.15
const Es = [-0.7, 1.3]
const Vs = [0.45, -0.6im]

function main(; Ns=[1, 2, 3, 4, 5], cases=[((0.0, 0.0), 0.0), ((0.3, -0.5), 1.2), ((-0.7, 0.4), 0.8)])
    io = stdout
    println(io, "=" ^ 74)
    println(io, "ExactIF 严格离散 IF 金标准误差标度（费米实时，Ng–Reichman 过程张量）")
    @printf(io, "浴：2 模 E=[%.1f,%.1f]，β=%.1f，Δt=%.2f\n", Es..., β, Δt)
    println(io, "err(N) = max|ρ_contract − ρ_direct|；严格 ⇒ 机器精度平线")
    println(io, "=" ^ 74)
    out = joinpath(@__DIR__, "results"); mkpath(out)
    open(joinpath(out, "fermions_realtime_scaling.csv"), "w") do f
        println(f, "case,eps_up,eps_dn,U,N,err_max,err_trace")
        for (i, (εs, U)) in enumerate(cases)
            println(io, "\n[case $i: ε = ($(εs[1]), $(εs[2])), U = $U]")
            for N in Ns
                r = if_path_values(Es, Vs, β, N, Δt)
                ρif = contract_impurity(r.Ivals; N=N, Δt=Δt, εs=εs, U=U)
                ρex = trotter_exact(Es, Vs, β, N, Δt; εs=εs, U=U)
                ez = maximum(abs.(ρif - ρex))
                et = abs(tr(ρif) - tr(ρex))
                @printf(io, "  N=%d:  err_ρ = %.3e  err_tr = %.3e\n", N, ez, et)
                println(f, "$i,$(εs[1]),$(εs[2]),$U,$N,$ez,$et")
            end
        end
    end
    println(io, "\nCSV 已写入 results/fermions_realtime_scaling.csv")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
