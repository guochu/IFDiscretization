# EDToulouseSolver 的实现：GL-M 谱离散（浴 h₀ 按需内联构造）
# ← QuAPI/exactif/test/exactif_api.jl 原样移植（kernel 未动）

"GL-M 求积节点上的谱数据（求积节点，非物理模式）。μ = 谱平移。"
function _quad_nodes(f::AbstractBoundedFunction, μ::Real, M::Int)
    lb, ub = lowerbound(f), upperbound(f)
    xs, ws = gauss_legendre(M)
    half = (ub - lb) / 2
    Es = [(lb + ub) / 2 + half * x - μ for x in xs]
    wΓ = [half * w * f((lb + ub) / 2 + half * x) for (x, w) in zip(xs, ws)]
    Vsq = sqrt.(max.(wΓ, 0.0))
    return Es, Vsq, wΓ
end

# 离散谱 = 物理模式直取（严格核的浴侧是二次型——离散浴下核天然精确，无求积
# 误差；模式数 = 谱线数，gfsolver 的 nbath 不消费，EDToulouseSolver docstring
# 「离散谱 = 谱线数目」的实装。此前缺失：DiracDelta 走 GL 路径在 f(ϵ) 求值处
# 抛「can not call to a delta function」、DiscreteSpectrum 的 ±Inf 撑界静默产
# 出 Inf/NaN——两者都不该被静默拒收/污染，2026-11-21 补齐）。
# 死模过滤契约沿用：wΓ ≤ 0 的线剔除（α = 0 线 = 死模，K ≥ 1 守卫不变）。
_quad_nodes(f::DiracDelta, μ::Real, ::Int) =
    ([f.ω - μ], [sqrt(max(f.α, 0.0))], [Float64(f.α)])

_quad_nodes(f::DiscreteSpectrum, μ::Real, ::Int) =
    (collect(Float64.(frequencies(f))) .- μ,
     sqrt.(max.(spectrumvalues(f), 0.0)),
     collect(Float64.(spectrumvalues(f))))
