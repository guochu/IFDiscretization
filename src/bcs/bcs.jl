# bcs —— BCS 统一入口：Δτ/Δt/Δm（bath-first 两层分派，见包 README/plan）
include("quapi/quapi.jl")
include("exact/exact.jl")       # 实轴严格核 exact_bcs_Δt（2026-11-25 退役后单轴恢复）

# ---------------- 统一入口：第一层分派 = bath 类型（统计），
#                  第二层分派 = scheme（核技术） ----------------

"""
    Δτ(bath::AbstractBCSBath; N, δτ=bath.β/N, scheme=QuAPI())

BCS（超导）浴虚时 hybridization function。`scheme::DiscretizationScheme`
选核技术：`QuAPI()`（默认，Bogoliubov 系数 + fermionic_Δτ 冻结核）；
`ExactEqualDistantKernel`（严格核路由）**未实现**——`exact_bcs_Δτ` 已于
2026-11-25 退役（恢复入口见 README todolist）。返回 `BCSCorrelationFunction`。
β、μ、Δ 从 bath 对象读取。
"""
Δτ(bath::AbstractBCSBath; N::Int, δτ::Real=bath.β/N,
   scheme::DiscretizationScheme=QuAPI()) = _bcs_Δτ_dispatch(scheme, bath, N, δτ)

_bcs_Δτ_dispatch(::QuAPI, bath::AbstractBCSBath, N::Int, δτ::Real) =
    bcs_Δτ(bath.spectrum, β=bath.β, N=N, μ=bath.μ, Δ=bath.Δ, δτ=δτ)
_bcs_Δτ_dispatch(::ExactEqualDistantKernel, bath::AbstractBCSBath, N::Int, δτ::Real) =
    error("未实现：BCS 严格核 exact_bcs_Δτ 已退役（2026-11-25 轴面收敛；恢复入口见 README todolist）")

"""
    Δt(bath::AbstractBCSBath; N, t, scheme=QuAPI())

BCS 浴实时（闭 Keldysh 轮廓）hybridization function，δt = t/N。
`QuAPI()`（默认，Bogoliubov 系数 + fermionic_Δt 冻结核）返回
`BCSCorrelationFunction`；`ExactEqualDistantKernel` 走严格核
`exact_bcs_Δt`（2026-11-25 退役后单轴恢复，FormB Nambu 翻倍），
返回 `BCSCorrelationFunction{ExactRealCorrelationFunction}`
（normdelta 经直接调用 `exact_bcs_Δt` 获取）。
"""
Δt(bath::AbstractBCSBath; N::Int, t::Real,
   scheme::DiscretizationScheme=QuAPI()) = _bcs_Δt_dispatch(scheme, bath, N, t)

_bcs_Δt_dispatch(::QuAPI, bath::AbstractBCSBath, N::Int, t::Real) =
    bcs_Δt(bath.spectrum, β=bath.β, μ=bath.μ, Δ=bath.Δ, N=N, t=t)
_bcs_Δt_dispatch(s::ExactEqualDistantKernel, bath::AbstractBCSBath, N::Int, t::Real) =
    exact_bcs_Δt(bath.spectrum, β=bath.β, μ=bath.μ, Δ=bath.Δ, N=N, t=t;
                 gfsolver=s.solver)[1]

"""
    Δm(bath::AbstractBCSBath; Nτ, t, Nt, δτ=bath.β/Nτ, scheme=QuAPI())

BCS 浴混合（Kadanoff-Baym L 形）轮廓 hybridization function。
scheme 语义同 `Δτ`（ExactEqualDistantKernel 路由未实现——`exact_bcs_Δm`
已退役）。返回 `BCSCorrelationFunction`。
"""
Δm(bath::AbstractBCSBath; Nτ::Int, t::Real, Nt::Int, δτ::Real=bath.β/Nτ,
   scheme::DiscretizationScheme=QuAPI()) = _bcs_Δm_dispatch(scheme, bath, Nτ, t, Nt, δτ)

_bcs_Δm_dispatch(::QuAPI, bath::AbstractBCSBath, Nτ::Int, t::Real, Nt::Int, δτ::Real) =
    bcs_Δm(bath.spectrum, β=bath.β, μ=bath.μ, Δ=bath.Δ, Nτ=Nτ, t=t, Nt=Nt, δτ=δτ)
_bcs_Δm_dispatch(::ExactEqualDistantKernel, bath::AbstractBCSBath, Nτ::Int, t::Real, Nt::Int, δτ::Real) =
    error("未实现：BCS 严格核 exact_bcs_Δm 已退役（2026-11-25 轴面收敛；恢复入口见 README todolist）")
