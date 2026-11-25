# bosons —— 玻色统一入口：Δτ/Δt/Δm（bath-first 两层分派，见包 README/plan）
include("quapi/quapi.jl")

# ---------------- 统一入口：第一层分派 = bath 类型（统计），
#                  第二层分派 = scheme（核技术） ----------------

"""
    Δτ(bath::AbstractBosonicNormalBath; N, δτ=bath.β/N, scheme=QuAPI())

玻色浴虚时 hybridization function。`scheme::DiscretizationScheme` 选核技术：
`QuAPI()`（默认，一阶冻结核梯形装配）；`ExactEqualDistantKernel`（严格核路由）
**未实现**——`exact_bosonic_Δτ` 已于 2026-11-25 退役（恢复入口见 README
todolist）。返回 `ImagCorrelationFunction`。β、μ 从 bath 对象读取。
"""
Δτ(bath::AbstractBosonicNormalBath; N::Int, δτ::Real=bath.β/N,
   scheme::DiscretizationScheme=QuAPI()) = _bosons_Δτ(scheme, bath, N, δτ)

_bosons_Δτ(::QuAPI, bath::AbstractBosonicNormalBath, N::Int, δτ::Real) =
    bosonic_Δτ(bath.spectrum, β=bath.β, N=N, μ=bath.μ, δτ=δτ)
_bosons_Δτ(::ExactEqualDistantKernel, bath::AbstractBosonicNormalBath, N::Int, δτ::Real) =
    error("未实现：玻色严格核 exact_bosonic_Δτ 已退役（2026-11-25 轴面收敛；恢复入口见 README todolist）")

"""
    Δt(bath::AbstractBosonicNormalBath; N, t, scheme=QuAPI())

玻色浴实时（闭 Keldysh 轮廓）hybridization function，δt = t/N。
scheme 语义同 `Δτ`（ExactEqualDistantKernel 路由未实现——`exact_bosonic_Δt`
已退役）。返回 `RealCorrelationFunction`。
"""
Δt(bath::AbstractBosonicNormalBath; N::Int, t::Real,
   scheme::DiscretizationScheme=QuAPI()) = _bosons_Δt(scheme, bath, N, t)

_bosons_Δt(::QuAPI, bath::AbstractBosonicNormalBath, N::Int, t::Real) =
    bosonic_Δt(bath.spectrum, β=bath.β, N=N, t=t, μ=bath.μ)
_bosons_Δt(::ExactEqualDistantKernel, bath::AbstractBosonicNormalBath, N::Int, t::Real) =
    error("未实现：玻色严格核 exact_bosonic_Δt 已退役（2026-11-25 轴面收敛；恢复入口见 README todolist）")

"""
    Δm(bath::AbstractBosonicNormalBath; Nτ, t, Nt, δτ=bath.β/Nτ, scheme=QuAPI())

玻色浴混合（Kadanoff-Baym L 形）轮廓 hybridization function。
scheme 语义同 `Δτ`（ExactEqualDistantKernel 路由未实现——`exact_bosonic_Δm`
已退役）。返回 `MixedCorrelationFunction`。
"""
Δm(bath::AbstractBosonicNormalBath; Nτ::Int, t::Real, Nt::Int, δτ::Real=bath.β/Nτ,
   scheme::DiscretizationScheme=QuAPI()) = _bosons_Δm(scheme, bath, Nτ, t, Nt, δτ)

_bosons_Δm(::QuAPI, bath::AbstractBosonicNormalBath, Nτ::Int, t::Real, Nt::Int, δτ::Real) =
    bosonic_Δm(bath.spectrum, β=bath.β, μ=bath.μ, Nτ=Nτ, t=t, Nt=Nt, δτ=δτ)
_bosons_Δm(::ExactEqualDistantKernel, bath::AbstractBosonicNormalBath, Nτ::Int, t::Real, Nt::Int, δτ::Real) =
    error("未实现：玻色严格核 exact_bosonic_Δm 已退役（2026-11-25 轴面收敛；恢复入口见 README todolist）")
