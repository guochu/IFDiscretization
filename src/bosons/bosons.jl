# bosons —— 玻色统一入口：Δτ/Δt/Δm（bath-first 两层分派，见包 README/plan）
include("quapi/quapi.jl")
include("exact/exact.jl")

# ---------------- 统一入口：第一层分派 = bath 类型（统计），
#                  第二层分派 = scheme（核技术） ----------------

"""
    Δτ(bath::AbstractBosonicNormalBath; N, δτ=bath.β/N, scheme=QuAPI())

玻色浴虚时 hybridization function。`scheme::DiscretizationScheme` 选核技术：
`QuAPI()`（默认，一阶冻结核梯形装配）或 `ExactIF(gfsolver)`（严格核，
每模闭式 + GL-M 求积，`EigenSolver(nbath)` 给节点数）。返回
`ImagCorrelationFunction`。β、μ 从 bath 对象读取。
"""
Δτ(bath::AbstractBosonicNormalBath; N::Int, δτ::Real=bath.β/N,
   scheme::DiscretizationScheme=QuAPI()) = _bosons_Δτ(scheme, bath, N, δτ)

_bosons_Δτ(::QuAPI, bath::AbstractBosonicNormalBath, N::Int, δτ::Real) =
    bosonic_Δτ(bath.spectrum, β=bath.β, N=N, μ=bath.μ, δτ=δτ)
_bosons_Δτ(s::ExactIF, bath::AbstractBosonicNormalBath, N::Int, δτ::Real) =
    exact_bosonic_Δτ(bath.spectrum, β=bath.β, N=N, μ=bath.μ, δτ=δτ; gfsolver=s.gfsolver)

"""
    Δt(bath::AbstractBosonicNormalBath; N, t, scheme=QuAPI())

玻色浴实时（闭 Keldysh 轮廓）hybridization function，δt = t/N。
scheme 语义同 `Δτ`。返回 `RealCorrelationFunction`。
"""
Δt(bath::AbstractBosonicNormalBath; N::Int, t::Real,
   scheme::DiscretizationScheme=QuAPI()) = _bosons_Δt(scheme, bath, N, t)

_bosons_Δt(::QuAPI, bath::AbstractBosonicNormalBath, N::Int, t::Real) =
    bosonic_Δt(bath.spectrum, β=bath.β, N=N, t=t, μ=bath.μ)
_bosons_Δt(s::ExactIF, bath::AbstractBosonicNormalBath, N::Int, t::Real) =
    exact_bosonic_Δt(bath.spectrum, β=bath.β, N=N, t=t, μ=bath.μ; gfsolver=s.gfsolver)

"""
    Δm(bath::AbstractBosonicNormalBath; Nτ, t, Nt, δτ=bath.β/Nτ, scheme=QuAPI())

玻色浴混合（Kadanoff-Baym L 形）轮廓 hybridization function。
scheme 语义同 `Δτ`。返回 `MixedCorrelationFunction`。
"""
Δm(bath::AbstractBosonicNormalBath; Nτ::Int, t::Real, Nt::Int, δτ::Real=bath.β/Nτ,
   scheme::DiscretizationScheme=QuAPI()) = _bosons_Δm(scheme, bath, Nτ, t, Nt, δτ)

_bosons_Δm(::QuAPI, bath::AbstractBosonicNormalBath, Nτ::Int, t::Real, Nt::Int, δτ::Real) =
    bosonic_Δm(bath.spectrum, β=bath.β, μ=bath.μ, Nτ=Nτ, t=t, Nt=Nt, δτ=δτ)
_bosons_Δm(s::ExactIF, bath::AbstractBosonicNormalBath, Nτ::Int, t::Real, Nt::Int, δτ::Real) =
    exact_bosonic_Δm(bath.spectrum, β=bath.β, μ=bath.μ, Nτ=Nτ, t=t, Nt=Nt, δτ=δτ; gfsolver=s.gfsolver)
