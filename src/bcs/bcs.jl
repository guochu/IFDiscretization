# bcs —— BCS 统一入口：Δτ/Δt/Δm（bath-first 两层分派，见包 README/plan）
include("quapi/quapi.jl")
include("exact/exact.jl")

# ---------------- 统一入口：第一层分派 = bath 类型（统计），
#                  第二层分派 = scheme（核技术） ----------------

"""
    Δτ(bath::AbstractBCSBath; N, δτ=bath.β/N, scheme=QuAPI())

BCS（超导）浴虚时 hybridization function。`scheme::DiscretizationScheme`
选核技术：`QuAPI()`（默认，Bogoliubov 系数 + fermionic_Δτ 冻结核）、
`ExactIF(gfsolver)`（严格核，Nambu/BdG 本征闭式；只支持 μ=0）。
返回 `BCSCorrelationFunction`。β、μ、Δ 从 bath 对象读取。
"""
Δτ(bath::AbstractBCSBath; N::Int, δτ::Real=bath.β/N,
   scheme::DiscretizationScheme=QuAPI()) = _bcs_Δτ_dispatch(scheme, bath, N, δτ)

_bcs_Δτ_dispatch(::QuAPI, bath::AbstractBCSBath, N::Int, δτ::Real) =
    bcs_Δτ(bath.spectrum, β=bath.β, N=N, μ=bath.μ, Δ=bath.Δ, δτ=δτ)
# BCS 严格核内定 δτ = β/N（原 exact API 无 δτ 旋钮，忠实保留）
_bcs_Δτ_dispatch(s::ExactIF, bath::AbstractBCSBath, N::Int, δτ::Real) =
    exact_bcs_Δτ(bath.spectrum, β=bath.β, N=N, μ=bath.μ, Δ=bath.Δ; gfsolver=s.gfsolver)

"""
    Δt(bath::AbstractBCSBath; N, t, scheme=QuAPI())

BCS 浴实时（闭 Keldysh 轮廓）hybridization function，δt = t/N。
scheme 语义同 `Δτ`。返回 `BCSCorrelationFunction`。
"""
Δt(bath::AbstractBCSBath; N::Int, t::Real,
   scheme::DiscretizationScheme=QuAPI()) = _bcs_Δt_dispatch(scheme, bath, N, t)

_bcs_Δt_dispatch(::QuAPI, bath::AbstractBCSBath, N::Int, t::Real) =
    bcs_Δt(bath.spectrum, β=bath.β, μ=bath.μ, Δ=bath.Δ, N=N, t=t)
_bcs_Δt_dispatch(s::ExactIF, bath::AbstractBCSBath, N::Int, t::Real) =
    exact_bcs_Δt(bath.spectrum, β=bath.β, N=N, t=t, μ=bath.μ, Δ=bath.Δ; gfsolver=s.gfsolver)

"""
    Δm(bath::AbstractBCSBath; Nτ, t, Nt, δτ=bath.β/Nτ, scheme=QuAPI())

BCS 浴混合（Kadanoff-Baym L 形）轮廓 hybridization function。
scheme 语义同 `Δτ`。返回 `BCSCorrelationFunction`。
"""
Δm(bath::AbstractBCSBath; Nτ::Int, t::Real, Nt::Int, δτ::Real=bath.β/Nτ,
   scheme::DiscretizationScheme=QuAPI()) = _bcs_Δm_dispatch(scheme, bath, Nτ, t, Nt, δτ)

_bcs_Δm_dispatch(::QuAPI, bath::AbstractBCSBath, Nτ::Int, t::Real, Nt::Int, δτ::Real) =
    bcs_Δm(bath.spectrum, β=bath.β, μ=bath.μ, Δ=bath.Δ, Nτ=Nτ, t=t, Nt=Nt, δτ=δτ)
_bcs_Δm_dispatch(s::ExactIF, bath::AbstractBCSBath, Nτ::Int, t::Real, Nt::Int, δτ::Real) =
    exact_bcs_Δm(bath.spectrum, β=bath.β, Nτ=Nτ, t=t, Nt=Nt, μ=bath.μ, Δ=bath.Δ, δτ=δτ; gfsolver=s.gfsolver)
