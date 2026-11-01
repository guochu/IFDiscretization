# fermions —— 费米统一入口：Δτ/Δt/Δm（bath-first 两层分派，见包 README/plan）
include("quapi/quapi.jl")
include("exact/exact.jl")

# ---------------- 统一入口：第一层分派 = bath 类型（统计），
#                  第二层分派 = scheme（核技术） ----------------

"""
    Δτ(bath::AbstractFermionicNormalBath; N, δτ=bath.β/N, scheme=QuAPI())

费米浴虚时 hybridization function。`scheme::DiscretizationScheme` 选核技术：
`QuAPI()`（默认，一阶冻结核梯形装配）或 `ExactIF(gfsolver)`（严格核，
本征求解器 + GL-M 求积）。返回 `ImagCorrelationFunction`。β、μ 从 bath
对象读取。

注：`exact_fermionic_*`（ExactIF 路线）接替 QuAPI 已删除的
ExactGrassmannIF 路线——返回标准 `ImagCorrelationFunction`
（旧版返回配对矩阵 + prefactor）。
"""
Δτ(bath::AbstractFermionicNormalBath; N::Int, δτ::Real=bath.β/N,
   scheme::DiscretizationScheme=QuAPI()) = _fermions_Δτ(scheme, bath, N, δτ)

_fermions_Δτ(::QuAPI, bath::AbstractFermionicNormalBath, N::Int, δτ::Real) =
    fermionic_Δτ(bath.spectrum, β=bath.β, N=N, μ=bath.μ, δτ=δτ)
_fermions_Δτ(s::ExactIF, bath::AbstractFermionicNormalBath, N::Int, δτ::Real) =
    exact_fermionic_Δτ(bath.spectrum, β=bath.β, N=N, μ=bath.μ, δτ=δτ; gfsolver=s.gfsolver)

"""
    Δt(bath::AbstractFermionicNormalBath; N, t, scheme=QuAPI())

费米浴实时（闭 Keldysh 轮廓）hybridization function，δt = t/N。
scheme 语义同 `Δτ`。返回 `RealCorrelationFunction`。
"""
Δt(bath::AbstractFermionicNormalBath; N::Int, t::Real,
   scheme::DiscretizationScheme=QuAPI()) = _fermions_Δt(scheme, bath, N, t)

_fermions_Δt(::QuAPI, bath::AbstractFermionicNormalBath, N::Int, t::Real) =
    fermionic_Δt(bath.spectrum, β=bath.β, N=N, t=t, μ=bath.μ)
_fermions_Δt(s::ExactIF, bath::AbstractFermionicNormalBath, N::Int, t::Real) =
    exact_fermionic_Δt(bath.spectrum, β=bath.β, N=N, t=t, μ=bath.μ; gfsolver=s.gfsolver)[1]

"""
    Δm(bath::AbstractFermionicNormalBath; Nτ, t, Nt, δτ=bath.β/Nτ, scheme=QuAPI())

费米浴混合（Kadanoff-Baym L 形）轮廓 hybridization function。
scheme 语义同 `Δτ`。返回 `MixedCorrelationFunction`。
"""
Δm(bath::AbstractFermionicNormalBath; Nτ::Int, t::Real, Nt::Int, δτ::Real=bath.β/Nτ,
   scheme::DiscretizationScheme=QuAPI()) = _fermions_Δm(scheme, bath, Nτ, t, Nt, δτ)

_fermions_Δm(::QuAPI, bath::AbstractFermionicNormalBath, Nτ::Int, t::Real, Nt::Int, δτ::Real) =
    fermionic_Δm(bath.spectrum, β=bath.β, μ=bath.μ, Nτ=Nτ, t=t, Nt=Nt, δτ=δτ)
_fermions_Δm(s::ExactIF, bath::AbstractFermionicNormalBath, Nτ::Int, t::Real, Nt::Int, δτ::Real) =
    exact_fermionic_Δm(bath.spectrum, β=bath.β, μ=bath.μ, Nτ=Nτ, t=t, Nt=Nt, δτ=δτ; gfsolver=s.gfsolver)
