# fermions —— 费米统一入口：Δτ/Δt/Δm（bath-first 两层分派，见包 README/plan）
include("quapi/quapi.jl")
include("exact/exact.jl")

# ---------------- 统一入口：第一层分派 = bath 类型（统计），
#                  第二层分派 = scheme（核技术） ----------------

"""
    Δτ(bath::AbstractFermionicNormalBath; N, δτ=bath.β/N, scheme=QuAPI())

费米浴虚时 hybridization function。`scheme::DiscretizationScheme` 选核技术：
`QuAPI()`（默认，一阶冻结核梯形装配），返回 `ImagCorrelationFunction`；
`ExactEqualDistantKernel`（虚轴严格核路由）**未实现**——`exact_fermionic_Δτ`
已于 2026-11-25 退役（恢复入口见 README todolist）。β、μ 从 bath 对象读取。
"""
Δτ(bath::AbstractFermionicNormalBath; N::Int, δτ::Real=bath.β/N,
   scheme::DiscretizationScheme=QuAPI()) = _fermions_Δτ(scheme, bath, N, δτ)

_fermions_Δτ(::QuAPI, bath::AbstractFermionicNormalBath, N::Int, δτ::Real) =
    fermionic_Δτ(bath.spectrum, β=bath.β, N=N, μ=bath.μ, δτ=δτ)
_fermions_Δτ(::ExactEqualDistantKernel, bath::AbstractFermionicNormalBath, N::Int, δτ::Real) =
    error("未实现：虚轴严格核 exact_fermionic_Δτ 已退役（2026-11-25 轴面收敛；恢复入口见 README todolist）")

"""
    Δt(bath::AbstractFermionicNormalBath; N, t, scheme=QuAPI())

费米浴实时（闭 Keldysh 轮廓）hybridization function，δt = t/N。
scheme 语义同 `Δτ`。`QuAPI()` 返回 `RealCorrelationFunction`；
`ExactEqualDistantKernel()` 返回 `ExactRealCorrelationFunction`（normdelta 经
直接调用 `exact_fermionic_Δt` 获取）。
"""
Δt(bath::AbstractFermionicNormalBath; N::Int, t::Real,
   scheme::DiscretizationScheme=QuAPI()) = _fermions_Δt(scheme, bath, N, t)

_fermions_Δt(::QuAPI, bath::AbstractFermionicNormalBath, N::Int, t::Real) =
    fermionic_Δt(bath.spectrum, β=bath.β, N=N, t=t, μ=bath.μ)
_fermions_Δt(s::ExactEqualDistantKernel, bath::AbstractFermionicNormalBath, N::Int, t::Real) =
    exact_fermionic_Δt(bath.spectrum, β=bath.β, N=N, t=t, μ=bath.μ; gfsolver=s.solver)[1]

"""
    Δm(bath::AbstractFermionicNormalBath; Nτ, t, Nt, δτ=bath.β/Nτ, scheme=QuAPI())

费米浴混合（Kadanoff-Baym L 形）轮廓 hybridization function。
`scheme::DiscretizationScheme` 选核技术：`QuAPI()`（默认），返回
`MixedCorrelationFunction`；`ExactEqualDistantKernel`（混合轴严格核路由）
**未实现**——`exact_fermionic_Δm` 已于 2026-11-25 退役（恢复入口见 README
todolist）。
"""
Δm(bath::AbstractFermionicNormalBath; Nτ::Int, t::Real, Nt::Int, δτ::Real=bath.β/Nτ,
   scheme::DiscretizationScheme=QuAPI()) = _fermions_Δm(scheme, bath, Nτ, t, Nt, δτ)

_fermions_Δm(::QuAPI, bath::AbstractFermionicNormalBath, Nτ::Int, t::Real, Nt::Int, δτ::Real) =
    fermionic_Δm(bath.spectrum, β=bath.β, μ=bath.μ, Nτ=Nτ, t=t, Nt=Nt, δτ=δτ)
_fermions_Δm(::ExactEqualDistantKernel, bath::AbstractFermionicNormalBath, Nτ::Int, t::Real, Nt::Int, δτ::Real) =
    error("未实现：混合轴严格核 exact_fermionic_Δm 已退役（2026-11-25 轴面收敛；恢复入口见 README todolist）")
