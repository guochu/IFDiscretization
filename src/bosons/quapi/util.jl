# 玻色冻结核共享内核（← QuAPI/src/{imagtime,realtime,mixedtime}/bosonic.jl 尾部 helpers
# 与 src/imagtime/util.jl 的 safe_mult 原样移植；_*_disperse 是装配唯一入口）

function safe_mult(β::Real, ϵ::Real)
    if (β == Inf) && (ϵ == 0)
        return zero(ϵ)
    end
    return β * ϵ
end

# ------------------------------ 虚时（i 系列） ------------------------------ #

function _bosonic_fⱼₖ_i_disperse(f::AbstractBoundedFunction, disperse, β, μ, Δk::Int, δτ)
    g = _bosonic_gⱼₖ_i(β, μ, Δk, δτ)
    g′ = ϵ -> g(disperse(ϵ))
    return f * g′
end
_bosonic_fⱼₖ_i(f::AbstractBoundedFunction, β, μ, Δk::Int, δτ) = f * _bosonic_gⱼₖ_i(β, μ, Δk, δτ)
function _bosonic_gⱼₖ_i(β, μ, Δk::Int, δτ)
    return ε -> begin
        # x = 1-exp(-safe_mult(β, ε-μ))
        if abs(ε) > get_quapi_tol()
            return -2*(1 + boseeinstein(β, μ, ε))*exp(-Δk*δτ*ε)*(1-cosh(δτ*ε))/ε^2
        else
            if μ == zero(μ)
                return exp(-Δk*δτ*ε)*δτ^2 / safe_mult(β, ε)
            else
                return (1 + boseeinstein(β, μ, ε)) * exp(-Δk*δτ*ε)*δτ^2
            end
        end
    end
end

function _bosonic_fⱼⱼ_i_disperse(f::AbstractBoundedFunction, disperse, β, μ, δτ)
    g = _bosonic_gⱼⱼ_i(β, μ, δτ)
    g′ = ϵ -> g(disperse(ϵ))
    return f * g′
end
_bosonic_fⱼⱼ_i(f::AbstractBoundedFunction, β, μ, δτ) = f * _bosonic_gⱼⱼ_i(β, μ, δτ)
function _bosonic_gⱼⱼ_i(β, μ, δτ)
    return ε -> begin
        x = 1-exp(-safe_mult(β, ε-μ))
        if abs(ε) > get_quapi_tol()
            return (1 + boseeinstein(β, μ, ε)) * (exp(-δτ*ε)-(1-δτ*ε))/ε^2
        else
            if μ == zero(μ)
                return 0.5*δτ^2 / safe_mult(β, ε)
            else
                return (1 + boseeinstein(β, μ, ε)) * 0.5*δτ^2
            end
        end
    end
end

function _bosonic_fₖⱼ_i_disperse(f::AbstractBoundedFunction, disperse, β, μ, Δk::Int, δτ)
    g = _bosonic_gₖⱼ_i(β, μ, Δk, δτ)
    g′ = ϵ -> g(disperse(ϵ))
    return f * g′
end
_bosonic_fₖⱼ_i(f::AbstractBoundedFunction, β, μ, Δk::Int, δτ) = f * _bosonic_gₖⱼ_i(β, μ, Δk, δτ)
function _bosonic_gₖⱼ_i(β, μ, Δk::Int, δτ)
    return ε -> begin
        if abs(ε) > get_quapi_tol()
            # -expm1(-x) avoids the catastrophic cancellation of 1-exp(-x) for small x
            x = -expm1(-safe_mult(β, ε-μ))
            y = exp(-safe_mult(β - Δk*δτ, ε-μ)) * exp(Δk*δτ*μ)
            return -2y*(1-cosh(δτ*ε))/(ε^2 * x)
        else
            if μ == zero(μ)
                return exp(Δk*δτ*ε)*δτ^2 / safe_mult(β, ε)
            else
                x = -expm1(-safe_mult(β, ε-μ))
                y = exp(-safe_mult(β - Δk*δτ, ε-μ)) * exp(Δk*δτ*μ)
                return y*δτ^2 / x
            end
        end
    end
end

function _bosonic_fₖₖ_i_disperse(f::AbstractBoundedFunction, disperse, β, μ, δτ)
    g = _bosonic_gₖₖ_i(β, μ, δτ)
    g′ = ϵ -> g(disperse(ϵ))
    return f * g′
end
_bosonic_fₖₖ_i(f::AbstractBoundedFunction, β, μ, δτ) = f * _bosonic_gₖₖ_i(β, μ, δτ)
function _bosonic_gₖₖ_i(β, μ, δτ)
    return ε -> begin
        # x = exp(-safe_mult(β, ε-μ))
        if abs(ε) > get_quapi_tol()
            return boseeinstein(β, μ, ε)*(exp(δτ*ε)-(1+δτ*ε))  / ε^2
        else
            if μ == zero(μ)
                return 0.5*δτ^2 / safe_mult(β, ε)
            else
                return boseeinstein(β, μ, ε)*0.5*δτ^2 / 2
            end
        end
    end
end

# ------------------------------ 实时（r 系列） ------------------------------ #

function _bosonic_fⱼₖ_r_disperse(f::AbstractBoundedFunction, disperse, β, μ, Δk::Int, δt)
    g = _bosonic_gⱼₖ_r(β, μ, Δk, δt)
    g′ = ϵ -> g(disperse(ϵ))
    return f * g′
end
_bosonic_fⱼₖ_r(f::AbstractBoundedFunction, β, μ, Δk::Int, δt) = f * _bosonic_gⱼₖ_r(β, μ, Δk, δt)
function _bosonic_gⱼₖ_r(β, μ, Δk::Int, δt)
    return ε -> begin
        if abs(ε) > get_quapi_tol()
            return 2*(1 + boseeinstein(β, μ, ε))*exp(-im*ε*Δk*δt)*(1-cos(ε*δt))/ε^2
        else
            if μ == zero(μ)
                return exp(-im*ε*Δk*δt)*δt^2/(β*ε)
            else
                return (1 + boseeinstein(β, μ, ε))*exp(-im*ε*Δk*δt)*δt^2
            end
        end
    end
end

function _bosonic_fₖⱼ_r_disperse(f::AbstractBoundedFunction, disperse, β, μ, Δk::Int, δt)
    g = _bosonic_gₖⱼ_r(β, μ, Δk, δt)
    g′ = ϵ -> g(disperse(ϵ))
    return f * g′
end
_bosonic_fₖⱼ_r(f::AbstractBoundedFunction, β, μ, Δk::Int, δt) = f * _bosonic_gₖⱼ_r(β, μ, Δk, δt)
function _bosonic_gₖⱼ_r(β, μ, Δk::Int, δt)
    return ε -> begin
        if abs(ε) > get_quapi_tol()
            return 2*boseeinstein(β, μ, ε)*exp(-im*ε*Δk*δt)*(1-cos(ε*δt))/ε^2
        else
            if μ == zero(μ)
                return exp(-im*ε*Δk*δt)*δt^2/(β*ε)
            else
                return boseeinstein(β, μ, ε) * exp(-im*ε*Δk*δt)*δt^2
            end
        end
    end
end

function _bosonic_fⱼⱼ_r_disperse(f::AbstractBoundedFunction, disperse, β, μ, δt)
    g = _bosonic_gⱼⱼ_r(β, μ, δt)
    g′ = ϵ -> g(disperse(ϵ))
    return f * g′
end
_bosonic_fⱼⱼ_r(f::AbstractBoundedFunction, β, μ, δt) = f * _bosonic_gⱼⱼ_r(β, μ, δt)
function _bosonic_gⱼⱼ_r(β, μ, δt)
    return ε -> begin
        if abs(ε) > get_quapi_tol()
            return (1 + boseeinstein(β, μ, ε))*((1-im*ε*δt)-exp(-im*ε*δt))/ε^2
        else
            if μ == zero(μ)
                return 0.5*δt^2/(β*ε)
            else
                return (1 + boseeinstein(β, μ, ε))*0.5*δt^2
            end
        end
    end
end

function _bosonic_fₖₖ_r_disperse(f::AbstractBoundedFunction, disperse, β, μ, δt)
    g = _bosonic_gₖₖ_r(β, μ, δt)
    g′ = ϵ -> g(disperse(ϵ))
    return f * g′
end
_bosonic_fₖₖ_r(f::AbstractBoundedFunction, β, μ, δt) = f * _bosonic_gₖₖ_r(β, μ, δt)
function _bosonic_gₖₖ_r(β, μ, δt)
    return ε -> begin
        if abs(ε) > get_quapi_tol()
            return boseeinstein(β, μ, ε)*((1+im*ε*δt)-exp(im*ε*δt))/ε^2
        else
            if μ == zero(μ)
                return 0.5*δt^2/(β*ε)
            else
                return boseeinstein(β, μ, ε)*0.5*δt^2
            end
        end
    end
end

# ------------------------------ 混合时间（L 轮廓 ζ 块） ------------------------------ #

function _bosonic_lⱼₖ_disperse(f::AbstractBoundedFunction, disperse, β, μ, j::Int, k::Int, δt, δτ)
    g = _bosonic_gⱼₖ(β, μ, j, k, δt, δτ)
    g′ = ϵ -> g(disperse(ϵ))
    return f * g′
end
_bosonic_lⱼₖ(f::AbstractBoundedFunction, β, μ, j::Int, k::Int, δt, δτ) = f * _bosonic_gⱼₖ(β, μ, j, k, δt, δτ)
function _bosonic_gⱼₖ(β, μ, j::Int, k::Int, δt, δτ)
    return ε -> begin
        # x = 1-exp(-safe_mult(β, ε-μ))
        if abs(ε) > get_quapi_tol()
            return -im*(1 + boseeinstein(β, μ, ε))*exp(-ε*j*δτ)*exp(im*ε*k*δt)*(exp(-ε*δτ)-1)*(exp(im*ε*δt)-1)/ε^2
        else
            if μ == zero(μ)
                return 2*exp(-ε*j*δτ)*exp(im*ε*k*δt)*δτ*δt / safe_mult(β, ε)
            else
                return -(1 + boseeinstein(β, μ, ε))*exp(-ε*j*δτ)*exp(im*ε*k*δt)*δτ*δt
            end
        end
    end
end

function _bosonic_lₖⱼ_disperse(f::AbstractBoundedFunction, disperse, β, μ, k::Int, j::Int, δt, δτ)
    g = _bosonic_gₖⱼ(β, μ, k, j, δt, δτ)
    g′ = ϵ -> g(disperse(ϵ))
    return f * g′
end
_bosonic_lₖⱼ(f::AbstractBoundedFunction, β, μ, k::Int, j::Int, δt, δτ) = f * _bosonic_gₖⱼ(β, μ, k, j, δt, δτ)
function _bosonic_gₖⱼ(β, μ, k::Int, j::Int, δt, δτ)
    return ε -> begin
        # x = 1 - exp(-safe_mult(β, ε-μ))
        # y = exp(-safe_mult(β - j*δτ, ε-μ)) * exp(j*δτ*μ)
        if abs(ε) > get_quapi_tol()
            x = 1 - exp(-safe_mult(β, ε-μ))
            y = exp(-safe_mult(β - j*δτ, ε-μ)) * exp(j*δτ*μ)
            return -im*y*exp(-im*ε*k*δt)*(exp(ε*δτ)-1)*(exp(-im*ε*δt)-1)/(ε^2 * x)
        else
            if μ == zero(μ)
                return 2*exp(ε*j*δτ)*exp(-im*ε*k*δt)*δτ*δt / safe_mult(β, ε)
            else
                x = 1 - exp(-safe_mult(β, ε-μ))
                y = exp(-safe_mult(β - j*δτ, ε-μ)) * exp(j*δτ*μ)
                return -y*exp(-im*ε*k*δt)*δτ*δt / x
            end
        end
    end
end
