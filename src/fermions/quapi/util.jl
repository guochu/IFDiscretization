# 费米冻结核共享内核（← QuAPI/src/{imagtime,realtime,mixedtime}/fermionic.jl 尾部 helpers
# 与 src/realtime/util.jl 的 `_gⱼₖ_r` 系列原样移植；safe_mult 归 bosons/quapi/util.jl）

# ------------------------------ 虚时（i 系列） ------------------------------ #

function _fermionic_fⱼₖ_i_disperse(f::AbstractBoundedFunction, disperse, β, μ, Δk::Int, δτ)
    g = _fermionic_gⱼₖ_i(β, μ, Δk, δτ)
    g′ = ϵ -> g(disperse(ϵ))
    return f * g′
end
_fermionic_fⱼₖ_i(f::AbstractBoundedFunction, β, μ, Δk::Int, δτ) = f * _fermionic_gⱼₖ_i(β, μ, Δk, δτ)
function _fermionic_gⱼₖ_i(β, μ, Δk::Int, δτ)
    return ε -> begin
        if ε >= μ
            x = exp(-safe_mult(β, ε-μ))+1
            if abs(ε) > get_quapi_tol()
                -2exp(-Δk*δτ*ε)*(1-cosh(δτ*ε))/(ε^2 * x)
            else
                exp(-Δk*δτ*ε)*δτ^2 / x
            end
        else
            x = exp(safe_mult(β, ε-μ)) + 1
            y = exp(safe_mult(β-Δk*δτ, ε-μ)) * exp(-Δk*δτ*μ)
            if abs(ε) > get_quapi_tol()
                -2y*(1-cosh(δτ*ε))/(ε^2 * x)
            else
                y*δτ^2 / x
            end
        end
    end
end

function _fermionic_fⱼⱼ_i_disperse(f::AbstractBoundedFunction, disperse, β, μ, δτ)
    g = _fermionic_gⱼⱼ_i(β, μ, δτ)
    g′ = ϵ -> g(disperse(ϵ))
    return f * g′
end
_fermionic_fⱼⱼ_i(f::AbstractBoundedFunction, β, μ, δτ) = f * _fermionic_gⱼⱼ_i(β, μ, δτ)
function _fermionic_gⱼⱼ_i(β, μ, δτ)
    return ε -> begin
        if ε >= μ
            x = exp(-safe_mult(β, ε-μ))+1
            if abs(ε) > get_quapi_tol()
                (exp(-δτ*ε)-(1-δτ*ε))/(ε^2 * x)
            else
                0.5*δτ^2 / x
            end
        else
            x = exp(safe_mult(β, ε-μ))
            if abs(ε) > get_quapi_tol()
                x*(exp(-δτ*ε)-(1-δτ*ε))/(ε^2 * (x+1))
            else
                0.5*δτ^2*x / (x+1)
            end
        end
    end
end

function _fermionic_fₖⱼ_i_disperse(f::AbstractBoundedFunction, disperse, β, μ, Δk::Int, δτ)
    g = _fermionic_gₖⱼ_i(β, μ, Δk, δτ)
    g′ = ϵ -> g(disperse(ϵ))
    return f * g′
end
_fermionic_fₖⱼ_i(f::AbstractBoundedFunction, β, μ, Δk::Int, δτ) = f * _fermionic_gₖⱼ_i(β, μ, Δk, δτ)
function _fermionic_gₖⱼ_i(β, μ, Δk::Int, δτ)
    return ε -> begin
        if ε >= μ
            x = exp(-safe_mult(β, ε-μ))+1
            y = exp(-safe_mult(β - Δk*δτ, ε-μ)) * exp(Δk*δτ*μ)
            if abs(ε) > get_quapi_tol()
                2y*(1-cosh(δτ*ε))/(ε^2 * x)
            else
                -y*δτ^2 / x
            end
        else
            x = exp(safe_mult(β, ε-μ))+1
            y = exp(Δk*δτ * ε)
            if abs(ε) > get_quapi_tol()
                2y*(1-cosh(δτ*ε))/(ε^2 * x)
            else
                -y*δτ^2 / x
            end

        end
    end
end

function _fermionic_fₖₖ_i_disperse(f::AbstractBoundedFunction, disperse, β, μ, δτ)
    g = _fermionic_gₖₖ_i(β, μ, δτ)
    g′ = ϵ -> g(disperse(ϵ))
    return f * g′
end
_fermionic_fₖₖ_i(f::AbstractBoundedFunction, β, μ, δτ) = f * _fermionic_gₖₖ_i(β, μ, δτ)
function _fermionic_gₖₖ_i(β, μ, δτ)
    return ε -> begin
        if ε >= μ
            x = exp(-safe_mult(β, ε-μ))
            if abs(ε) > get_quapi_tol()
                -(exp(δτ*ε)-(1+δτ*ε)) * x / (ε^2 * (x+1))
            else
                -0.5*δτ^2 * x / (x+1)
            end
        else
            x = exp(safe_mult(β, ε-μ))
            if abs(ε) > get_quapi_tol()
                -(exp(δτ*ε)-(1+δτ*ε)) / (ε^2 * (x+1))
            else
                -0.5*δτ^2 / (x+1)
            end
        end
    end
end

# ------------------------------ 实时（r 系列；实时/混合共用） ------------------------------ #

function _fⱼₖ_r_disperse(f::AbstractBoundedFunction, disperse, Δk::Int, δt)
    g = _gⱼₖ_r(Δk, δt)
    g′ = ϵ -> g(disperse(ϵ))
    return f * g′
end
_fⱼₖ_r(f::AbstractBoundedFunction, Δk::Int, δt) = f * _gⱼₖ_r(Δk, δt)
function _gⱼₖ_r(Δk::Int, δt)
    return ε -> begin
        if abs(ε) > get_quapi_tol()
            2exp(-im*ε*Δk*δt)*(1-cos(ε*δt))/ε^2
        else
            exp(-im*ε*Δk*δt)*δt^2
        end
    end
end

function _fⱼⱼ_r_disperse(f::AbstractBoundedFunction, disperse, δt)
    g = _gⱼⱼ_r(δt)
    g′ = ϵ -> g(disperse(ϵ))
    return f * g′
end
_fⱼⱼ_r(f::AbstractBoundedFunction, δt) = f * _gⱼⱼ_r(δt)
function _gⱼⱼ_r(δt)
    return ε -> begin
        if abs(ε) > get_quapi_tol()
            ((1-im*ε*δt)-exp(-im*ε*δt))/ε^2
        else
            0.5*δt^2
        end
    end
end

function _fₖₖ_r_disperse(f::AbstractBoundedFunction, disperse, δt)
    g = _gₖₖ_r(δt)
    g′ = ϵ -> g(disperse(ϵ))
    return f * g′
end
_fₖₖ_r(f::AbstractBoundedFunction, δt) = f * _gₖₖ_r(δt)
function _gₖₖ_r(δt)
    return ε -> begin
        if abs(ε) > get_quapi_tol()
            ((1+im*ε*δt)-exp(im*ε*δt))/ε^2
        else
            0.5*δt^2
        end
    end
end

# ------------------------------ 混合时间（L 轮廓 ζ 块） ------------------------------ #

function _fermionic_lⱼₖ_disperse(f::AbstractBoundedFunction, disperse, β, μ, j::Int, k::Int, δt, δτ)
    g = _fermionic_gⱼₖ(β, μ, j, k, δt, δτ)
    g′ = ϵ -> g(disperse(ϵ))
    return f * g′
end
_fermionic_lⱼₖ(f::AbstractBoundedFunction, β, μ, j::Int, k::Int, δt, δτ) = f * _fermionic_gⱼₖ(β, μ, j, k, δt, δτ)
function _fermionic_gⱼₖ(β, μ, j::Int, k::Int, δt, δτ)
    return ε -> begin
        if β >= μ
            x = exp(-safe_mult(β, ε-μ))+1
            if abs(ε) > get_quapi_tol()
                -im*exp(-ε*j*δτ)*exp(im*ε*k*δt)*(exp(-ε*δτ)-1)*(exp(im*ε*δt)-1)/(ε^2 * x)
            else
                -exp(-ε*j*δτ)*exp(im*ε*k*δt)*δτ*δt / x
            end
        else
            x = exp(safe_mult(β, ε-μ)) + 1
            y = exp(safe_mult(β-j*δτ, ε-μ)) * exp(-j*δτ*μ)
            if abs(ε) > get_quapi_tol()
                -im*y*exp(im*ε*k*δt)*(exp(-ε*δτ)-1)*(exp(im*ε*δt)-1)/(ε^2 * x)
            else
                -y*exp(im*ε*k*δt)*δτ*δt / x
            end
        end
    end
end

function _fermionic_lₖⱼ_disperse(f::AbstractBoundedFunction, disperse, β, μ, k::Int, j::Int, δt, δτ)
    g = _fermionic_gₖⱼ(β, μ, k, j, δt, δτ)
    g′ = ϵ -> g(disperse(ϵ))
    return f * g′
end
_fermionic_lₖⱼ(f::AbstractBoundedFunction, β, μ, k::Int, j::Int, δt, δτ) = f * _fermionic_gₖⱼ(β, μ, k, j, δt, δτ)
function _fermionic_gₖⱼ(β, μ, k::Int, j::Int, δt, δτ)
    return ε -> begin
        if β >= μ
            x = exp(-safe_mult(β, ε-μ))+1
            y = exp(-safe_mult(β - j*δτ, ε-μ)) * exp(j*δτ*μ)
            if abs(ε) > get_quapi_tol()
                im*y*exp(-im*ε*k*δt)*(exp(ε*δτ)-1)*(exp(-im*ε*δt)-1)/(ε^2 * x)
            else
                y*exp(-im*ε*k*δt)*δτ*δt / x
            end
        else
            x = exp(safe_mult(β, ε-μ))+1
            y = exp(j*δτ * ε)

            if abs(ε) > get_quapi_tol()
                im*y*exp(-im*ε*k*δt)*(exp(ε*δτ)-1)*(exp(-im*ε*δt)-1)/(ε^2 * x)
            else
                y*exp(-im*ε*k*δt)*δτ*δt / x
            end
        end
    end
end
