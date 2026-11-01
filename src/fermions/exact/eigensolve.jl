# EigenSolver 的实现：GL-M 谱离散 + 浴 h₀ 本征分解
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

"纯浴 h₀（杂质位 1，ε=0）本征分解。"
function _bath_eig(Es, Vsq)
    M = length(Es)
    h0 = zeros(ComplexF64, M + 1, M + 1)
    for k in 1:M
        h0[1, 1+k] = Vsq[k]; h0[1+k, 1] = conj(Vsq[k]); h0[1+k, 1+k] = Es[k]
    end
    return eigen(Hermitian(h0))
end
