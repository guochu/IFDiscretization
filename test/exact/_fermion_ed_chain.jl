# _fermion_ed_chain.jl —— 费米 uniform 核世界线证书的 ED 参照（Fock 门序列链；JW；
# 模式 1 = 杂质（bit 0），浴模 k = mode 1+k（bit k）；维数 2^(K+1)）
#
# 调度 = GTEMPO native_schedule 同源：每 gap 先门 D(link_g) = e^{−i·link_g·ε_d·n_d} 后
# cell C(cell_g) = e^{−i·cell_g·H}（g = 1..N；末 link_{N+1} 为开放端窗）。参照侧全 Fock 矩阵，
# 不做任何谱恒等式加速。ρ₀ = ρ0v ⊗ ρ_bath（真空 [1 0; 0 0] 或占据 [0 0; 0 1]）。

"`ed(nmodes, cells, links, β, εd; ρ0v) -> (Z, n̄)`；谱数据取自 IFD._quad_nodes（死模剔除）。"
function fermion_ed_chain(f, nmodes::Int, cells::AbstractVector, links::AbstractVector, β::Real, εd::Real;
                          ρ0v=ComplexF64[1 0; 0 0])
    Es, Vsq, wΓ = IFD._quad_nodes(f, 0.0, nmodes)
    keep = wΓ .> 0
    εs, Vs = Es[keep], Vsq[keep]
    K = length(εs)
    K >= 1 || throw(ArgumentError("谱在支撑上的积分为零"))
    h0 = zeros(ComplexF64, K + 1, K + 1)
    for k in 1:K
        h0[1, 1+k] = conj(Vs[k]); h0[1+k, 1] = Vs[k]; h0[1+k, 1+k] = εs[k]
    end
    dim = 1 << (K + 1)
    Gmb = mb_generator(h0)
    nb = [1 / (1 + exp(β * e)) for e in εs]
    ρbath = zeros(ComplexF64, dim ÷ 2, dim ÷ 2)
    for b in 0:(dim ÷ 2 - 1)
        ρbath[b+1, b+1] = prod((1 - nb[k], nb[k])[(b >> (k - 1)) & 1 + 1] for k in 1:K)
    end
    ρfull = kron(ρbath, Matrix{ComplexF64}(ρ0v))
    U = Matrix{ComplexF64}(I, dim, dim)
    for g in eachindex(links)
        D = Diagonal([exp(-im * links[g] * εd * (J & 1)) for J in 0:dim-1])
        C = (g > length(cells) || cells[g] == 0) ? Matrix{ComplexF64}(I, dim, dim) : exp(-1im * cells[g] .* Gmb)
        U = C * D * U
    end
    R = U * ρfull * U'
    Z = real(tr(R))
    T = reshape(R, 2, dim ÷ 2, 2, dim ÷ 2)
    n̄ = sum(T[2, B, 2, B] for B in 1:(dim ÷ 2)) / Z
    return Z, real(n̄)
end
