# Stage-1 validation: is the exact discrete process tensor Gaussian on the
# trajectory variables?  i.e. does I[S]/c == Pf(Θ[S]) for ALL 4-leg subsets?
using IFDiscretization, LinearAlgebra, Printf

Es = [0.4]; Vs = [0.23 + 0.11im]; β = 1.3
N = 2; Δt = 0.4 / N
iv = if_path_values(Es, Vs, β, N, Δt)
c, Θ, Ivals = iv.c, iv.Θ, iv.Ivals
nvert = 4N
println("c = ", round(real(c), digits=6), "  (bath partition weight)")

"⊖-ordered Pfaffian of the submatrix over the sorted subset"
function pfaff(Θ::AbstractMatrix, subset::Vector{Int})
    k = length(subset)
    iseven(k) || return 0.0 + 0.0im
    k == 2 && return Θ[subset[1], subset[2]]
    s = 0.0 + 0.0im
    for j in 2:k                                    # pair (1, j) expansion
        rest = [subset[i] for i in 1:k if i != 1 && i != j]
        s += (-1.0)^(j - 2) * Θ[subset[1], subset[j]] * pfaff(Θ, rest)
    end
    return s
end

function check4(c, Θ, Ivals, nvert)
    worst = 0.0; ngood = 0; ntot = 0; nzero = 0
    for S in 0:(1 << nvert) - 1
        cnt = count_ones(S)
        cnt < 4 && continue
        subset = [i for i in 1:nvert if (S >> (i - 1)) & 1 == 1]
        pred = c * pfaff(Θ, subset)
        absd = abs(pred - Ivals[S+1])
        ntot += 1
        abs(Ivals[S+1]) < 1e-30 && (nzero += 1)
        if absd > 1e-12 * max(abs(pred), abs(Ivals[S+1]), 1e-30)
            worst = max(worst, absd)
            if ntot - ngood ≤ 6
                subset_str = join(subset, ",")
                @printf("  MISMATCH bits=%#x legs=(%s): |I|=%.3e |Pf·c|=%.3e\n",
                        S, subset_str, abs(Ivals[S+1]), abs(pred))
            end
        else
            ngood += 1
        end
    end
    return worst, ngood, ntot, nzero
end
worst, ngood, ntot, nzero = check4(c, Θ, Ivals, nvert)
@printf("4-leg subsets: %d/%d match Pf(Θ)·c (worst ABS err %.2e, zero-I configs: %d)\n", ngood, ntot, worst, nzero)

function check6(c, Θ, Ivals, nvert)
    bad6 = 0; tot6 = 0
    for S in 0:(1 << nvert) - 1
        count_ones(S) == 6 || continue
        subset = [i for i in 1:nvert if (S >> (i - 1)) & 1 == 1]
        pred = c * pfaff(Θ, subset)
        tot6 += 1
        abs(pred - Ivals[S+1]) < 1e-10 * max(abs(Ivals[S+1]), 1e-14) || (bad6 += 1)
    end
    return bad6, tot6
end
bad6, tot6 = check6(c, Θ, Ivals, nvert)
@printf("6-leg subsets: %d/%d mismatches\n", bad6, tot6)
