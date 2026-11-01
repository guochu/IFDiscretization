# 测试本地锚：JW 多体生成元（← QuAPI/exactif/brute.jl 的 mb_generator 原样拷贝，
# 出处注明；供 exact_bcs_check.jl 的 Fock ED 定义级锚定使用）
"""
    mb_generator(h) -> 2ⁿ×2ⁿ

单粒子 h（n 模，JW）的多体生成元。
"""
function mb_generator(h::AbstractMatrix)
    n = size(h, 1)
    dim = 1 << n
    G = zeros(ComplexF64, dim, dim)
    for J in 0:dim-1
        for j in 1:n                       # c_j（湮灭，作用在浴/杂质模 j）
            (J >> (j - 1)) & 1 == 0 && continue
            sj = count_ones(J & ((1 << (j - 1)) - 1))   # c_j 的 JW 符号
            Jj = J ⊻ (1 << (j - 1))
            for i in 1:n                   # c_i†
                (Jj >> (i - 1)) & 1 == 1 && continue
                si = count_ones(Jj & ((1 << (i - 1)) - 1))  # c_i† 的 JW 符号
                Istate = Jj ⊻ (1 << (i - 1))
                h[i, j] == 0 && continue
                G[Istate+1, J+1] += (-1)^(sj + si) * h[i, j]
            end
        end
    end
    return G
end
