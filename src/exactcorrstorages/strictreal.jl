# strictreal.jl —— 实时 Keldysh 轮廓严格核容器

"""
    StrictRealCorrelationFunction(G₊₊, G₊₋, G₋₊, G₋₋) <: AbstractRealCorrelationFunction

实时 Keldysh 严格核（ExactKernel 族）的装配级存储：四个分支配对块
`G[b1,b2][i, j]` = 系数 ā^{b1}_i · a^{b2}_j（`i, j` = 各支时间节点 1..N+1，
时序即冻结核 contour_ordering 实时序）。四块相互独立（bath Schur 产物），
无 sector 关系（点值容器的 branch 推导在此不适用）。

`branch(x, b1, b2)` 按分支对直接返回对应字段（无损）。
"""
struct StrictRealCorrelationFunction <: AbstractRealCorrelationFunction
    G₊₊::Matrix{ComplexF64}
    G₊₋::Matrix{ComplexF64}
    G₋₊::Matrix{ComplexF64}
    G₋₋::Matrix{ComplexF64}
    function StrictRealCorrelationFunction(G₊₊::Matrix{ComplexF64},
                                           G₊₋::Matrix{ComplexF64},
                                           G₋₊::Matrix{ComplexF64},
                                           G₋₋::Matrix{ComplexF64})
        n = size(G₊₊, 1)
        all(m -> size(m) == (n, n), (G₊₋, G₋₊, G₋₋)) ||
            throw(DimensionMismatch("四块须同阶（$n×$n）"))
        new(G₊₊, G₊₋, G₋₊, G₋₋)
    end
end

branch(x::StrictRealCorrelationFunction, b1::Symbol, b2::Symbol) =
    _strict_block(x, b1, b2)
branch(x::StrictRealCorrelationFunction; b1::Symbol, b2::Symbol) = branch(x, b1, b2)

_strict_block(x::StrictRealCorrelationFunction, b1::Symbol, b2::Symbol) =
    b1 === :+ ? (b2 === :+ ? x.G₊₊ : x.G₊₋) : (b2 === :+ ? x.G₋₊ : x.G₋₋)

Base.size(x::StrictRealCorrelationFunction, i::Int...) = size(x.G₊₊, i...)
Base.show(io::IO, ::MIME"text/plain", x::StrictRealCorrelationFunction) =
    print(io, "Strict Real Correlation Function [$(size(x, 1))] (4-block pairing)")
Base.:(==)(x::StrictRealCorrelationFunction, y::StrictRealCorrelationFunction) =
    x.G₊₊ == y.G₊₊ && x.G₊₋ == y.G₊₋ && x.G₋₊ == y.G₋₊ && x.G₋₋ == y.G₋₋
Base.isapprox(x::StrictRealCorrelationFunction, y::StrictRealCorrelationFunction; kwargs...) =
    isapprox(x.G₊₊, y.G₊₊; kwargs...) && isapprox(x.G₊₋, y.G₊₋; kwargs...) &&
    isapprox(x.G₋₊, y.G₋₊; kwargs...) && isapprox(x.G₋₋, y.G₋₋; kwargs...)
