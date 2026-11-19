# strictmixed.jl —— 混合 Kadanoff–Baym 轮廓严格核容器

"""
    StrictMixedCorrelationFunction(Gττ, Gτ₊, Gτ₋, G₊τ, G₋τ,
                                   G₊₊, G₊₋, G₋₊, G₋₋) <: AbstractMixedCorrelationFunction

混合 L 轮廓严格核（ExactKernel 族）的装配级存储：九个分支配对块
`G[b1,b2][i, j]` = 系数 ā^{b1}_i · a^{b2}_j。节点序同冻结核 contour_ordering
混合序：τ 节点 `i = 1..Nτ`（时序）、Keldysh 节点 `j = 1..Nt+1`（每节点
+ 先 − 后）。九块相互独立（bath Schur 产物 + 两端闭合边破缺共轭对称），
无 sector 关系。`branch(x, b1, b2)` 按分支对直接返回对应字段（无损）。
"""
struct StrictMixedCorrelationFunction <: AbstractMixedCorrelationFunction
    Gττ::Matrix{ComplexF64}
    Gτ₊::Matrix{ComplexF64}
    Gτ₋::Matrix{ComplexF64}
    G₊τ::Matrix{ComplexF64}
    G₋τ::Matrix{ComplexF64}
    G₊₊::Matrix{ComplexF64}
    G₊₋::Matrix{ComplexF64}
    G₋₊::Matrix{ComplexF64}
    G₋₋::Matrix{ComplexF64}
end

branch(x::StrictMixedCorrelationFunction, b1::Symbol, b2::Symbol) =
    _strict_block(x, b1, b2)
branch(x::StrictMixedCorrelationFunction; b1::Symbol, b2::Symbol) = branch(x, b1, b2)

function _strict_block(x::StrictMixedCorrelationFunction, b1::Symbol, b2::Symbol)
    b1 === :τ && return b2 === :τ ? x.Gττ : b2 === :+ ? x.Gτ₊ : x.Gτ₋
    b1 === :+ && return b2 === :τ ? x.G₊τ : b2 === :+ ? x.G₊₊ : x.G₊₋
    return b2 === :τ ? x.G₋τ : b2 === :+ ? x.G₋₊ : x.G₋₋
end

Base.size(x::StrictMixedCorrelationFunction, i::Int...) = size(x.Gττ, 1)
Base.show(io::IO, ::MIME"text/plain", x::StrictMixedCorrelationFunction) =
    print(io, "Strict Mixed Correlation Function [$(size(x.Gττ, 1))×$(size(x.G₊₊, 1))] (9-block pairing)")
function Base.:(==)(x::StrictMixedCorrelationFunction, y::StrictMixedCorrelationFunction)
    all(getfield(x, f) == getfield(y, f)
        for f in fieldnames(StrictMixedCorrelationFunction))
end
function Base.isapprox(x::StrictMixedCorrelationFunction, y::StrictMixedCorrelationFunction;
                       kwargs...)
    all(isapprox(getfield(x, f), getfield(y, f); kwargs...)
        for f in fieldnames(StrictMixedCorrelationFunction))
end
