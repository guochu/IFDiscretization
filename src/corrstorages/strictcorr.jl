# strictcorr.jl —— 严格核（ExactKernel 族）的装配级载体
#
# 严格核输出的配对矩阵 Δ 是轮廓腿基上的 (ā 行 × a 列) 双线性系数矩阵，
# 分支块相互独立（bath Schur 消元的产物），不满足点值容器的
# unified-convention sector 关系（branch 推导有损）——故用专用容器携带
# **完整 Δ + 腿基**，`branch` 对严格载体直接切片返回真块（无损）。

for (T, sup) in ((:StrictRealCorrelationFunction, :AbstractRealCorrelationFunction),
                 (:StrictMixedCorrelationFunction, :AbstractMixedCorrelationFunction))
    @eval begin
        """
            $($T)(basis, Δ) <: $($sup)

        严格核（ExactKernel 族）的装配级载体：`basis` 为链序腿列表
        （每节点 (a, ā) 相邻），`Δ` 为这些腿上的双线性配对矩阵
        （`Δ[i, j]` = ξ_i ξ_j 的系数；ā 行 × a 列）。与点值容器的 sector
        语义无关——`branch` 切片返回真块（无损）。
        """
        struct $T <: $sup
            basis::Vector{GrassmannContourIndex}
            Δ::Matrix{ComplexF64}
            function $T(basis::Vector{GrassmannContourIndex},
                        Δ::Matrix{ComplexF64})
                n = length(basis)
                size(Δ) == (n, n) ||
                    throw(DimensionMismatch("Δ 须为腿基同阶矩阵（$n×$n），得 $(size(Δ))"))
                new(basis, Δ)
            end
        end
    end
end

"严格载体的分支腿位切片：`branch(x, b1, b2)` = (ā^{b1} 行, a^{b2} 列) 真块。"
function branch(x::Union{StrictRealCorrelationFunction, StrictMixedCorrelationFunction},
                b1::Symbol, b2::Symbol)
    rows = [i for (i, b) in enumerate(x.basis) if b.branch == b1 && b.conj]
    cols = [j for (j, b) in enumerate(x.basis) if b.branch == b2 && !b.conj]
    (isempty(rows) || isempty(cols)) &&
        throw(ArgumentError("分支 ($b1, $b2) 不在该载体轮廓中"))
    return x.Δ[rows, cols]
end
branch(x::Union{StrictRealCorrelationFunction, StrictMixedCorrelationFunction};
       b1::Symbol, b2::Symbol) = branch(x, b1, b2)

branches(::StrictRealCorrelationFunction) = (:+, :-)

Base.size(x::Union{StrictRealCorrelationFunction, StrictMixedCorrelationFunction}, i::Int...) =
    div(size(x.Δ, 1), 2)

function Base.show(io::IO, ::MIME"text/plain",
                   x::Union{StrictRealCorrelationFunction, StrictMixedCorrelationFunction})
    print(io, nameof(typeof(x)), " [", size(x, 1), "] (strict pairing)")
end
Base.:(==)(x::T, y::T) where {T <: Union{StrictRealCorrelationFunction, StrictMixedCorrelationFunction}} =
    (x.basis == y.basis) && (x.Δ == y.Δ)
Base.isapprox(x::T, y::T; kwargs...) where {T <: Union{StrictRealCorrelationFunction, StrictMixedCorrelationFunction}} =
    (x.basis == y.basis) && isapprox(x.Δ, y.Δ; kwargs...)
