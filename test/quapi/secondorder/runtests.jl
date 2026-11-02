# =============================================================================
# test/quapi/secondorder —— 冻结核 + K/IF 拆分（Weyl-K）装配的全局二阶验证
#
# 同一批冻结核条目（与 ../firstorder/ 完全相同的 9 个核函数），只改杂质 K 侧
# 装配即可把全局误差抬到二阶（本目录各扇区文件，performance/secondorder 树
# 的快照拷贝）：
#   · 杂质自由演化解析积出，以半步相位 f = e^{∓iϵ_d h/2} 进入（K/IF 拆分）；
#   · 核源取 Weyl ½ 梯形平均 φ_k = (A_k + A_{k+1})/2；
#   · 等时接触条目 d₀ 隐式并入对角；
#   · 虚时环的 KMS 跳变常数完整代入自由项与 φ₁ 左半，混合轮廓另含消峰 γ
#     收缩与 Duhamel 源。
# IF 数据本身一阶不变——「阶数由 K 侧修正承担」正是这组断言的验证对象。
# 目录结构镜像 performance/secondorder；common.jl 为 fermions/bcs 的模型与
# 严格参照快照。文件自包含（模块隔离），任意顺序 include 均可。
# =============================================================================
println("----------------------------------------------")
println("|  Frozen kernels, Weyl-K assembly: order 2  |")
println("----------------------------------------------")

for bath in ("bosons", "fermions", "bcs"), sector in ("realtime", "imagtime", "mixedtime")
    include(joinpath(@__DIR__, bath, "$sector.jl"))
end
