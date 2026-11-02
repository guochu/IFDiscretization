# =============================================================================
# test/quapi/firstorder —— 冻结核朴素（face-value）装配的全局一阶验证
#
# 9 个核函数（3 统计 × 3 轮廓）各驱动一个朴素装配求解器（本目录各扇区文件，
# performance/firstorder 树的快照拷贝），在自由 Toulouse 模型上对严格参照取
# N=16/32/64 误差阶梯，断言全局收敛阶 ≈ 1。朴素装配特征：冻结核条目按
# face-value 配节点（无顶点移位）、矩形源（无 Weyl ½ 梯形平均）、整步自由
# 相位（无半步）——K 侧修正后的二阶见 ../secondorder/。
# 目录结构镜像 performance/firstorder：{bosons,fermions,bcs} × {realtime,
# imagtime,mixedtime}；文件自包含（模块隔离），任意顺序 include 均可。
# =============================================================================
println("----------------------------------------------")
println("|  Frozen kernels, naive assembly: order 1   |")
println("----------------------------------------------")

for bath in ("bosons", "fermions", "bcs"), sector in ("realtime", "imagtime", "mixedtime")
    include(joinpath(@__DIR__, bath, "$sector.jl"))
end
