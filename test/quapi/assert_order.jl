# =============================================================================
# test/quapi/assert_order.jl —— 阶数断言辅助（firstorder/ 与 secondorder/ 共用）
#
# test/quapi 的职责（与 test/api 的语义/小规模正确性测试相对）：在自由玻色 /
# 自由费米 / BCS 三种全二次型 Toulouse 模型上（多项式复杂度的快速求解器——
# 单粒子链 / Dyson 求逆 / Nambu 矩阵递推，可支撑细网格长时间演化），验证
# 9 个冻结核函数（3 统计 × 3 轮廓）驱动的两种装配的全局收敛阶：
#   firstorder/   朴素 face-value 装配         ⇒ 一阶
#   secondorder/  杂质 K/IF 拆分（Weyl-K）装配  ⇒ 二阶
# 求解器是 performance/ 树的快照拷贝（见各扇区文件头注；不跨树 include）。
# =============================================================================

const ORDER_NS = (16, 32, 64)          # 逐倍加密阶梯（3 点 ⟹ 2 个阶数估计）
const ORDER_FRACS = (0.25, 0.5, 0.75)  # 取样节点（避开端点，与 performance 基准一致）

frac_nodes(N) = round.(Int, ORDER_FRACS .* N)

"""
    assert_order(errs, lo, hi)

误差阶梯断言：随 N 逐倍加密单调下降，且逐级收敛阶 log2(errᵢ/errᵢ₊₁) 全部
落在 [lo, hi]。一阶装配取 (0.75, 1.30)、二阶取 (1.80, 2.20)；容差依据存档
CSV 在 N=16..64 的实测范围（朴素 0.91–1.08，Weyl-K 1.96–2.01）。
"""
function assert_order(errs, lo, hi)
    @test all(diff(errs) .< 0)
    ords = [log2(errs[i] / errs[i+1]) for i in 1:length(errs)-1]
    @test all(lo .<= ords .<= hi)
    return ords
end
