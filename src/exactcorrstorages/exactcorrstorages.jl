# exactcorrstorages —— 严格核（ExactKernel 族）的装配级存储容器
#
# 严格核输出的配对数据 = 轮廓分支块上的双线性系数矩阵（ā^{b1} × a^{b2}），
# 各块相互独立（bath Schur 消元的产物），不满足点值容器的 unified-convention
# sector 关系（branch 推导有损，数值定谳见 test/exact/fermionic_realtime.jl
# 头注）——故按点值容器的存储风格，以**稠密分支块为字段**携带全部独立数据，
# 不带腿基字段（轮廓节点序 = 各自 contour_ordering 约定，由块尺寸 + 类型
# 完全确定）。`branch` 按分支对直接返回对应字段（无损）。
# 2026-11-25 轴面收敛：ExactMixed 变体随混合轴严格核退役（strictmixed.jl 删，
# 恢复入口见 README todolist），现仅存实时容器。
include("strictreal.jl")
