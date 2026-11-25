# bcs/exact —— BCS 浴严格 IF 核（实轴，FormB Nambu 翻倍推广）
# **纯耦合项严格核**族契约（见 fermions/exact/real.jl exact_fermionic_Δt
# docstring；BCS 同族对齐，2026-11-25 计划 bcs-realtime-exact-kernel-plan.md）：
# 核内只含 BCS 浴（BdG 翻倍二次型）+ 杂质-浴耦合（无 ε_d/劈分阶数/kick 相位），
# 消费收缩 = 同调度直接 Trotter 离散链的迹（任意 ε_d 机器精确）。
# 入口 = realtime.jl exact_bcs_Δt（2026-11-25 轴面收敛退役后的单轴恢复；
# exact_bcs_Δτ/Δm 仍退役，恢复入口见 README todolist）。
# 带踢算符忠实核构造 = 测试 helper（test/exact/_bcs_kicks_helper.jl，非导出面）。
# 退役 raw 收缩闭式（Nambu 本征分解）= 测试参照 helper
#（test/exact/_bcs_raw_reference.jl；与 FormB 核非同一物体，共同锚 = ED）。
include("realtime.jl")
