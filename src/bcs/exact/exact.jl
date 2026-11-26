# bcs/exact —— BCS 浴严格 IF 核（实轴，粒子字典 FormB 链）
# **纯耦合项严格核**族契约（见 fermions/exact/real.jl exact_fermionic_Δt
# docstring；BCS 同族对齐，plan/bcs-realtime-exact-kernel-plan.md）：
# 核内只含 BCS 浴（配对二次型）+ 杂质-浴耦合（无 ε_d/劈分阶数/kick 相位），
# 消费收缩 = 同调度直接 Trotter 离散链的迹（任意 ε_d 机器精确）。
# 入口 = realtime.jl exact_bcs_Δt（exact_bcs_Δτ/Δm 退役，恢复入口见 README todolist）。
# **消费契约 = 冻结核字面放置语义**（= GTEMPO AnomalousPairing 四键消费：两味皆
# 物理粒子、ρ₀ 真空、ud/du 作反常项、dd 直读）——下游与冻结核同一装配路径，
# 零特例（ED 端到端证书 = test/exact/bcs_realtime.jl，消费参考实现 =
# test/exact/_worldline.jl）。2026-11-25 的 Nambu 空穴字典版本已退役（@ git 82a2368）。
# 退役 raw 收缩闭式（Nambu 本征分解）= 测试参照 helper
#（test/exact/_bcs_raw_reference.jl；raw 收缩 ≠ 消元核系数，二者共同锚 = ED）。
include("realtime.jl")
