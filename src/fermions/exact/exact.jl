# fermions/exact —— 费米浴严格 IF 核
# **纯耦合项严格核**族契约（2026-11-22，见 real.jl exact_fermionic_Δt docstring）：
# 核内只含浴 + 杂质-浴耦合（无 ε_d/劈分阶数/kick 相位），消费收缩 = 同调度
# 直接 Trotter 离散链的迹（任意 ε_d 机器精确）。入口 = real.jl
# exact_fermionic_Δt（FormB 消元，2026-11-22 契约换代；直连调用不经过
# scheme——虚轴/混合轴严格核 exact_fermionic_Δτ/Δm 已于 2026-11-25 退役，
# 恢复入口见 README todolist）。
# 带踢算符忠实核构造 = 测试 helper（test/exact/_kicks_helper.jl，非导出面）。
# 装配契约与金标准见 test/exact/fermionic_realtime.jl。
include("real.jl")
