# fermions/exact —— 费米浴严格 IF 核（ExactEqualDistantKernel）
# **纯耦合项严格核**族契约（2026-11-22，见 real.jl exact_fermionic_Δt docstring）：
# 核内只含浴 + 杂质-浴耦合（无 ε_d/劈分阶数/kick 相位），消费收缩 = 同调度
# 直接 Trotter 离散链的迹（任意 ε_d 机器精确）。入口（签名同冻结核）：
#   imag.jl  exact_fermionic_Δτ（谱闭式；kick 线性析出 Δ_IF(x)=x·Δ_IF(1)）；
#   real.jl  exact_fermionic_Δt（FormB 消元，2026-11-22 契约换代）；
#   mixed.jl exact_fermionic_Δm（⚠️ 仍带 kicksτ/kickst 接口——契约换代待做）。
# 带踢算符忠实核构造 = 测试 helper（test/exact/_kicks_helper.jl，非导出面）。
# 装配契约与金标准见 test/exact/fermionic_{imagtime,realtime,mixedtime}.jl。
include("eigensolve.jl")
include("imag.jl")
include("real.jl")
include("mixed.jl")
