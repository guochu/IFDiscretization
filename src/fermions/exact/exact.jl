# fermions/exact —— 费米浴严格 IF 核（ExactIF）
# 纯浴对象契约（2026-11-18）：只含三个入口 exact_fermionic_Δτ/Δt/Δm
#（签名同冻结核，无杂质信息）；点值格林函数、(order,x) 特定严格核、
# Ng–Reichman 过程张量与 Trotter 参照机均为测试侧 helper
#（test/exact/_exactif_helpers.jl）。
include("eigensolve.jl")
include("fermionic.jl")
