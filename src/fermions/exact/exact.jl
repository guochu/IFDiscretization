# fermions/exact —— 费米浴严格 IF 核（ExactIF）
# 纯浴对象契约（2026-11-18/19）：三个入口 exact_fermionic_Δτ/Δt/Δm
#（签名同冻结核，无杂质信息；Δt/Δm 另有可选 kicks 门相位接口）；
# 装配契约与金标准见 test/exact/fermionic_{imagtime,realtime,mixedtime}.jl。
# 点值格林函数、Trotter 参照机均为测试侧 helper。
include("eigensolve.jl")
include("fermionic.jl")
