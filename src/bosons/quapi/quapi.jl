# bosons/quapi —— 玻色冻结核装配（自 QuAPI/src/{imagtime,realtime,mixedtime}/bosonic.jl 移植；
# 共享 g 内核与 safe_mult 集中到 util.jl，include 序 util 最先）
include("util.jl")
include("realtime.jl")
include("imagtime.jl")
include("mixedtime.jl")
