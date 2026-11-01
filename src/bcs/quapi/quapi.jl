# bcs/quapi —— BCS（超导）浴冻结核装配
# （自 QuAPI/src/bcs/{imagtime,realtime,mixedtime}.jl 移植；复用 fermions 的
#   fermionic_Δ* + disperse；Bogoliubov 系数与块包装在 corrstorages/bcscorr.jl）
include("imagtime.jl")
include("realtime.jl")
include("mixedtime.jl")
