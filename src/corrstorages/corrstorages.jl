# corrstorages —— 纯存储容器（无装配逻辑）
# ← QuAPI/src/{corrmat.jl, abstractcorr.jl, imagtime/imagcorr.jl,
#   realtime/realcorr.jl, mixedtime/mixedcorr.jl, bcs/bcscorr.jl} 原样移植
include("abstractcorr.jl")
include("corrmat.jl")
include("imagcorr.jl")
include("realcorr.jl")
include("mixedcorr.jl")
include("bcscorr.jl")
