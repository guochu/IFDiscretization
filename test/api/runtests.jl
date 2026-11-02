# =============================================================================
# test/api —— correlation-function API 测试
#
# 两类内容：
#   1. 语义/约定测试（imagtime/realtime/mixedtime/bcs）：branch 约定、容器、
#      Δm ↔ Δτ/Δt 逐元对齐、跨块共享、bath 便捷式；
#   2. 小规模物理正确性测试（spinboson_*、bcs_grassmann）：自旋-玻色 ED 对拍
#      与 Grassmann-Pfaffian 装配语义（2026-11-02 自 test/quapi 移入——
#      test/quapi 现在只承担自由模型上的 1/2 阶收敛断言）。
# =============================================================================
println("=============== test/api: correlation-function API tests ===============")
for f in ("imagtime", "realtime", "mixedtime", "bcs",
          "spinboson_imag", "spinboson_real", "spinboson_mixed", "bcs_grassmann")
    println(">>> $f")
    include(joinpath(@__DIR__, "$f.jl"))
end
