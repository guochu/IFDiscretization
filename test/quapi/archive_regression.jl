# 存档回归测试：performance/ 全部基准输出（2 套件 × 3 浴 × 3 轮廓 = 18 个 CSV）
# 与本目录各存档文件夹逐字节一致。
#
#   一致 = 数值行为不变的最终防线：任何 branch 约定 / 构造器 / 内核改动只要
#   扰动一个字节，这里立刻暴露。CSV 更新需显式重新拷贝存档（有意识的行为变更）。
#
# ← QuAPI/test/quapi/archive_regression.jl 移植：存档 CSV 自 QuAPI/test/quapi/
#   archive_* 原样拷入（19 个中 18 个；firstorder/fermions 的 imagtime_dense
#   属 QuAPI.Dense 稠密张量变体，随 Dense 留 QuAPI，不回归）。
#   performance/ 脚本树同步移植（`using QuAPI` → `using IFDiscretization`；
#   *_dense.jl 脚本不迁）。
#
# 覆盖矩阵：
#   firstorder/{bosons,fermions,bcs}  × {realtime,imagtime,mixedtime}
#   secondorder/{bosons,fermions,bcs} × {realtime,imagtime,mixedtime}
using Test

const ROOT = joinpath(@__DIR__, "..", "..")
const PERF = joinpath(ROOT, "performance")
const SUITES = ("firstorder", "secondorder")
const BATHS = ("bosons", "fermions", "bcs")
const SECTORS = ("realtime", "imagtime", "mixedtime")

archive_dir(suite, bath) = joinpath(@__DIR__, "archive_$(suite)_$(bath)")
live_dir(suite, bath) = joinpath(PERF, suite, bath, "results")

function readcsv(path)
    lines = readlines(path)
    return lines[1], lines[2:end]
end

function run_and_compare(script, archive, live, name)
    @testset "$name" begin
        old = isfile(live) ? read(live, String) : nothing
        try
            run(`$(Base.julia_cmd()) --startup-file=no --project=$ROOT $script`)
            @test isfile(archive)
            @test isfile(live)
            ahead, arows = readcsv(archive)
            lhead, lrows = readcsv(live)
            @test lhead == ahead
            @test length(lrows) == length(arows)
            for (a, b) in zip(arows, lrows)
                @test a == b
            end
        finally
            if old === nothing
                rm(live; force=true)
            else
                write(live, old)
            end
        end
    end
end

@testset "archive regression (performance vs archived CSVs)" begin
    @testset "$suite $bath $sector" for suite in SUITES, bath in BATHS, sector in SECTORS
        run_and_compare(joinpath(PERF, suite, bath, "$sector.jl"),
                        joinpath(archive_dir(suite, bath), "$sector.csv"),
                        joinpath(live_dir(suite, bath), "$sector.csv"),
                        "$suite/$bath/$sector")
    end
    # QuAPI.Dense 的 imagtime_dense 变体（firstorder/fermions 特有）随 Dense 留
    # QuAPI/test/quapi/archive_regression.jl，不在此回归。
end
