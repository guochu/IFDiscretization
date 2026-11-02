# exactif —— ExactIF 严格离散 IF 的金标准验证（费米实时）

Ng–Reichman 2023（PRB 107, 125103）实时路线：严格离散 IF（过程张量
I[{s}]，纯浴对象、不含 H_imp——论文正文 §III.B）+ 对角 kick 收缩
（ε_d/U 只在收缩侧进入）== 同劈分直接 n 阶 Trotter 演化（多体直乘）。

- `fermions_realtime.jl`：金标准误差标度（N=1..5 × 三组 ε/U 参数）。
  err(N) = max|ρ_contract − ρ_direct| 为机器精度平线——"严格"的定量含义。
- `plot_realtime_scaling.py`：标度图（平线 1e-14）。
- 被测机器在包内：`src/fermions/exact/{processtensor,contract,trotterref}.jl`
  （原样移植自 QuAPI/exactif 已验证链条：runtests L0/L5/L7）。
- 金标准断言常驻包测试：`test/exact/processtensor.jl`。

历史注记：本目录曾存放环配对（Schur 补）路线的探索脚本——该路线的装配
约定未走通且与论文 G 块/F 项表示重复，已删除；正确路线以本 README 所述
过程张量 + kick 收缩为准。
