#!/usr/bin/env python3
"""误差标度图：ExactIF 收缩 vs 直接演化的偏差（机器精度平线）。
输入：results/fermions_realtime_scaling.csv；输出：fermions_realtime_scaling.png"""
import csv
import collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = collections.defaultdict(list)
with open("results/fermions_realtime_scaling.csv") as f:
    for row in csv.DictReader(f):
        key = f"ε=({row['eps_up']},{row['eps_dn']}), U={row['U']}"
        data[key].append((int(row["N"]), float(row["err_max"])))

fig, ax = plt.subplots(figsize=(6.2, 4.2))
for key, pts in sorted(data.items()):
    ns = [p[0] for p in pts]
    errs = [max(p[1], 1e-17) for p in pts]
    ax.semilogy(ns, errs, "o-", label=key)
ax.axhline(1e-14, color="gray", ls="--", lw=0.8)
ax.text(1.02, 1.1e-14, "机器精度", color="gray", fontsize=8)
ax.set_xlabel("N（Trotter 步数）")
ax.set_ylabel("err = max |ρ_contract − ρ_direct|")
ax.set_title("ExactIF 严格离散 IF + kick 收缩 vs 直接 Trotter 演化\n误差 = 机器精度平线（不随 δt = 0.15/N 增长）")
ax.legend(fontsize=8)
ax.set_ylim(1e-17, 1e-9)
fig.tight_layout()
fig.savefig("results/fermions_realtime_scaling.png", dpi=150)
print("图已写入 results/fermions_realtime_scaling.png")
