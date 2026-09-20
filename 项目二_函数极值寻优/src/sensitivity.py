# -*- coding: utf-8 -*-
# 参数敏感性实验：考察各算法关键参数对寻优性能的影响
# =====================================================================
# 实验设置：在 10 维下选取 4 个代表性函数（Sphere / Rosenbrock / Rastrigin / Ackley），
#          每个参数取值重复 10 次（种子 1..10），统计平均误差与成功率。
# 输出：results/tables/参数敏感性.csv

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import MAX_ITER, POP_SIZE, TABLE_DIR  # noqa: E402
from algorithms import ALGORITHMS  # noqa: E402
from functions import BENCHMARKS  # noqa: E402

DIM = 10
FUNCS = ["sphere", "rosenbrock", "rastrigin", "ackley"]
SEEDS = list(range(1, 11))

# 参数网格：算法 -> [(参数名, 候选取值)]
PARAM_GRIDS = {
    "GA": [
        ("pc", [0.6, 0.75, 0.9, 1.0]),
        ("pm_factor", [0.2, 0.5, 1.0, 2.0, 5.0]),
    ],
    "PSO": [
        ("w", [0.2, 0.4, 0.6, 0.729, 0.9]),
        ("c1", [0.5, 1.0, 1.49445, 2.0, 2.5]),
    ],
    "DE": [
        ("F", [0.2, 0.4, 0.6, 0.8, 1.0]),
        ("CR", [0.1, 0.3, 0.6, 0.9, 1.0]),
    ],
    "SA": [
        ("step_ratio", [0.01, 0.02, 0.05, 0.10, 0.20]),
        ("cool_power", [0.2, 0.3, 0.5, 0.8]),
    ],
    "GWO": [
        ("pop_factor", [0.5, 1.0, 2.0]),
    ],
}


def evaluate_setting(algo, param, value, seeds=SEEDS):
    """固定一个参数取值，在 4 个函数上各重复若干次，返回平均误差与成功率。"""
    errors = []
    for key in FUNCS:
        bench = BENCHMARKS[key]
        for seed in seeds:
            kw = {}
            if param == "pm_factor":
                kw["pm"] = value / DIM          # 变异概率相对 1/dim 的倍数
            elif param == "pop_factor":
                pass                            # 只改变种群规模
            else:
                kw[param] = value
            pop = int(POP_SIZE * value) if param == "pop_factor" else POP_SIZE
            res = ALGORITHMS[algo](bench, DIM, max_iter=MAX_ITER, pop_size=pop,
                                   seed=seed, **kw)
            errors.append(res["error"])
    e = np.array(errors, dtype=float)
    return {"算法": algo, "参数": param, "取值": value,
            "平均误差": float(e.mean()),
            "误差中位数": float(np.median(e)),
            "log10平均误差": float(np.log10(e.mean() + 1e-12)),
            "成功率": float(np.mean(e < 1e-2)),
            "样本数": len(e)}


def main():
    out_file = TABLE_DIR / "参数敏感性.csv"
    rows = []
    t0 = time.perf_counter()
    print("=" * 78)
    print("参数敏感性实验（10 维，4 个代表性函数 x 10 次重复）")
    print("=" * 78)
    for algo, grids in PARAM_GRIDS.items():
        for param, values in grids:
            for v in values:
                row = evaluate_setting(algo, param, v)
                rows.append(row)
                print(f"  {algo:<5} {param:<12} = {v:<8} 平均误差 {row['平均误差']:.4e} "
                      f"成功率 {row['成功率']:.0%}")
    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\n[输出] {out_file.relative_to(ROOT)}（{len(df)} 行，"
          f"耗时 {time.perf_counter() - t0:.1f} 秒）")
    return df


if __name__ == "__main__":
    main()
