# -*- coding: utf-8 -*-
# 批量实验：12 个测试函数 x 8 种算法 x 30 次独立运行
# =====================================================================
# 实验设计：
#   - 维度：2（可视化）、10（主实验）、30（可扩展性）
#   - 低维函数（Himmelblau / Beale / Six-Hump Camel）仅在 2 维参与
#   - 2 维额外加入 网格搜索 GRID 与 随机搜索 RANDOM 两个基线
#   - 随机种子固定为 1..30，任何一次运行的随机性都可复现
#   - 断点续跑：每个 (维度, 函数, 算法) 任务的结果单独落盘，重跑时自动跳过
#
# 用法：
#   python src/experiment.py                  # 全量实验（自动并行）
#   python src/experiment.py --dims 2         # 只跑 2 维
#   python src/experiment.py --algos GA,PSO   # 只跑指定算法
#   python src/experiment.py --force          # 忽略已有断点，全部重跑

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import (DIMS, HISTORY_STRIDE, MAX_ITER, PART_DIR, POP_SIZE,  # noqa: E402
                    PROCESSED_DIR, RAW_DIR, RUNS_FINAL, RUNS_HISTORY, SEEDS)
from algorithms import ALGORITHMS, BASELINES, DEFAULT_PARAMS  # noqa: E402
from functions import BENCHMARKS, functions_for_dim  # noqa: E402

# 每个维度参与的算法集合：2 维额外加入两种基线方法
ALGOS_BY_DIM = {
    2: list(ALGORITHMS) + list(BASELINES),
    10: list(ALGORITHMS),
    30: list(ALGORITHMS),
}
ALL_RUNNERS = {**ALGORITHMS, **BASELINES}

FINAL_COLUMNS = ["dim", "func", "func_cn", "category", "algo", "run", "best_value",
                 "error", "success", "convergence_gen", "n_evals", "elapsed", "best_x"]
HIST_COLUMNS = ["dim", "func", "func_cn", "algo", "run", "generation",
                "best_value", "mean_value", "best_error"]


def run_one(dim, key, algo, seed, max_iter, pop_size, stride):
    """执行一次独立运行，返回（最终结果行, 收敛过程 DataFrame）。"""
    bench = BENCHMARKS[key]
    runner = ALL_RUNNERS[algo]
    params = DEFAULT_PARAMS.get(algo, {})
    res = runner(bench, dim, max_iter=max_iter, pop_size=pop_size, seed=seed,
                 stride=stride, **params)

    final = {
        "dim": dim, "func": key, "func_cn": bench.cn, "category": bench.category,
        "algo": algo, "run": seed,
        "best_value": round(res["best_value"], 10),
        "error": res["error"],
        "success": bool(res["success"]),
        "convergence_gen": res["convergence_gen"] if res["convergence_gen"] is not None else -1,
        "n_evals": res["n_evals"],
        "elapsed": round(res["elapsed"], 6),
        "best_x": json.dumps([round(float(v), 6) for v in res["best_x"]]),
    }
    hist = pd.DataFrame({
        "dim": dim, "func": key, "func_cn": bench.cn, "algo": algo, "run": seed,
        "generation": res["history_gen"],
        "best_value": res["history_best"],
        "mean_value": res["history_mean"],
    })
    hist["best_error"] = np.abs(hist["best_value"] - bench.fmin)
    return final, hist


def run_task(dim, key, algo, seeds, max_iter, pop_size, stride, force=False):
    """执行一个 (维度, 函数, 算法) 任务下的全部独立重复，并落盘断点。"""
    final_path = PART_DIR / f"final_{dim}_{key}_{algo}.csv"
    hist_path = PART_DIR / f"hist_{dim}_{key}_{algo}.csv"
    if not force and final_path.exists() and hist_path.exists():
        return dim, key, algo, "skip", 0.0

    t0 = time.perf_counter()
    finals, hists = [], []
    for seed in seeds:
        f, h = run_one(dim, key, algo, seed, max_iter, pop_size, stride)
        finals.append(f)
        hists.append(h)
    pd.DataFrame(finals, columns=FINAL_COLUMNS).to_csv(final_path, index=False,
                                                       encoding="utf-8-sig")
    pd.concat(hists, ignore_index=True)[HIST_COLUMNS].to_csv(
        hist_path, index=False, encoding="utf-8-sig")
    return dim, key, algo, "done", time.perf_counter() - t0


def build_tasks(dims, algos_filter=None, funcs_filter=None):
    tasks = []
    for dim in dims:
        algos = [a for a in ALGOS_BY_DIM[dim] if not algos_filter or a in algos_filter]
        for key in functions_for_dim(dim):
            if funcs_filter and key not in funcs_filter:
                continue
            for algo in algos:
                tasks.append((dim, key, algo))
    return tasks


def merge_parts(dims):
    """把断点文件合并为 runs_final.csv 与 runs_history.csv。"""
    finals, hists = [], []
    for dim in dims:
        for p in sorted(PART_DIR.glob(f"final_{dim}_*.csv")):
            finals.append(pd.read_csv(p))
        for p in sorted(PART_DIR.glob(f"hist_{dim}_*.csv")):
            hists.append(pd.read_csv(p))
    if not finals:
        return None, None
    f = pd.concat(finals, ignore_index=True).sort_values(["dim", "func", "algo", "run"])
    h = pd.concat(hists, ignore_index=True).sort_values(
        ["dim", "func", "algo", "run", "generation"])
    f.to_csv(RUNS_FINAL, index=False, encoding="utf-8-sig")
    h.to_csv(RUNS_HISTORY, index=False, encoding="utf-8-sig")
    return f, h


def main(argv=None):
    ap = argparse.ArgumentParser(description="函数极值寻优 —— 批量实验")
    ap.add_argument("--dims", default=",".join(map(str, DIMS)),
                    help="参与实验的维度，逗号分隔")
    ap.add_argument("--algos", default="", help="只运行指定算法，逗号分隔")
    ap.add_argument("--funcs", default="", help="只运行指定函数，逗号分隔")
    ap.add_argument("--seeds", type=int, default=len(SEEDS),
                    help="每组的独立重复次数（默认 30）")
    ap.add_argument("--max-iter", type=int, default=MAX_ITER)
    ap.add_argument("--pop-size", type=int, default=POP_SIZE)
    ap.add_argument("--jobs", type=int, default=0, help="并行进程数，0 表示自动")
    ap.add_argument("--force", action="store_true", help="忽略断点，全部重跑")
    args = ap.parse_args(argv)

    dims = [int(d) for d in args.dims.split(",") if d.strip()]
    algos_filter = {a.strip() for a in args.algos.split(",") if a.strip()}
    funcs_filter = {f.strip() for f in args.funcs.split(",") if f.strip()}
    seeds = SEEDS[:args.seeds]

    unknown = (algos_filter | funcs_filter) - set(ALL_RUNNERS) - set(BENCHMARKS)
    if unknown:
        print(f"[错误] 未知的算法或函数名：{', '.join(sorted(unknown))}")
        return 2

    tasks = build_tasks(dims, algos_filter, funcs_filter)
    n_runs = len(tasks) * len(seeds)
    print("=" * 78)
    print("群智能优化算法函数极值寻优 —— 批量实验")
    print("=" * 78)
    print(f"维度        : {dims}")
    print(f"任务数      : {len(tasks)} 个 (维度 x 函数 x 算法)")
    print(f"独立重复    : {len(seeds)} 次 / 任务    合计 {n_runs} 次寻优")
    print(f"单次预算    : {args.max_iter} 代 x {args.pop_size} 个体 "
          f"= {args.max_iter * args.pop_size} 次函数评价")
    print(f"输出        : {RUNS_FINAL.name} / {RUNS_HISTORY.name}")
    print(f"断点目录    : {PART_DIR}")

    n_jobs = args.jobs if args.jobs > 0 else max(1, min(6, (os.cpu_count() or 4) - 1))
    t0 = time.perf_counter()
    results = []
    try:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(run_task)(dim, key, algo, seeds, args.max_iter, args.pop_size,
                              HISTORY_STRIDE[dim], args.force)
            for dim, key, algo in tasks)
    except ImportError:
        print("[提示] 未安装 joblib，改为串行执行")
        for i, (dim, key, algo) in enumerate(tasks, 1):
            res = run_task(dim, key, algo, seeds, args.max_iter, args.pop_size,
                           HISTORY_STRIDE[dim], args.force)
            results.append(res)
            print(f"  [{i}/{len(tasks)}] {dim}D {key} {algo} -> {res[3]}")

    done = [r for r in results if r[3] == "done"]
    skipped = [r for r in results if r[3] == "skip"]
    print(f"\n[完成] 新执行 {len(done)} 个任务，复用断点 {len(skipped)} 个任务，"
          f"耗时 {time.perf_counter() - t0:.1f} 秒")

    f, h = merge_parts(dims)
    if f is None:
        print("[警告] 没有可合并的结果")
        return 1
    print(f"[合并] {RUNS_FINAL.relative_to(ROOT)}：{len(f):,} 行 "
          f"({RUNS_FINAL.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"[合并] {RUNS_HISTORY.relative_to(ROOT)}：{len(h):,} 行 "
          f"({RUNS_HISTORY.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"\n成功率概览：{f['success'].mean():.1%}（判据 |f - f*| < 1e-2）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
