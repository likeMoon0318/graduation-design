# -*- coding: utf-8 -*-
# 传统数学规划方法与基线方法
# =====================================================================
# BFGS       : 拟牛顿法（带边界的 L-BFGS-B），依赖梯度、收敛快但极易陷入局部最优
# NelderMead : 单纯形法，无需梯度，但高维退化严重
# 网格搜索   : 逐点枚举（仅 2 维可行，用于说明“维度灾难”）
# 随机搜索   : 在搜索空间内均匀采样，作为“无智能”基线

import time

import numpy as np
from scipy.optimize import minimize

from .base import Counter, Recorder, build_result


def _wrap_objective(bench, counter):
    """把目标函数包装成 scipy 需要的标量函数，同时统计评价次数。"""
    def obj(x):
        counter.add(1)
        return float(bench.fn(np.asarray(x, dtype=float).reshape(1, -1))[0])
    return obj


def _traditional(method, bench, dim, max_iter, seed, stride, keep_traj, label):
    rng = np.random.default_rng(seed)
    lb, ub = bench.bounds(dim)
    counter = Counter()
    rec = Recorder(stride, keep_traj, final_gen=max_iter)
    x0 = lb + rng.random(dim) * (ub - lb)          # 单次运行的随机初始点
    f0 = float(bench.fn(x0.reshape(1, -1))[0])
    counter.add(1)

    state = {"best_f": f0, "best_x": x0.copy(), "gen": 0}
    rec.record(0, f0, f0, None, x0)

    def callback(xk):
        state["gen"] += 1
        f = float(bench.fn(np.asarray(xk, dtype=float).reshape(1, -1))[0])
        counter.add(1)
        if f < state["best_f"]:
            state["best_f"], state["best_x"] = f, np.asarray(xk, dtype=float).copy()
        rec.record(state["gen"], state["best_f"], f, None, state["best_x"])

    t0 = time.perf_counter()
    options = {"maxiter": max_iter, "disp": False}
    res = minimize(_wrap_objective(bench, counter), x0, method=method,
                   bounds=list(zip(lb, ub)), callback=callback, options=options)
    elapsed = time.perf_counter() - t0

    fx = float(res.fun)
    xr = np.clip(np.asarray(res.x, dtype=float), lb, ub)
    if fx < state["best_f"]:
        state["best_f"], state["best_x"] = fx, xr
    return build_result(bench, dim, state["best_x"], state["best_f"], rec, counter, elapsed,
                        extra={"method": label, "scipy_success": bool(res.success),
                               "iters": int(getattr(res, "nit", 0))})


def run_bfgs(bench, dim, max_iter=200, pop_size=40, seed=1, stride=1, keep_traj=False):
    """拟牛顿法 BFGS（带边界实现 L-BFGS-B）。"""
    return _traditional("L-BFGS-B", bench, dim, max_iter, seed, stride, keep_traj,
                        "L-BFGS-B（BFGS 带边界）")


def run_nelder_mead(bench, dim, max_iter=200, pop_size=40, seed=1, stride=1, keep_traj=False):
    """单纯形法 Nelder-Mead。"""
    return _traditional("Nelder-Mead", bench, dim, max_iter, seed, stride, keep_traj,
                        "Nelder-Mead")


def run_grid(bench, dim, max_iter=200, pop_size=40, seed=1, stride=1, keep_traj=False,
             budget=None):
    """网格搜索基线（仅 2 维）：把评价预算平均分配到二维网格上逐点枚举。"""
    if dim != 2:
        raise ValueError("网格搜索仅用于 2 维问题")
    budget = int(budget or max_iter * pop_size)
    n = int(np.sqrt(budget))
    lb, ub = bench.bounds(2)
    xs = np.linspace(lb[0], ub[0], n)
    ys = np.linspace(lb[1], ub[1], n)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    counter = Counter()
    rec = Recorder(stride, keep_traj, final_gen=max_iter)
    t0 = time.perf_counter()
    vals = bench.fn(pts)
    counter.add(len(pts))
    run_best = np.minimum.accumulate(vals)
    for gen in range(0, max_iter + 1):
        pos = min(int(gen / max_iter * (len(pts) - 1)), len(pts) - 1)
        rec.record(gen, run_best[pos], float(np.mean(vals[:pos + 1])), None, pts[pos])
    bi = int(np.argmin(vals))
    return build_result(bench, 2, pts[bi], float(vals[bi]), rec, counter,
                        time.perf_counter() - t0, extra={"grid_points": n * n})


def run_random(bench, dim, max_iter=200, pop_size=40, seed=1, stride=1, keep_traj=False):
    """随机搜索基线：每代均匀采样 pop_size 个点，用于对照“智能”算法。"""
    rng = np.random.default_rng(seed)
    lb, ub = bench.bounds(dim)
    counter = Counter()
    rec = Recorder(stride, keep_traj, final_gen=max_iter)
    best_f, best_x = np.inf, lb.copy()
    t0 = time.perf_counter()
    for gen in range(0, max_iter + 1):
        X = lb + rng.random((pop_size, dim)) * (ub - lb)
        fX = bench.fn(X)
        counter.add(pop_size)
        bi = int(np.argmin(fX))
        if fX[bi] < best_f:
            best_f, best_x = float(fX[bi]), X[bi].copy()
        rec.record(gen, best_f, float(fX.mean()), None, best_x)
    return build_result(bench, dim, best_x, best_f, rec, counter,
                        time.perf_counter() - t0, extra={"strategy": "uniform random"})
