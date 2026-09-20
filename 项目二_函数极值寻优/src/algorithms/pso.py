# -*- coding: utf-8 -*-
# 粒子群算法 PSO 与自适应权重粒子群 APSO
# =====================================================================
# PSO ：收缩因子版本（w=0.729, c1=c2=1.49445），速度越界时直接清零该维
# APSO：惯性权重线性递减 0.9 -> 0.4，同时 c1 递减、c2 递增
#       （前期重视个体经验以增强探索，后期重视群体最优以加速收敛）

import time

import numpy as np

from .base import Counter, Recorder, build_result, evaluate_ok


def _run_core(bench, dim, max_iter, pop_size, seed, stride, keep_traj,
              w0, w1, c1_0, c1_1, c2_0, c2_1):
    rng = np.random.default_rng(seed)
    lb, ub = bench.bounds(dim)
    span = ub - lb
    counter = Counter()
    rec = Recorder(stride, keep_traj, final_gen=max_iter)

    X = lb + rng.random((pop_size, dim)) * span
    V = (rng.random((pop_size, dim)) - 0.5) * 0.2 * span
    fX = evaluate_ok(bench, X, counter)

    pbest, pbest_f = X.copy(), fX.copy()
    gi = int(np.argmin(fX))
    gbest, gbest_f = X[gi].copy(), float(fX[gi])
    rec.record(0, gbest_f, float(fX.mean()), X, gbest)

    t0 = time.perf_counter()
    for gen in range(1, max_iter + 1):
        r = gen / max_iter
        w = w0 + (w1 - w0) * r                 # 惯性权重
        c1 = c1_0 + (c1_1 - c1_0) * r          # 个体学习因子
        c2 = c2_0 + (c2_1 - c2_0) * r          # 社会学习因子
        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = X + V
        # 越界修复：截断到边界，并把该维速度清零（防止粒子持续“贴边”）
        out = (X < lb) | (X > ub)
        X = np.clip(X, lb, ub)
        V[out] = 0.0

        fX = evaluate_ok(bench, X, counter)
        improved = fX < pbest_f
        pbest[improved], pbest_f[improved] = X[improved], fX[improved]
        gi = int(np.argmin(pbest_f))
        if pbest_f[gi] < gbest_f:
            gbest_f, gbest = float(pbest_f[gi]), pbest[gi].copy()
        rec.record(gen, gbest_f, float(fX.mean()), X, gbest)

    return build_result(bench, dim, gbest, gbest_f, rec, counter,
                        time.perf_counter() - t0)


def run(bench, dim, max_iter=200, pop_size=40, seed=1, stride=1, keep_traj=False,
        w=0.729, c1=1.49445, c2=1.49445):
    """标准粒子群算法（常数参数）。"""
    return _run_core(bench, dim, max_iter, pop_size, seed, stride, keep_traj,
                     w, w, c1, c1, c2, c2)


def run_apso(bench, dim, max_iter=200, pop_size=40, seed=1, stride=1, keep_traj=False,
             w0=0.9, w1=0.4, c1_0=2.5, c1_1=0.5, c2_0=0.5, c2_1=2.5):
    """自适应权重粒子群算法（参数随代数线性调度）。"""
    return _run_core(bench, dim, max_iter, pop_size, seed, stride, keep_traj,
                     w0, w1, c1_0, c1_1, c2_0, c2_1)
