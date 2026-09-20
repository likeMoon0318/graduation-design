# -*- coding: utf-8 -*-
# 差分进化算法 DE/rand/1/bin
# =====================================================================
# 变异：v = x_r1 + F (x_r2 - x_r3)      交叉：二项式交叉（逐维以 CR 概率替换）
# 选择：贪婪选择（子代优于父代才替换，保证种群质量不退化）

import time

import numpy as np

from .base import Counter, Recorder, build_result, clip, evaluate_ok, init_population


def run(bench, dim, max_iter=200, pop_size=40, seed=1, stride=1, keep_traj=False,
        F=0.5, CR=0.9, strategy="rand1"):
    """执行一次差分进化寻优。"""
    rng = np.random.default_rng(seed)
    lb, ub = bench.bounds(dim)
    counter = Counter()
    rec = Recorder(stride, keep_traj, final_gen=max_iter)

    pop = init_population(rng, pop_size, lb, ub)
    fit = evaluate_ok(bench, pop, counter)
    bi = int(np.argmin(fit))
    best_x, best_f = pop[bi].copy(), float(fit[bi])
    rec.record(0, best_f, float(fit.mean()), pop, best_x)

    t0 = time.perf_counter()
    for gen in range(1, max_iter + 1):
        # 为每个个体抽取三个互不相同的同伴（且不等于自身）
        idx = np.tile(np.arange(pop_size)[:, None], (1, 3))
        for j in range(3):
            shift = rng.integers(1, pop_size, size=pop_size)
            idx[:, j] = (np.arange(pop_size) + shift) % pop_size
        r1, r2, r3 = pop[idx[:, 0]], pop[idx[:, 1]], pop[idx[:, 2]]

        if strategy == "best1":
            base = np.tile(best_x, (pop_size, 1))
        else:
            base = r1
        mutant = clip(base + F * (r2 - r3), lb, ub)

        # 二项式交叉：保证至少有一维来自变异向量
        cross = rng.random((pop_size, dim)) < CR
        jrand = rng.integers(0, dim, size=pop_size)
        cross[np.arange(pop_size), jrand] = True
        trial = np.where(cross, mutant, pop)

        tfit = evaluate_ok(bench, trial, counter)
        better = tfit < fit
        pop[better], fit[better] = trial[better], tfit[better]
        bi = int(np.argmin(fit))
        if fit[bi] < best_f:
            best_f, best_x = float(fit[bi]), pop[bi].copy()
        rec.record(gen, best_f, float(fit.mean()), pop, best_x)

    return build_result(bench, dim, best_x, best_f, rec, counter,
                        time.perf_counter() - t0)
