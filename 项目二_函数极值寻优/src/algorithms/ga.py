# -*- coding: utf-8 -*-
# 遗传算法 GA（实数编码）
# =====================================================================
# 选择：二元锦标赛    交叉：模拟二进制交叉 SBX    变异：多项式变异
# 策略：精英保留（每代直接复制最优 2 个个体，保证收敛过程单调）

import time

import numpy as np

from .base import Counter, Recorder, build_result, clip, evaluate_ok, init_population


def _tournament(rng, fitness, k=2):
    """二元锦标赛选择，返回被选中的索引数组。"""
    n = len(fitness)
    idx = rng.integers(0, n, size=(n, k))
    winner = idx[np.arange(n), np.argmin(fitness[idx], axis=1)]
    return winner


def _sbx(rng, p1, p2, lb, ub, eta=15.0, pc=0.9):
    """模拟二进制交叉（SBX）：实数编码遗传算法的标准交叉算子。"""
    n, dim = p1.shape
    c1, c2 = p1.copy(), p2.copy()
    mask = rng.random((n, dim)) < pc
    u = rng.random((n, dim))
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    c1 = np.where(mask, 0.5 * ((1 + beta) * p1 + (1 - beta) * p2), c1)
    c2 = np.where(mask, 0.5 * ((1 - beta) * p1 + (1 + beta) * p2), c2)
    return clip(c1, lb, ub), clip(c2, lb, ub)


def _poly_mutation(rng, X, lb, ub, eta=20.0, pm=None):
    """多项式变异：$$x' = x + \\delta (ub-lb)$$，pm 为逐维变异概率。"""
    n, dim = X.shape
    pm = 1.0 / dim if pm is None else pm
    mask = rng.random((n, dim)) < pm
    u = rng.random((n, dim))
    delta = np.where(u < 0.5,
                     (2 * u) ** (1 / (eta + 1)) - 1,
                     1 - (2 * (1 - u)) ** (1 / (eta + 1)))
    X = X + mask * delta * (ub - lb)
    return clip(X, lb, ub)


def run(bench, dim, max_iter=200, pop_size=40, seed=1, stride=1, keep_traj=False,
        pc=0.9, pm=None, eta_c=15.0, eta_m=20.0, elite=2):
    """执行一次遗传算法寻优。"""
    rng = np.random.default_rng(seed)
    lb, ub = bench.bounds(dim)
    counter = Counter()
    rec = Recorder(stride, keep_traj, final_gen=max_iter)

    pop = init_population(rng, pop_size, lb, ub)
    fit = evaluate_ok(bench, pop, counter)
    t0 = time.perf_counter()
    best_idx = int(np.argmin(fit))
    best_x, best_f = pop[best_idx].copy(), float(fit[best_idx])
    rec.record(0, best_f, float(fit.mean()), pop, best_x)

    for gen in range(1, max_iter + 1):
        # 1) 选择
        sel = _tournament(rng, fit)
        parents = pop[sel]
        # 2) 交叉（配对处理）
        order = rng.permutation(pop_size)
        c1, c2 = _sbx(rng, parents[order[:pop_size // 2]], parents[order[pop_size // 2:]],
                      lb, ub, eta_c, pc)
        children = np.vstack([c1, c2])[:pop_size]
        # 3) 变异
        children = _poly_mutation(rng, children, lb, ub, eta_m, pm)
        cfit = evaluate_ok(bench, children, counter)

        # 4) 精英保留：父代最优 elite 个直接进入下一代，其余名额由子代竞争
        elite_idx = np.argsort(fit)[:elite]
        child_keep = np.argsort(cfit)[:pop_size - elite]
        pop = np.vstack([pop[elite_idx], children[child_keep]])
        fit = np.concatenate([fit[elite_idx], cfit[child_keep]])

        best_idx = int(np.argmin(fit))
        if fit[best_idx] < best_f:
            best_f, best_x = float(fit[best_idx]), pop[best_idx].copy()
        rec.record(gen, best_f, float(fit.mean()), pop, best_x)

    return build_result(bench, dim, best_x, best_f, rec, counter,
                        time.perf_counter() - t0)
