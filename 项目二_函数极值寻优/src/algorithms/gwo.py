# -*- coding: utf-8 -*-
# 灰狼优化算法 GWO
# =====================================================================
# 模拟灰狼社会等级：alpha（最优）、beta（次优）、delta（第三）共同引导
# 其余个体（omega）向三者加权靠近，收敛因子 a 由 2 线性衰减到 0

import time

import numpy as np

from .base import Counter, Recorder, build_result, clip, evaluate_ok, init_population


def run(bench, dim, max_iter=200, pop_size=40, seed=1, stride=1, keep_traj=False):
    """执行一次灰狼优化寻优。"""
    rng = np.random.default_rng(seed)
    lb, ub = bench.bounds(dim)
    counter = Counter()
    rec = Recorder(stride, keep_traj, final_gen=max_iter)

    pop = init_population(rng, pop_size, lb, ub)
    fit = evaluate_ok(bench, pop, counter)
    order = np.argsort(fit)
    alpha, alpha_f = pop[order[0]].copy(), float(fit[order[0]])
    beta, beta_f = pop[order[1]].copy(), float(fit[order[1]])
    delta, delta_f = pop[order[2]].copy(), float(fit[order[2]])
    rec.record(0, alpha_f, float(fit.mean()), pop, alpha)

    t0 = time.perf_counter()
    for gen in range(1, max_iter + 1):
        a = 2.0 - 2.0 * gen / max_iter            # 收敛因子线性衰减
        # 三只头狼（alpha / beta / delta）分别给出位置建议，取算术平均作为新位置
        cand_sum = None
        for leader in (alpha, beta, delta):
            r1 = rng.random((pop_size, dim))
            r2 = rng.random((pop_size, dim))
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            D = np.abs(C * leader - pop)
            cand = leader - A * D
            cand_sum = cand if cand_sum is None else cand_sum + cand
        new_pop = cand_sum / 3.0
        pop = clip(new_pop, lb, ub)
        fit = evaluate_ok(bench, pop, counter)

        order = np.argsort(fit)
        alpha, alpha_f = pop[order[0]].copy(), float(fit[order[0]])
        beta, beta_f = pop[order[1]].copy(), float(fit[order[1]])
        delta, delta_f = pop[order[2]].copy(), float(fit[order[2]])
        rec.record(gen, alpha_f, float(fit.mean()), pop, alpha)

    return build_result(bench, dim, alpha, alpha_f, rec, counter,
                        time.perf_counter() - t0,
                        extra={"beta_value": beta_f, "delta_value": delta_f})
