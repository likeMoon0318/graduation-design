# -*- coding: utf-8 -*-
# 模拟退火算法 SA（多链并行版本）
# =====================================================================
# 经典 SA 是单点串行算法，直接串行执行会带来两个问题：
#   1) 单次运行的评价次数远低于群体算法，横向对比不公平；
#   2) 每条链的随机性方差很大，30 次独立运行仍不稳定。
# 因此这里采用「多链并行 SA」：同时在 pop_size 条独立马尔可夫链上退火，
# 每代每条链做一次邻域扰动 -> 每代评价次数与群体算法完全一致（pop_size 次）。
# 温度按几何降温：T(k) = T0 * (T1/T0)^(k/K)，
# 邻域半径随温度收缩，接受准则使用 Metropolis 判据。

import time

import numpy as np

from .base import Counter, Recorder, build_result, clip, evaluate_ok, init_population


def run(bench, dim, max_iter=200, pop_size=40, seed=1, stride=1, keep_traj=False,
        T1_ratio=1e-3, step_ratio=0.05, cool_power=0.3, accept0=0.80):
    """执行一次多链并行模拟退火寻优。"""
    rng = np.random.default_rng(seed)
    lb, ub = bench.bounds(dim)
    span = ub - lb
    counter = Counter()
    rec = Recorder(stride, keep_traj, final_gen=max_iter)

    X = init_population(rng, pop_size, lb, ub)
    fX = evaluate_ok(bench, X, counter)
    bi = int(np.argmin(fX))
    best_x, best_f = X[bi].copy(), float(fX[bi])
    rec.record(0, best_f, float(fX.mean()), X, best_x)

    # 初始温度自适应标定：先用一次试探性邻域扰动估计目标函数的变化尺度，
    # 再取 T0 使“平均幅度的劣化解”以 accept0 的概率被接受（此处 accept0=0.8），
    # 这样不同函数、不同维度的量纲差异不会影响退火行为。
    probe = clip(X + rng.normal(0.0, 1.0, X.shape) * step_ratio * span, lb, ub)
    fP = evaluate_ok(bench, probe, counter)
    dE0 = float(np.mean(np.abs(fP - fX)))
    T0 = max(dE0 / np.log(1.0 / (1.0 - accept0)), 1e-9)
    T1 = max(T0 * T1_ratio, 1e-12)

    t0 = time.perf_counter()
    for gen in range(1, max_iter + 1):
        T = T0 * (T1 / T0) ** (gen / max_iter)
        # 邻域半径随温度收缩；指数小于 1 使搜索半径衰减得比温度慢，
        # 避免温度快速下降后链被“冻住”而无法在后期做精细搜索
        radius = step_ratio * span * (T / T0) ** cool_power
        cand = clip(X + rng.normal(0.0, 1.0, X.shape) * radius, lb, ub)
        fC = evaluate_ok(bench, cand, counter)

        dE = fC - fX
        accept = (dE < 0) | (rng.random(pop_size) < np.exp(-np.clip(dE, 0, None) / T))
        X[accept], fX[accept] = cand[accept], fC[accept]

        bi = int(np.argmin(fX))
        if fX[bi] < best_f:
            best_f, best_x = float(fX[bi]), X[bi].copy()
        rec.record(gen, best_f, float(fX.mean()), X, best_x)

    return build_result(bench, dim, best_x, best_f, rec, counter,
                        time.perf_counter() - t0,
                        extra={"T0": float(T0), "T1": float(T1)})
