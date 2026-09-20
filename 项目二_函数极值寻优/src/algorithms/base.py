# -*- coding: utf-8 -*-
# 算法公共基础设施：统一接口、评价计数器、历史记录器
# =====================================================================
# 所有算法遵循同一接口：
#     run(bench, dim, max_iter, pop_size, seed, **params) -> dict
# 返回字段：
#     best_x / best_value / history / n_evals / elapsed / extra

import time

import numpy as np


class Counter:
    """函数评价次数计数器：所有算法共用，用于统一评价预算口径。"""

    __slots__ = ("n",)

    def __init__(self):
        self.n = 0

    def add(self, k):
        self.n += int(k)
        return self.n


class Recorder:
    """按代记录收敛过程。

    - best_full  : 每一代的最优值（不抽样，用于精确判定收敛代数）
    - rows       : 按 stride 抽样后的 (代数, 最优值, 群体均值)，用于落盘与绘图
    - traj       : 可选的最优位置与群体快照，供演示器动画使用
    """

    def __init__(self, stride=1, keep_traj=False, final_gen=None):
        self.stride = max(1, int(stride))
        self.keep_traj = keep_traj
        self.final_gen = final_gen
        self.best_full = []
        self.rows = []
        self.traj = []
        self._last_mean = np.inf
        self.last_gen = 0

    def record(self, gen, best_value, mean_value, population=None, best_x=None):
        self.last_gen = int(gen)
        self._last_mean = float(mean_value)
        self.best_full.append((int(gen), float(best_value)))
        if gen % self.stride == 0 or gen == self.final_gen:
            self.rows.append((int(gen), float(best_value), float(mean_value)))
        if self.keep_traj and population is not None:
            self.traj.append((gen, np.array(best_x, dtype=float),
                              np.array(population, dtype=float)))

    def arrays(self):
        if not self.rows:
            return np.array([0]), np.array([np.inf]), np.array([np.inf])
        # 保证最后一代一定落在曲线上（抽样步长可能跳过它）
        if self.rows[-1][0] != self.last_gen:
            self.rows.append((self.last_gen, self.best_full[-1][1], self._last_mean))
        arr = np.array(self.rows, dtype=float)
        return arr[:, 0].astype(int), arr[:, 1], arr[:, 2]

    def convergence_gen(self, fmin, tol):
        """首个使 |f - f*| < tol 的代数（按逐代最优值精确判定）。"""
        for gen, val in self.best_full:
            if abs(val - fmin) < tol:
                return int(gen)
        return None


def init_population(rng, pop_size, lb, ub):
    """在边界内均匀随机初始化种群。"""
    dim = len(lb)
    return lb + rng.random((pop_size, dim)) * (ub - lb)


def evaluate_ok(bench, X, counter):
    """统一求值入口：自动累加评价次数。"""
    X = np.atleast_2d(X)
    counter.add(len(X))
    return np.asarray(bench.fn(X), dtype=float).ravel()


def clip(X, lb, ub):
    """边界裁剪（越界修复策略：截断到边界）。"""
    return np.clip(X, lb, ub)


def build_result(bench, dim, best_x, best_value, recorder, counter, elapsed, extra=None):
    """打包算法输出，统一计算误差与收敛代数。"""
    gen, best_curve, mean_curve = recorder.arrays()
    err = float(bench.error(best_value))
    conv_gen = recorder.convergence_gen(bench.fmin, bench.tol)
    result = {
        "dim": dim,
        "best_x": np.asarray(best_x, dtype=float),
        "best_value": float(best_value),
        "error": err,
        "success": bool(err < bench.tol),
        "convergence_gen": conv_gen,
        "n_evals": counter.n,
        "elapsed": float(elapsed),
        "history_gen": gen,
        "history_best": best_curve,
        "history_mean": mean_curve,
        "extra": extra or {},
    }
    if recorder.keep_traj:
        result["trajectory"] = recorder.traj
    return result


def timeit(fn):
    """小工具：返回 (结果, 耗时秒)。"""
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


__all__ = [
    "Counter", "Recorder", "init_population", "evaluate_ok", "clip",
    "build_result", "timeit",
]
