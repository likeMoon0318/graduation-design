# -*- coding: utf-8 -*-
# 标准测试函数库（12 个）：全部为向量化实现，可一次评估整个种群
# =====================================================================
# 约定：输入 X 形状为 (n, dim) 或 (dim,)，返回形状为 (n,) 或标量
#       fmin 为理论全局最优值，xmin 为对应的一个全局最优解

from dataclasses import dataclass, field

import numpy as np


def _as2d(X):
    X = np.asarray(X, dtype=float)
    return X.reshape(1, -1) if X.ndim == 1 else X


# ---------------------------------------------------------------- 单峰函数

def sphere(X):
    X = _as2d(X)
    return np.sum(X ** 2, axis=1)


def schwefel222(X):
    X = _as2d(X)
    return np.sum(np.abs(X), axis=1) + np.prod(np.abs(X), axis=1)


def rosenbrock(X):
    X = _as2d(X)
    return np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (X[:, :-1] - 1.0) ** 2, axis=1)


def dixon_price(X):
    X = _as2d(X)
    n, d = X.shape
    i = np.arange(2, d + 1)
    term = i * (2.0 * X[:, 1:] ** 2 - X[:, :-1]) ** 2
    return (X[:, 0] - 1.0) ** 2 + np.sum(term, axis=1)


# ---------------------------------------------------------------- 多峰函数

def rastrigin(X):
    X = _as2d(X)
    return 10.0 * X.shape[1] + np.sum(X ** 2 - 10.0 * np.cos(2 * np.pi * X), axis=1)


def ackley(X):
    X = _as2d(X)
    a = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(X ** 2, axis=1)))
    b = -np.exp(np.mean(np.cos(2 * np.pi * X), axis=1))
    return a + b + 20.0 + np.e


def griewank(X):
    X = _as2d(X)
    i = np.arange(1, X.shape[1] + 1)
    return 1.0 + np.sum(X ** 2, axis=1) / 4000.0 - np.prod(np.cos(X / np.sqrt(i)), axis=1)


def levy(X):
    X = _as2d(X)
    w = 1.0 + (X - 1.0) / 4.0
    term1 = np.sin(np.pi * w[:, 0]) ** 2
    term3 = (w[:, -1] - 1.0) ** 2 * (1.0 + np.sin(2 * np.pi * w[:, -1]) ** 2)
    mid = np.sum((w[:, :-1] - 1.0) ** 2 *
                 (1.0 + 10.0 * np.sin(np.pi * w[:, :-1] + 1.0) ** 2), axis=1)
    return term1 + mid + term3


def schwefel226(X):
    X = _as2d(X)
    return 418.9828872724338 * X.shape[1] - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)


# ---------------------------------------------------------------- 低维多峰（仅 2 维，便于可视化）

def himmelblau(X):
    X = _as2d(X)
    x, y = X[:, 0], X[:, 1]
    return (x ** 2 + y - 11.0) ** 2 + (x + y ** 2 - 7.0) ** 2


def beale(X):
    X = _as2d(X)
    x, y = X[:, 0], X[:, 1]
    return ((1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2
            + (2.625 - x + x * y ** 3) ** 2)


def six_hump_camel(X):
    X = _as2d(X)
    x, y = X[:, 0], X[:, 1]
    return ((4.0 - 2.1 * x ** 2 + x ** 4 / 3.0) * x ** 2 + x * y
            + (-4.0 + 4.0 * y ** 2) * y ** 2)


# ---------------------------------------------------------------- 注册表

@dataclass(frozen=True)
class Benchmark:
    key: str                 # 英文键（数据文件、代码使用）
    cn: str                  # 中文名（图表、报告使用）
    category: str            # 单峰 / 多峰 / 低维多峰
    fn: object               # 目标函数
    lb: float                # 下界（标量，代表各维相同）
    ub: float                # 上界
    fmin: float              # 理论最优值
    xmin: tuple              # 理论最优解（一个代表点）
    tol: float = 1e-2        # 成功判据阈值（|f - f*| < tol）
    dims: tuple = (2, 10, 30)
    lb_vec: tuple = field(default=None)
    ub_vec: tuple = field(default=None)
    note: str = ""

    def bounds(self, dim):
        """返回该维度下的 (lb, ub) 数组。低维函数支持逐维不同界。"""
        lb = np.full(dim, self.lb, dtype=float) if self.lb_vec is None \
            else np.array(self.lb_vec[:dim], dtype=float)
        ub = np.full(dim, self.ub, dtype=float) if self.ub_vec is None \
            else np.array(self.ub_vec[:dim], dtype=float)
        return lb, ub

    def error(self, value):
        """与理论最优值的偏差（针对最小化问题）。"""
        return np.abs(np.asarray(value, dtype=float) - self.fmin)


BENCHMARKS = {
    "sphere": Benchmark(
        "sphere", "Sphere", "单峰", sphere, -100.0, 100.0, 0.0, (0.0,), 1e-6,
        note="最经典的球函数，用于检验收敛速度"),
    "schwefel222": Benchmark(
        "schwefel222", "Schwefel 2.22", "单峰", schwefel222, -10.0, 10.0, 0.0, (0.0,), 1e-2,
        note="绝对值之和与乘积之和，变量间存在耦合"),
    "rosenbrock": Benchmark(
        "rosenbrock", "Rosenbrock", "单峰", rosenbrock, -30.0, 30.0, 0.0, (1.0,), 1e-2,
        note="香蕉形狭长谷底，传统梯度法在谷底推进极慢"),
    "dixon_price": Benchmark(
        "dixon_price", "Dixon-Price", "单峰", dixon_price, -10.0, 10.0, 0.0,
        (1.0, 0.7071067811865476), 1e-2,
        note="非可分单峰函数，对坐标旋转敏感"),
    "rastrigin": Benchmark(
        "rastrigin", "Rastrigin", "多峰", rastrigin, -5.12, 5.12, 0.0, (0.0,), 1e-2,
        note="大量规则分布的局部极小点，检验全局搜索能力"),
    "ackley": Benchmark(
        "ackley", "Ackley", "多峰", ackley, -32.0, 32.0, 0.0, (0.0,), 1e-2,
        note="平坦外围 + 中心深坑，容易早熟停滞"),
    "griewank": Benchmark(
        "griewank", "Griewank", "多峰", griewank, -600.0, 600.0, 0.0, (0.0,), 1e-2,
        note="乘积项随维度衰减，高维下局部陷阱变浅"),
    "levy": Benchmark(
        "levy", "Levy", "多峰", levy, -10.0, 10.0, 0.0, (1.0,), 1e-2,
        note="近最优区域极其狭窄，考验收敛精度"),
    "schwefel226": Benchmark(
        "schwefel226", "Schwefel 2.26", "多峰", schwefel226, -500.0, 500.0, 0.0,
        (420.9687,), 1e-2,
        note="最优解位于边界附近，且离次优极小点极远，最易迷惑算法"),
    "himmelblau": Benchmark(
        "himmelblau", "Himmelblau", "低维多峰", himmelblau, -5.0, 5.0, 0.0,
        (3.0, 2.0), 1e-2, dims=(2,),
        note="2 维，有 4 个完全对称的全局极小点，用于观察算法能否反复命中"),
    "beale": Benchmark(
        "beale", "Beale", "低维多峰", beale, -4.5, 4.5, 0.0, (3.0, 0.5), 1e-2,
        dims=(2,), note="2 维，边界区域平坦，梯度法容易停滞"),
    "six_hump": Benchmark(
        "six_hump", "Six-Hump Camel", "低维多峰", six_hump_camel, -3.0, 3.0,
        -1.031628453489877, (0.0898, -0.7126), 1e-2, dims=(2,),
        lb_vec=(-3.0, -2.0), ub_vec=(3.0, 2.0),
        note="2 维，6 个局部极小点中仅 2 个是全局最优"),
}

# 函数顺序（报告中固定使用该顺序）
FUNC_KEYS = list(BENCHMARKS)
FUNC_CN = {k: v.cn for k, v in BENCHMARKS.items()}


def functions_for_dim(dim):
    """返回指定维度下可用的测试函数键列表。"""
    return [k for k, b in BENCHMARKS.items() if dim in b.dims]


def evaluate(key, X):
    """按函数键求值。"""
    return BENCHMARKS[key].fn(X)


def grid_surface(key, n=120, pad=0.0):
    """生成 2 维函数的网格曲面数据，供等高线与 3D 曲面绘制使用。"""
    b = BENCHMARKS[key]
    lb, ub = b.bounds(2)
    lb = lb - pad * (ub - lb)
    ub = ub + pad * (ub - lb)
    xs = np.linspace(lb[0], ub[0], n)
    ys = np.linspace(lb[1], ub[1], n)
    XX, YY = np.meshgrid(xs, ys)
    Z = b.fn(np.column_stack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    return XX, YY, Z


def main():
    """自检：打印 12 个测试函数的理论最优值与在最优解处的取值。"""
    print(f"{'函数':<16}{'类型':<8}{'维度':<14}{'理论最优':>12}{'数值校验':>14}")
    print("-" * 68)
    for key, b in BENCHMARKS.items():
        x = np.array(b.xmin, dtype=float)
        if len(x) < 2:
            x = np.repeat(x, 2)
        val = float(b.fn(x))
        print(f"{b.cn:<16}{b.category:<8}{str(b.dims):<14}{b.fmin:>12.6f}{val:>14.6f}")
    print("\n[提示] 数值校验列应等于理论最优值（浮点误差范围内）。")


if __name__ == "__main__":
    main()
