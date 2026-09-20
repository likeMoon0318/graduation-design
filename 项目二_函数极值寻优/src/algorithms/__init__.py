# -*- coding: utf-8 -*-
# 算法注册表：统一暴露 8 种主算法 + 2 种基线方法

from . import de, ga, gwo, pso, sa, traditional

# 主实验使用的 8 种算法（覆盖 基线 / 群体智能 / 扩展改进 三个层次）
ALGORITHMS = {
    "GA": ga.run,                       # 遗传算法
    "PSO": pso.run,                     # 粒子群算法
    "APSO": pso.run_apso,               # 自适应权重粒子群
    "DE": de.run,                       # 差分进化
    "GWO": gwo.run,                     # 灰狼优化
    "SA": sa.run,                       # 多链并行模拟退火
    "BFGS": traditional.run_bfgs,       # 拟牛顿法（带边界）
    "NM": traditional.run_nelder_mead,  # 单纯形法
}

# 基线方法（网格搜索仅支持 2 维，随机搜索任意维）
BASELINES = {
    "GRID": traditional.run_grid,
    "RANDOM": traditional.run_random,
}

# 各算法默认参数（供演示器界面初始化与参数敏感性实验使用）
DEFAULT_PARAMS = {
    "GA": {"pc": 0.9, "pm": None, "eta_c": 15.0, "eta_m": 20.0, "elite": 2},
    "PSO": {"w": 0.729, "c1": 1.49445, "c2": 1.49445},
    "APSO": {"w0": 0.9, "w1": 0.4, "c1_0": 2.5, "c1_1": 0.5, "c2_0": 0.5, "c2_1": 2.5},
    "DE": {"F": 0.5, "CR": 0.9, "strategy": "rand1"},
    "GWO": {},
    "SA": {"T1_ratio": 1e-3, "step_ratio": 0.05, "cool_power": 0.3, "accept0": 0.80},
    "BFGS": {},
    "NM": {},
}

# 算法分层说明，供报告与界面展示
ALGO_LAYERS = {
    "GA": "群体智能（演化计算）",
    "PSO": "群体智能（群智能）",
    "APSO": "群体智能（改进策略）",
    "DE": "群体智能（演化计算）",
    "GWO": "群体智能（群智能）",
    "SA": "邻域搜索（概率接受准则）",
    "BFGS": "传统数学规划（一阶梯度）",
    "NM": "传统数学规划（直接搜索）",
}

__all__ = ["ALGORITHMS", "BASELINES", "DEFAULT_PARAMS", "ALGO_LAYERS",
           "ga", "pso", "sa", "de", "gwo", "traditional"]
