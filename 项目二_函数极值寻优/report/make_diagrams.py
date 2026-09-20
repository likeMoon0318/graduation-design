# -*- coding: utf-8 -*-
# 生成报告用的流程示意图（实验流程图、算法统一接口示意图）
# 运行： python report/make_diagrams.py

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_cache_report2"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import PALETTE, setup_chinese_font  # noqa: E402

FONT = setup_chinese_font()
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, text, fc, ec, fs=9.5, tc="white", bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=3, fontweight="bold" if bold else "normal", linespacing=1.5)


def arrow(ax, p1, p2, color="#6B7A8C", lw=1.3, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=lw, color=color, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11.4, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    main, accent, green, red = (PALETTE["main"], PALETTE["accent"],
                                PALETTE["accent2"], PALETTE["accent3"])

    box(ax, 0.03, 0.79, 0.42, 0.135,
        "测试函数库（12 个）\n单峰 4 个 + 多峰 5 个 + 低维多峰 3 个", accent, accent)
    box(ax, 0.55, 0.79, 0.42, 0.135,
        "算法库（8 种）\nGA / PSO / APSO / DE / GWO / SA / BFGS / NM", accent, accent)

    box(ax, 0.20, 0.585, 0.60, 0.115,
        "批量实验：维度 {2, 10, 30} x 30 次独立运行（种子 1..30）\n"
        "统一评价预算 300 代 x 50 个体", main, main, fs=10)
    arrow(ax, (0.24, 0.79), (0.34, 0.70))
    arrow(ax, (0.76, 0.79), (0.66, 0.70))

    box(ax, 0.03, 0.40, 0.44, 0.115,
        "逐代记录：最优值 / 群体均值 / 耗时\n-> runs_history.csv（62.6 万行）",
        green, green, fs=9.5)
    box(ax, 0.53, 0.40, 0.44, 0.115,
        "最终结果：最优解 / 误差 / 收敛代数 / 成功标记\n-> runs_final.csv（7,920 行）",
        green, green, fs=9.5)
    arrow(ax, (0.35, 0.585), (0.25, 0.515))
    arrow(ax, (0.65, 0.585), (0.75, 0.515))

    box(ax, 0.03, 0.215, 0.20, 0.115, "汇总统计\n最优/均值/标准差\n成功率/收敛代数", main, main, fs=9)
    box(ax, 0.26, 0.215, 0.20, 0.115, "Friedman 检验\nNemenyi 临界差值", main, main, fs=9)
    box(ax, 0.49, 0.215, 0.20, 0.115, "Wilcoxon 两两检验\nHolm 多重比较校正", main, main, fs=9)
    box(ax, 0.72, 0.215, 0.25, 0.115, "参数敏感性实验\n（交叉率/变异率/权重）", main, main, fs=9)
    for x in (0.13, 0.36, 0.59):
        arrow(ax, (x, 0.40), (x, 0.33))

    box(ax, 0.20, 0.045, 0.60, 0.10,
        "可视化（17 张图）+ 结论\n平均排名图 / 临界差异图 / 收敛曲线 / 热力图 / 雷达图",
        red, red, fs=10)
    for x in (0.13, 0.36, 0.59, 0.84):
        arrow(ax, (x, 0.215), (x, 0.145) if x in (0.13, 0.84) else (x, 0.145))
    ax.plot([0.13, 0.84], [0.145, 0.145], color="#6B7A8C", lw=1.3, zorder=1)
    arrow(ax, (0.5, 0.145), (0.5, 0.145), lw=0.1)

    ax.set_title("群智能优化算法函数极值寻优系统 实验流程", fontsize=14, pad=10)
    fig.tight_layout()
    path = OUT / "实验流程图.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_interface():
    fig, ax = plt.subplots(figsize=(11.4, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    main, green, red = PALETTE["main"], PALETTE["accent2"], PALETTE["accent3"]

    box(ax, 0.02, 0.55, 0.20, 0.34,
        "统一调用入口\nrun(bench, dim, max_iter,\npop_size, seed, **params)",
        main, main, fs=9)
    items = [("GA 遗传算法", "SBX 交叉 + 多项式变异 + 精英保留"),
             ("PSO / APSO 粒子群", "速度更新 + 递减惯性权重"),
             ("DE 差分进化", "DE/rand/1/bin + 贪婪选择"),
             ("GWO 灰狼优化", "三级头狼引导 + 收敛因子衰减"),
             ("SA 模拟退火", "多链并行 + 自适应温度标定"),
             ("BFGS / Nelder-Mead", "scipy 传统数学规划方法")]
    y = 0.86
    for name, desc in items:
        box(ax, 0.30, y - 0.11, 0.66, 0.10, f"{name}：{desc}", green, green,
            fs=9.5, tc="white")
        y -= 0.145
    box(ax, 0.02, 0.06, 0.94, 0.21,
        "统一输出：最优解向量 / 最优值 / 误差 / 是否达标 / 收敛代数 / 评价次数 / 耗时 / 逐代收敛曲线",
        red, red, fs=10)
    arrow(ax, (0.22, 0.72), (0.30, 0.72))
    arrow(ax, (0.63, 0.55), (0.63, 0.28), lw=1.4)

    ax.set_title("算法统一接口设计", fontsize=14, pad=10)
    fig.tight_layout()
    path = OUT / "算法统一接口.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main():
    for fn in (fig_pipeline, fig_interface):
        print("[生成]", fn().relative_to(ROOT))


if __name__ == "__main__":
    main()
