# -*- coding: utf-8 -*-
# 可视化模块：生成实验报告所需的全部图表
# =====================================================================
# 输出：results/figures/*.png（300dpi，报告插图）+ *.pdf（矢量图）
# 图表编号与报告中的图号一一对应

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import (ALGO_CN, ALGO_COLORS, FIG_DIR, MAIN_ALGOS, PALETTE,  # noqa: E402
                    RUNS_FINAL, RUNS_HISTORY, SUMMARY_FILE, TABLE_DIR, save_figure,
                    setup_chinese_font)
from algorithms import DEFAULT_PARAMS  # noqa: E402
from functions import BENCHMARKS, grid_surface  # noqa: E402

FONT = setup_chinese_font()
sns.set_style("whitegrid", {"font.sans-serif": [FONT], "axes.unicode_minus": False})

EPS = 1e-12
DIM_LABEL = {2: "2 维", 10: "10 维", 30: "30 维"}

# 报告正文中重点讨论的代表性函数
REP_FUNCS = ["sphere", "rosenbrock", "rastrigin", "ackley", "schwefel226", "levy"]


def err(x):
    """误差取正下界，避免 log10(0) 报错。"""
    return np.clip(np.asarray(x, dtype=float), EPS, None)


def load_all():
    runs = pd.read_csv(RUNS_FINAL)
    summary = pd.read_csv(SUMMARY_FILE)
    hist = pd.read_csv(RUNS_HISTORY)
    return runs, summary, hist


def algo_color(a):
    return ALGO_COLORS.get(a, PALETTE["neutral"])


def set_log_axis(ax, axis="y"):
    """把指定坐标轴设为对数刻度，并使用 ASCII 科学计数标签。

    matplotlib 默认的对数刻度标签走 mathtext，会把负号渲染成 U+2212；
    本机中文字体缺少该字形，会出现占位方块，因此这里改用纯 ASCII 标签。
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    def fmt(v, _pos=None):
        if v <= 0:
            return ""
        e = np.log10(v)
        if abs(e - round(e)) > 1e-9:
            return ""
        return f"1e{int(round(e))}"

    target = ax.yaxis if axis == "y" else ax.xaxis
    if axis == "y":
        ax.set_yscale("log")
    else:
        ax.set_xscale("log")
    target.set_major_locator(LogLocator(base=10.0))
    target.set_major_formatter(FuncFormatter(fmt))
    target.set_minor_formatter(NullFormatter())
    return ax


# =====================================================================
# 图 1：测试函数 3D 曲面（标注理论全局最优）
# =====================================================================

def fig01_function_surfaces():
    keys = ["rastrigin", "ackley", "schwefel226", "himmelblau", "beale", "six_hump"]
    fig = plt.figure(figsize=(14.5, 8.6))
    for i, key in enumerate(keys, 1):
        b = BENCHMARKS[key]
        XX, YY, ZZ = grid_surface(key, n=90)
        ax = fig.add_subplot(2, 3, i, projection="3d")
        ax.plot_surface(XX, YY, ZZ, cmap="viridis", alpha=0.92,
                        rstride=2, cstride=2, linewidth=0)
        xopt = np.asarray(b.xmin, dtype=float)
        if xopt.size == 1:
            xopt = np.repeat(xopt, 2)          # 各维最优值相同的函数
        ax.scatter([xopt[0]], [xopt[1]], [b.fmin], color=PALETTE["accent3"],
                   s=55, depthshade=False, label="理论最优")
        ax.set_title(f"{b.cn}（{b.category}）", fontsize=12, pad=6)
        ax.set_xlabel("x1", fontsize=9)
        ax.set_ylabel("x2", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.view_init(elev=32, azim=-58)
        ax.grid(False)
    fig.suptitle("六个代表性测试函数的搜索空间与理论全局最优位置", fontsize=15, y=0.98)
    return save_figure(fig, "图01_测试函数3D曲面")


# =====================================================================
# 图 2-3：平均收敛曲线（含标准差阴影）
# =====================================================================

def _convergence_panel(hist, dim, keys, fig, axes):
    sub = hist[hist["dim"] == dim]
    for ax, key in zip(axes, keys):
        b = BENCHMARKS[key]
        d = sub[sub["func"] == key]
        for algo in MAIN_ALGOS + [x for x in ["GRID", "RANDOM"] if x in set(sub["algo"])]:
            g = d[d["algo"] == algo].groupby("generation")["best_error"]
            if g.ngroups == 0:
                continue
            m = g.mean()
            s = g.std().fillna(0)
            xs = m.index.values
            mv = err(m.values)
            sv = err(s.values)
            ax.plot(xs, mv, color=algo_color(algo), lw=1.7, label=algo)
            ax.fill_between(xs, mv, mv + sv, color=algo_color(algo), alpha=0.12, lw=0)
        set_log_axis(ax, "y")
        ax.set_title(f"{b.cn}（{b.category}）", fontsize=11)
        ax.set_xlabel("迭代代数", fontsize=9)
        ax.set_ylabel("平均最优误差（对数刻度）", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.axhline(b.tol, color=PALETTE["neutral"], ls=":", lw=1)
    return axes


def fig02_convergence_2d(hist):
    keys = ["sphere", "rosenbrock", "rastrigin", "ackley", "schwefel226", "himmelblau"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    _convergence_panel(hist, 2, keys, fig, axes.ravel())
    handles = [plt.Line2D([], [], color=algo_color(a), lw=2.4,
                          label=f"{a}（{ALGO_CN.get(a, a)}）")
               for a in MAIN_ALGOS + ["GRID", "RANDOM"]]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("2 维问题上的平均收敛曲线（30 次独立运行的均值，虚线为成功判据）",
                 fontsize=15, y=0.99)
    fig.tight_layout(rect=(0, 0.045, 1, 0.965))
    return save_figure(fig, "图02_平均收敛曲线_2维")


def fig03_convergence_10d(hist):
    keys = ["sphere", "rosenbrock", "rastrigin", "ackley", "schwefel226", "levy"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    _convergence_panel(hist, 10, keys, fig, axes.ravel())
    handles = [plt.Line2D([], [], color=algo_color(a), lw=2.4, label=a) for a in MAIN_ALGOS]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("10 维主实验：六个代表性函数上的平均收敛曲线", fontsize=15, y=0.99)
    fig.tight_layout(rect=(0, 0.045, 1, 0.965))
    return save_figure(fig, "图03_平均收敛曲线_10维")


# =====================================================================
# 图 4-5：误差分布箱线图与小提琴图
# =====================================================================

def fig04_boxplots(runs):
    keys = ["sphere", "rosenbrock", "rastrigin", "ackley"]
    sub = runs[(runs["dim"] == 10) & (runs["func"].isin(keys))]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.8), sharey=True)
    for ax, key in zip(axes, keys):
        d = sub[sub["func"] == key]
        data, colors, labels = [], [], []
        for algo in MAIN_ALGOS:
            v = d[d["algo"] == algo]["error"].values
            if len(v):
                data.append(np.log10(err(v)))
                colors.append(algo_color(algo))
                labels.append(algo)
        bp = ax.boxplot(data, patch_artist=True, showfliers=True, widths=0.62,
                        flierprops=dict(marker="o", markersize=3, alpha=0.45))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
        for med in bp["medians"]:
            med.set_color("white")
            med.set_linewidth(1.4)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(f"{BENCHMARKS[key].cn}", fontsize=12)
        ax.axhline(np.log10(1e-2), color=PALETTE["neutral"], ls=":", lw=1)
    axes[0].set_ylabel("log10(误差)", fontsize=10)
    fig.suptitle("10 维主实验：四种代表性函数上 30 次独立运行的误差分布"
                 "（虚线为成功判据 1e-2）", fontsize=14, y=1.0)
    fig.tight_layout()
    return save_figure(fig, "图04_误差分布箱线图")


def fig05_violin(runs):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    for ax, dim in zip(axes, [10, 30]):
        d = runs[runs["dim"] == dim]
        data, labels, colors = [], [], []
        for algo in MAIN_ALGOS:
            v = d[d["algo"] == algo]["error"].values
            if len(v):
                data.append(np.log10(err(v)))
                labels.append(algo)
                colors.append(algo_color(algo))
        parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.8)
        for body, c in zip(parts["bodies"], colors):
            body.set_facecolor(c)
            body.set_alpha(0.55)
            body.set_edgecolor(c)
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_color("#4A4A4A")
                parts[key].set_linewidth(1.0)
        for i, v in enumerate(data, 1):
            jitter = (np.random.default_rng(i).random(len(v)) - 0.5) * 0.16
            ax.scatter(np.full(len(v), i) + jitter, v, s=3, color=colors[i - 1], alpha=0.25)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("log10(误差)", fontsize=10)
        ax.set_title(f"{DIM_LABEL[dim]}：全部函数合并后的误差分布", fontsize=12)
        ax.axhline(np.log10(1e-2), color=PALETTE["neutral"], ls=":", lw=1)
    fig.tight_layout()
    return save_figure(fig, "图05_误差分布小提琴图")


# =====================================================================
# 图 6-7：平均排名与 Nemenyi 临界差异图
# =====================================================================

def _average_ranks(runs):
    """返回 {维度: Series(算法 -> 平均排名)}。"""
    out = {}
    for dim in sorted(runs["dim"].unique()):
        blocks = (runs[runs["dim"] == dim]
                  .pivot_table(index=["func", "run"], columns="algo",
                               values="error", aggfunc="mean")
                  .dropna(axis=0, how="any"))
        out[dim] = blocks.rank(axis=1, method="average").mean()
    return out


def fig06_average_ranks(runs):
    ranks = _average_ranks(runs)
    dims = sorted(ranks)
    algos = MAIN_ALGOS
    x = np.arange(len(algos))
    width = 0.8 / len(dims)
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    shades = [PALETTE["main"], PALETTE["accent"], PALETTE["accent2"], PALETTE["accent3"]]
    for i, dim in enumerate(dims):
        vals = [ranks[dim].get(a, np.nan) for a in algos]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width=width * 0.92,
                      color=shades[i % len(shades)], label=DIM_LABEL[dim])
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}",
                        ha="center", fontsize=8, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n{ALGO_CN.get(a, '')}" for a in algos], fontsize=9)
    ax.set_ylabel("平均排名（1 为最优）", fontsize=10)
    ax.invert_yaxis()
    ax.legend(fontsize=10)
    ax.set_title("各算法在不同维度下的平均排名（Friedman 检验的秩统计量）", fontsize=14)
    fig.tight_layout()
    return save_figure(fig, "图06_平均排名")


NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}


def fig07_critical_difference(runs, dim=10, q_alpha=None, ax=None, title=None):
    """Nemenyi 临界差异图：平均排名差小于 CD 的算法之间用横线连接（差异不显著）。"""
    ranks = _average_ranks(runs)[dim]
    blocks = (runs[runs["dim"] == dim]
              .pivot_table(index=["func", "run"], columns="algo", values="error",
                           aggfunc="mean").dropna(axis=0, how="any"))
    N, k = blocks.shape
    q = q_alpha if q_alpha is not None else NEMENYI_Q05.get(k, 3.164)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * N))
    r = ranks.sort_values()

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(11.5, 4.6))
    else:
        fig = ax.figure
    ax.set_xlim(0, k + 0.6)
    ax.set_ylim(-0.6, len(r) * 0.62 + 0.5)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.grid(False)

    y = 0.0
    for i, (algo, rank) in enumerate(r.items()):
        ax.plot([rank], [y], marker="o", ms=9, color=algo_color(algo))
        ax.text(rank, y - 0.22, f"{rank:.2f}", ha="center", fontsize=9, color="#333333")
        side = 1 if i % 2 == 0 else -1
        ax.plot([rank, rank], [y, y + side * 0.24], color=algo_color(algo), lw=1.4)
        ax.text(rank, y + side * 0.30, f"{algo}（{ALGO_CN.get(algo, '')}）",
                ha="center", va="top" if side < 0 else "bottom", fontsize=9,
                color=algo_color(algo))
        y += 0.62

    # 连接差异不显著的算法组
    order = list(r.items())
    used = set()
    for i in range(len(order)):
        grp = [i]
        for j in range(i + 1, len(order)):
            if order[j][1] - order[i][1] <= cd:
                grp.append(j)
        if len(grp) > 1 and not set(grp).issubset(used):
            used.update(grp)
            yline = len(order) * 0.62 - 0.05 * len(used)
            ax.plot([order[grp[0]][1], order[grp[-1]][1]], [yline, yline],
                    color=PALETTE["neutral"], lw=2.0)

    ax.set_xlabel("平均排名（数值越小越好）", fontsize=10)
    ax.text(0.99, 0.02, f"Nemenyi 临界差值 CD = {cd:.3f}（k={k}, N={N}, α=0.05）",
            transform=ax.transAxes, ha="right", fontsize=9, color=PALETTE["neutral"])
    ax.set_title(title or f"{DIM_LABEL[dim]}下的算法排序与 Nemenyi 临界差异图",
                 fontsize=13, pad=12)
    if own_fig:
        fig.tight_layout()
        return save_figure(fig, "图07_临界差异图")
    return fig


# =====================================================================
# 图 8-9：算法 x 函数 热力图（误差 / 成功率）
# =====================================================================

def fig08_error_heatmap(runs):
    sub = runs[runs["dim"] == 10]
    order = list(BENCHMARKS)
    med = (sub.pivot_table(index="algo", columns="func", values="error", aggfunc="median")
           .reindex(index=MAIN_ALGOS, columns=[k for k in order if k in set(sub["func"])]))
    mat = np.log10(err(med.values))
    fig, ax = plt.subplots(figsize=(13.5, 5.4))
    im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(med.shape[1]))
    ax.set_xticklabels([f"{BENCHMARKS[k].cn}\n({BENCHMARKS[k].category})"
                        for k in med.columns], fontsize=9)
    ax.set_yticks(range(med.shape[0]))
    ax.set_yticklabels([f"{a}   {ALGO_CN.get(a, '')}" for a in med.index], fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt = "0" if v <= -10 else f"{v:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if v > 2 or v < -6 else "#333333")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.012)
    cb.set_label("log10(误差中位数)", fontsize=9)
    ax.set_title("10 维主实验：算法 x 函数 误差中位数热力图（数值越小越好）", fontsize=13, pad=12)
    fig.tight_layout()
    return save_figure(fig, "图08_误差热力图")


def fig09_success_heatmap(runs):
    sub = runs[runs["dim"] == 10]
    rate = (sub.pivot_table(index="algo", columns="func", values="success", aggfunc="mean")
            .reindex(index=MAIN_ALGOS, columns=[k for k in BENCHMARKS if k in set(sub["func"])]))
    fig, ax = plt.subplots(figsize=(13.5, 5.0))
    im = ax.imshow(rate.values * 100, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(rate.shape[1]))
    ax.set_xticklabels([BENCHMARKS[k].cn for k in rate.columns], fontsize=9, rotation=25,
                       ha="right")
    ax.set_yticks(range(rate.shape[0]))
    ax.set_yticklabels([f"{a}   {ALGO_CN.get(a, '')}" for a in rate.index], fontsize=9)
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            v = rate.values[i, j] * 100
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 55 else "#333333")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.012)
    cb.set_label("成功率（%）", fontsize=9)
    ax.set_title("10 维主实验：算法 x 函数 成功率热力图（判据 |f - f*| < 1e-2）",
                 fontsize=13, pad=12)
    fig.tight_layout()
    return save_figure(fig, "图09_成功率热力图")


# =====================================================================
# 图 10：参数敏感性
# =====================================================================

def fig10_sensitivity():
    path = TABLE_DIR / "参数敏感性.csv"
    if not path.exists():
        print("[跳过] 参数敏感性.csv 不存在，请先运行 src/sensitivity.py")
        return None
    df = pd.read_csv(path)
    combos = [("GA", "pc"), ("GA", "pm_factor"), ("PSO", "w"), ("PSO", "c1"),
              ("DE", "F"), ("DE", "CR"), ("SA", "step_ratio"), ("SA", "cool_power"),
              ("GWO", "pop_factor")]
    label_map = {"pc": "交叉概率 pc", "pm_factor": "变异概率倍数(×1/d)", "w": "惯性权重 w",
                 "c1": "学习因子 c1", "F": "缩放因子 F", "CR": "交叉概率 CR",
                 "step_ratio": "邻域步长比例", "cool_power": "降温半径指数",
                 "pop_factor": "种群规模倍数"}
    fig, axes = plt.subplots(3, 3, figsize=(15, 10.2))
    for ax, (algo, param) in zip(axes.ravel(), combos):
        d = df[(df["算法"] == algo) & (df["参数"] == param)].sort_values("取值")
        if d.empty:
            ax.axis("off")
            continue
        ax.plot(d["取值"], d["log10平均误差"], marker="o", lw=2,
                color=algo_color(algo), label="log10(平均误差)")
        ax.set_xlabel(label_map.get(param, param), fontsize=9)
        ax.set_ylabel("log10(平均误差)", fontsize=9, color=algo_color(algo))
        ax.tick_params(labelsize=8)
        ax2 = ax.twinx()
        ax2.plot(d["取值"], d["成功率"] * 100, marker="s", ls="--", lw=1.6,
                 color=PALETTE["neutral"])
        ax2.set_ylabel("成功率（%）", fontsize=9, color=PALETTE["neutral"])
        ax2.set_ylim(-3, 103)
        ax2.grid(False)
        # 标注最优取值
        bi = int(np.argmin(d["log10平均误差"].values))
        ax.scatter([d["取值"].values[bi]], [d["log10平均误差"].values[bi]], s=90,
                   facecolor="none", edgecolor=PALETTE["accent3"], lw=1.8, zorder=5)
        ax.set_title(f"{algo} —— {label_map.get(param, param)}", fontsize=11)
    fig.suptitle("参数敏感性分析（10 维，4 个代表性函数 x 10 次重复；"
                 "红圈标记该参数的最优取值）", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return save_figure(fig, "图10_参数敏感性")


# =====================================================================
# 图 11：维度-性能衰减
# =====================================================================

def fig11_dimension_scaling(runs):
    ranks = _average_ranks(runs)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # (a) 平均排名随维度的变化
    ax = axes[0]
    for algo in MAIN_ALGOS:
        vals = [ranks[d].get(algo, np.nan) for d in sorted(ranks)]
        ax.plot(sorted(ranks), vals, marker="o", lw=2, color=algo_color(algo), label=algo)
    ax.set_xticks(sorted(ranks))
    ax.set_xticklabels([DIM_LABEL[d] for d in sorted(ranks)])
    ax.set_ylabel("平均排名（越小越好）", fontsize=10)
    ax.invert_yaxis()
    ax.set_title("(a) 平均排名随维度的变化", fontsize=11)

    # (b) 成功率随维度的变化
    ax = axes[1]
    for algo in MAIN_ALGOS:
        vals = [runs[(runs["dim"] == d) & (runs["algo"] == algo)]["success"].mean() * 100
                for d in sorted(ranks)]
        ax.plot(sorted(ranks), vals, marker="s", lw=2, color=algo_color(algo))
    ax.set_xticks(sorted(ranks))
    ax.set_xticklabels([DIM_LABEL[d] for d in sorted(ranks)])
    ax.set_ylabel("成功率（%）", fontsize=10)
    ax.set_title("(b) 成功率随维度的变化", fontsize=11)

    # (c) 相对 2 维的误差放大倍数（几何平均）
    ax = axes[2]
    valid = [k for k, b in BENCHMARKS.items() if len(b.dims) == 3]
    for algo in MAIN_ALGOS:
        ratios = []
        for d in (10, 30):
            e2 = runs[(runs["dim"] == 2) & (runs["algo"] == algo) &
                      (runs["func"].isin(valid))].groupby("func")["error"].median()
            ed = runs[(runs["dim"] == d) & (runs["algo"] == algo) &
                      (runs["func"].isin(valid))].groupby("func")["error"].median()
            ratio = np.exp(np.mean(np.log(err(ed.values) / err(e2.values))))
            ratios.append(ratio)
        ax.plot([10, 30], ratios, marker="^", lw=2, color=algo_color(algo))
    set_log_axis(ax, "y")
    ax.set_xticks([10, 30])
    ax.set_xticklabels(["10 维", "30 维"])
    ax.set_ylabel("相对 2 维的误差放大倍数", fontsize=10)
    ax.set_title("(c) 维度升高带来的误差放大", fontsize=11)

    handles = [Patch(facecolor=algo_color(a), label=a) for a in MAIN_ALGOS]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("维度对算法性能的影响（维度灾难的量化证据）", fontsize=14, y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return save_figure(fig, "图11_维度性能衰减")


# =====================================================================
# 图 12：精度-耗时散点（气泡 = 成功率）
# =====================================================================

def fig12_accuracy_time(runs):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    for ax, dim in zip(axes, [2, 10, 30]):
        d = runs[runs["dim"] == dim]
        stat = d.groupby("algo").agg(
            误差=("error", lambda s: float(np.median(np.log10(err(s))))),
            耗时=("elapsed", "mean"),
            成功率=("success", "mean")).reindex([a for a in MAIN_ALGOS if a in set(d["algo"])])
        for algo, row in stat.iterrows():
            ax.scatter(row["耗时"] * 1000, row["误差"],
                       s=60 + row["成功率"] * 700, color=algo_color(algo), alpha=0.75,
                       edgecolor="white", linewidth=1.2)
            ax.annotate(algo, (row["耗时"] * 1000, row["误差"]),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=9, color=algo_color(algo))
        ax.set_xlabel("平均单次运行耗时（毫秒）", fontsize=10)
        ax.set_ylabel("log10(误差中位数)", fontsize=10)
        ax.set_title(f"{DIM_LABEL[dim]}（气泡大小 = 成功率）", fontsize=11)
    fig.suptitle("精度—耗时权衡：气泡越大表示成功率越高", fontsize=14, y=1.0)
    fig.tight_layout()
    return save_figure(fig, "图12_精度耗时散点")


# =====================================================================
# 图 13：多指标雷达图
# =====================================================================

def fig13_radar(runs, dim=10):
    d = runs[runs["dim"] == dim]
    ranks = _average_ranks(runs)[dim]
    conv = (d[d["success"] & (d["convergence_gen"] >= 0)]
            .groupby("algo")["convergence_gen"].mean())
    stat = pd.DataFrame({
        "精度": -ranks,
        "成功率": d.groupby("algo")["success"].mean(),
        "收敛速度": -conv,
        "稳定性": -d.groupby("algo")["error"].apply(
            lambda s: float(np.std(np.log10(err(s))))),
        "计算效率": -d.groupby("algo")["elapsed"].mean(),
    }).dropna()
    metrics = list(stat.columns)
    norm = (stat - stat.min()) / (stat.max() - stat.min() + 1e-12)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7.6, 7.2), subplot_kw=dict(polar=True))
    for algo in norm.index:
        vals = norm.loc[algo].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, lw=2, color=algo_color(algo),
                label=f"{algo}（{ALGO_CN.get(algo, '')}）")
        ax.fill(angles, vals, color=algo_color(algo), alpha=0.06)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.set_title(f"{DIM_LABEL[dim]}主实验：五种评价指标下的算法综合画像"
                 "（各指标已归一化到 0-1，越靠外越好）", fontsize=13, pad=26)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=9)
    fig.tight_layout()
    return save_figure(fig, "图13_多指标雷达图")


# =====================================================================
# 图 14：2 维搜索轨迹（群体移动过程）
# =====================================================================

def fig14_search_trajectory():
    from algorithms import ALGORITHMS

    panels = [("rastrigin", ["PSO", "GWO"], "Rastrigin"),
              ("schwefel226", ["DE", "GA"], "Schwefel 2.26")]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    for ax, (key, algos, title) in zip(axes, panels):
        b = BENCHMARKS[key]
        XX, YY, ZZ = grid_surface(key, n=220)
        levels = np.logspace(np.log10(max(ZZ.min(), 1e-3)) + 0.4, np.log10(ZZ.max()), 16)
        ax.contourf(XX, YY, ZZ, levels=levels, cmap="viridis", alpha=0.75)
        ax.contour(XX, YY, ZZ, levels=levels, colors="white", linewidths=0.4, alpha=0.5)
        xopt = np.asarray(b.xmin, dtype=float)
        if xopt.size == 1:
            xopt = np.repeat(xopt, 2)
        ax.scatter([xopt[0]], [xopt[1]], marker="*", s=220,
                   color=PALETTE["accent3"], edgecolor="white", zorder=6,
                   label="理论最优位置")
        for i, algo in enumerate(algos):
            res = ALGORITHMS[algo](b, 2, max_iter=120, pop_size=50, seed=3, keep_traj=True)
            style = dict(PSO=("-", "o"), GWO=("--", "s"), DE=("-", "^"), GA=("--", "D"))[algo]
            for gen, best_x, pop in res["trajectory"]:
                if gen % 12 == 0:
                    ax.scatter(pop[:, 0], pop[:, 1], s=9, alpha=0.35,
                               color=algo_color(algo), zorder=3)
            traj = np.array([t[1] for t in res["trajectory"]])
            ax.plot(traj[:, 0], traj[:, 1], style[0], lw=1.6, color=algo_color(algo),
                    alpha=0.95, marker=style[1], ms=3, markevery=8, zorder=5,
                    label=f"{algo} 历代最优位置")
        ax.set_title(f"{title} 上的群体移动轨迹", fontsize=12)
        ax.set_xlabel("x1", fontsize=10)
        ax.set_ylabel("x2", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("2 维搜索轨迹：散点为各代种群个体分布，折线为历代最优位置",
                 fontsize=14, y=1.0)
    fig.tight_layout()
    return save_figure(fig, "图14_搜索轨迹")


# =====================================================================
# 图 15：误差分布 ECDF
# =====================================================================

def _ecdf(x):
    xs = np.sort(np.asarray(x, dtype=float))
    return xs, np.arange(1, len(xs) + 1) / len(xs)


def fig15_ecdf_error(runs):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    for ax, dim in zip(axes, [10, 30]):
        d = runs[runs["dim"] == dim]
        for algo in MAIN_ALGOS:
            v = d[d["algo"] == algo]["error"].values
            if len(v) == 0:
                continue
            xs, ys = _ecdf(np.log10(err(v)))
            ax.plot(xs, ys, lw=2, color=algo_color(algo), label=algo)
        ax.axvline(np.log10(1e-2), color=PALETTE["neutral"], ls=":", lw=1.2)
        ax.set_xlabel("log10(误差)", fontsize=10)
        ax.set_ylabel("累计比例 F(x)", fontsize=10)
        ax.set_title(f"{DIM_LABEL[dim]}：全部 (函数 x 运行) 样本的误差 ECDF", fontsize=11)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("误差经验累积分布（曲线越靠左、越靠上表示精度越高）", fontsize=14, y=1.0)
    fig.tight_layout()
    return save_figure(fig, "图15_误差ECDF")


# =====================================================================
# 图 16：基线方法 vs 智能算法（2 维）
# =====================================================================

def fig16_baseline_compare(runs):
    sub = runs[runs["dim"] == 2]
    keys = list(BENCHMARKS)
    med = sub.pivot_table(index="algo", columns="func", values="error", aggfunc="median")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2))

    # (a) 每个函数上：网格 / 随机 / 最优智能算法 的误差中位数
    ax = axes[0]
    x = np.arange(len(keys))
    w = 0.26
    grid = [med.loc["GRID", k] for k in keys]
    rand = [med.loc["RANDOM", k] for k in keys]
    best, best_algo = [], []
    for k in keys:
        cols = [a for a in MAIN_ALGOS if a in med.index]
        a = med.loc[cols, k].idxmin()
        best.append(med.loc[a, k])
        best_algo.append(a)
    ax.bar(x - w, np.log10(err(grid)), width=w, color=PALETTE["neutral"], label="网格搜索")
    ax.bar(x, np.log10(err(rand)), width=w, color=PALETTE["accent3"], label="随机搜索")
    ax.bar(x + w, np.log10(err(best)), width=w, color=PALETTE["accent2"],
           label="最优群体智能算法")
    for xi, a in zip(x, best_algo):
        ax.text(xi + w, 0.4, a, ha="center", fontsize=7.5, rotation=90, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels([BENCHMARKS[k].cn for k in keys], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("log10(误差中位数)", fontsize=10)
    ax.set_title("(a) 基线方法与智能算法的精度对比", fontsize=11)
    ax.legend(fontsize=9)

    # (b) 最优智能算法相对随机搜索的误差改进倍数
    ax = axes[1]
    gain = np.array([err(g) / err(r) for g, r in zip(best, rand)]).ravel()
    bars = ax.bar(x, gain, color=[PALETTE["main"] if g > 1 else PALETTE["neutral"]
                                  for g in gain])
    for b, g in zip(bars, gain):
        ax.text(b.get_x() + b.get_width() / 2, g * 1.15, f"{g:.1f}×", ha="center",
                fontsize=8, color="#333333")
    set_log_axis(ax, "y")
    ax.set_xticks(x)
    ax.set_xticklabels([BENCHMARKS[k].cn for k in keys], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("误差改进倍数（对数刻度）", fontsize=10)
    ax.set_title("(b) 智能算法相对随机搜索的精度提升", fontsize=11)

    fig.suptitle("基线方法 vs 群体智能算法（2 维，同等评价预算 15000 次）", fontsize=14, y=1.0)
    fig.tight_layout()
    return save_figure(fig, "图16_基线与智能算法对比")


# =====================================================================
# 图 17：收敛速度 ECDF
# =====================================================================

def fig17_ecdf_convergence(runs):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    for ax, dim in zip(axes, [2, 10]):
        d = runs[(runs["dim"] == dim) & runs["success"] & (runs["convergence_gen"] >= 0)]
        empty = True
        for algo in MAIN_ALGOS:
            v = d[d["algo"] == algo]["convergence_gen"].values
            if len(v) < 3:
                continue
            empty = False
            xs, ys = _ecdf(v)
            ax.plot(xs, ys, lw=2, color=algo_color(algo),
                    label=f"{algo}（成功 {len(v)} 次）")
        if empty:
            ax.text(0.5, 0.5, "该维度下成功运行的样本不足", ha="center", va="center",
                    fontsize=11, color=PALETTE["neutral"], transform=ax.transAxes)
        ax.set_xlabel("达到成功判据所需的迭代代数", fontsize=10)
        ax.set_ylabel("累计比例 F(x)", fontsize=10)
        ax.set_title(f"{DIM_LABEL[dim]}：收敛速度 ECDF", fontsize=11)
        ax.legend(fontsize=8)
    fig.suptitle("收敛速度对比（仅统计达到 |f - f*| < 1e-2 的成功运行，曲线越靠左越快）",
                 fontsize=14, y=1.0)
    fig.tight_layout()
    return save_figure(fig, "图17_收敛速度ECDF")


# =====================================================================
# 主流程
# =====================================================================

def main():
    runs, summary, hist = load_all()
    print("=" * 78)
    print("可视化：生成全部图表")
    print("=" * 78)
    print(f"运行记录 {len(runs):,} 行；收敛过程 {len(hist):,} 行")

    todo = [
        ("图01 测试函数 3D 曲面", lambda: fig01_function_surfaces()),
        ("图02 平均收敛曲线（2 维）", lambda: fig02_convergence_2d(hist)),
        ("图03 平均收敛曲线（10 维）", lambda: fig03_convergence_10d(hist)),
        ("图04 误差分布箱线图", lambda: fig04_boxplots(runs)),
        ("图05 误差分布小提琴图", lambda: fig05_violin(runs)),
        ("图06 平均排名", lambda: fig06_average_ranks(runs)),
        ("图07 临界差异图", lambda: fig07_critical_difference(runs, dim=10)),
        ("图08 误差热力图", lambda: fig08_error_heatmap(runs)),
        ("图09 成功率热力图", lambda: fig09_success_heatmap(runs)),
        ("图10 参数敏感性", lambda: fig10_sensitivity()),
        ("图11 维度性能衰减", lambda: fig11_dimension_scaling(runs)),
        ("图12 精度耗时散点", lambda: fig12_accuracy_time(runs)),
        ("图13 多指标雷达图", lambda: fig13_radar(runs, dim=10)),
        ("图14 搜索轨迹", lambda: fig14_search_trajectory()),
        ("图15 误差 ECDF", lambda: fig15_ecdf_error(runs)),
        ("图16 基线与智能算法对比", lambda: fig16_baseline_compare(runs)),
        ("图17 收敛速度 ECDF", lambda: fig17_ecdf_convergence(runs)),
    ]
    n = 0
    for name, fn in todo:
        path = fn()
        plt.close("all")
        if path is not None:
            n += 1
            print(f"[出图] {name:<26} -> {path.name}")
    print(f"\n[完成] 共 {n} 张图，输出目录：{FIG_DIR}")
    return n


if __name__ == "__main__":
    main()
