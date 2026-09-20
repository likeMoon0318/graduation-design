# -*- coding: utf-8 -*-
# 统计分析与显著性检验（stats_analysis.py）
# =====================================================================
# 说明：本模块原本命名为 statistics.py，但该文件名会遮蔽 Python 标准库的
#      statistics 模块（seaborn 等第三方库依赖它），因此改名为 stats_analysis.py。
# 1) 汇总统计：最优值 / 均值 / 标准差 / 中位数 / 成功率 / 平均收敛代数 / 平均耗时
# 2) Friedman 检验：同一维度下 8 种算法在全部 (函数 x 运行) 配对样本上的整体差异
# 3) Nemenyi 事后检验：给出临界差值 CD，用于绘制临界差异图
# 4) Wilcoxon 符号秩检验：算法两两比较，Holm 方法校正多重比较
# 5) 胜负平矩阵：逐问题比较算法优劣

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import ALGO_CN, MAIN_ALGOS, RUNS_FINAL, SUMMARY_FILE, TABLE_DIR  # noqa: E402

# Nemenyi 检验的临界值 q_alpha（alpha = 0.05，自由度无穷大）
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}

DIM_LABEL = {2: "2 维", 10: "10 维", 30: "30 维"}


def load_runs():
    df = pd.read_csv(RUNS_FINAL)
    df["algo_cn"] = df["algo"].map(lambda a: ALGO_CN.get(a, a))
    return df


# ---------------------------------------------------------------- 1) 汇总统计

def summarize(df):
    g = df.groupby(["dim", "func_cn", "category", "algo", "algo_cn"], as_index=False).agg(
        运行次数=("run", "count"),
        最优值best=("best_value", "min"),
        平均最优值mean=("best_value", "mean"),
        标准差std=("best_value", "std"),
        中位数median=("best_value", "median"),
        最差值worst=("best_value", "max"),
        平均误差=("error", "mean"),
        误差中位数=("error", "median"),
        成功率=("success", "mean"),
        平均耗时秒=("elapsed", "mean"),
        平均评价次数=("n_evals", "mean"),
    )
    # 平均收敛代数只统计“成功的那几次运行”，失败运行不参与平均
    conv = (df[df["success"] & (df["convergence_gen"] >= 0)]
            .groupby(["dim", "func_cn", "algo"])["convergence_gen"].mean()
            .rename("平均收敛代数").reset_index())
    g = g.merge(conv, on=["dim", "func_cn", "algo"], how="left")
    for c in ["最优值best", "平均最优值mean", "标准差std", "中位数median", "最差值worst",
              "平均误差", "误差中位数", "平均耗时秒", "平均评价次数"]:
        g[c] = g[c].astype(float).round(6)
    g["成功率"] = (g["成功率"] * 100).round(1)
    g["平均收敛代数"] = g["平均收敛代数"].round(1)
    g = g.sort_values(["dim", "func_cn", "平均误差"])
    g.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")
    return g


# ---------------------------------------------------------------- 2) 平均排名

def average_ranks(df, dims=None):
    """按 (维度, 函数, 运行) 分块，对算法按误差升序排名（1 为最好）。"""
    # 注意：不同维度参与的算法集合不同（2 维含网格/随机搜索基线），
    # 因此必须逐维度构造配对区组，避免整体 dropna 把 10/30 维的样本全部删掉。
    out = []
    for dim in sorted(df["dim"].unique()):
        if dims is not None and dim not in dims:
            continue
        sub = df[df["dim"] == dim]
        blocks = sub.pivot_table(index=["func", "run"], columns="algo",
                                 values="error", aggfunc="mean").dropna(axis=0, how="any")
        ranks = blocks.rank(axis=1, method="average")
        for algo in ranks.columns:
            out.append({"dim": dim, "algo": algo, "algo_cn": ALGO_CN.get(algo, algo),
                        "样本块数": int(ranks.shape[0]), "平均排名": ranks[algo].mean(),
                        "排名标准差": ranks[algo].std()})
    res = pd.DataFrame(out)
    res["名次"] = res.groupby("dim")["平均排名"].rank(method="min").astype(int)
    res["平均排名"] = res["平均排名"].round(3)
    res["排名标准差"] = res["排名标准差"].round(3)
    res = res.sort_values(["dim", "平均排名"])
    res.to_csv(TABLE_DIR / "平均排名.csv", index=False, encoding="utf-8-sig")
    return res, None


# ---------------------------------------------------------------- 3) Friedman + Nemenyi

def friedman_and_nemenyi(df):
    rows = []
    blocks_all = df.pivot_table(index=["func", "run", "algo"], values="error",
                                aggfunc="mean").reset_index()
    for dim in sorted(df["dim"].unique()) + ["全部维度"]:
        sub = df if dim == "全部维度" else df[df["dim"] == dim]
        # 每个 (函数, 运行) 是一个区组，若同一函数出现在多个维度，则把维度也作为区分
        idx = ["dim", "func", "run"] if dim == "全部维度" else ["func", "run"]
        blocks = sub.pivot_table(index=idx, columns="algo", values="error", aggfunc="mean")
        algos = [a for a in blocks.columns if blocks[a].notna().all()]
        blocks = blocks[algos]
        if blocks.shape[0] < 2 or len(algos) < 3:
            continue
        stat, p = stats.friedmanchisquare(*[blocks[a].values for a in algos])
        n, k = blocks.shape
        cd = None
        if k in NEMENYI_Q05:
            cd = NEMENYI_Q05[k] * np.sqrt(k * (k + 1) / (6.0 * n))
        rows.append({
            "维度": DIM_LABEL.get(dim, dim), "算法数k": k, "区组数N": n,
            "Friedman统计量": round(float(stat), 4), "p值": float(p),
            "显著性": "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "不显著",
            "Nemenyi临界差值CD": None if cd is None else round(float(cd), 3),
        })
    res = pd.DataFrame(rows)
    res.to_csv(TABLE_DIR / "friedman检验.csv", index=False, encoding="utf-8-sig")
    return res


# ---------------------------------------------------------------- 4) Wilcoxon 两两检验

def holm_adjust(pvals):
    """Holm-Bonferroni 逐步校正，返回校正后的 p 值（保持原顺序）。"""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def wilcoxon_pairs(df, dims=None):
    dims = sorted(df["dim"].unique()) if dims is None else dims
    rows = []
    for dim in dims:
        sub = df[df["dim"] == dim]
        blocks = sub.pivot_table(index=["func", "run"], columns="algo",
                                 values="error", aggfunc="mean").dropna(axis=0, how="any")
        algos = list(blocks.columns)
        pairs = list(itertools.combinations(algos, 2))
        raw = []
        for a, b in pairs:
            x, y = blocks[a].values, blocks[b].values
            diff = x - y
            if np.allclose(diff, 0):
                w, p = np.nan, 1.0
            else:
                w, p = stats.wilcoxon(x, y, zero_method="wilcox")
            raw.append((a, b, w, p, int(np.sum(x < y)), int(np.sum(x == y)), int(np.sum(x > y))))
        adj = holm_adjust([r[3] for r in raw])
        for (a, b, w, p, win, tie, lose), pa in zip(raw, adj):
            rows.append({
                "维度": DIM_LABEL.get(dim, dim), "算法A": a, "算法B": b,
                "对称样本数": int(blocks.shape[0]),
                "A优于B次数": win, "持平次数": tie, "A劣于B次数": lose,
                "统计量W": None if np.isnan(w) else round(float(w), 2),
                "p值": p, "Holm校正p值": round(float(pa), 6),
                "校正后显著性": "***" if pa < 0.001 else "**" if pa < 0.01
                                else "*" if pa < 0.05 else "不显著",
            })
    res = pd.DataFrame(rows)
    res.to_csv(TABLE_DIR / "wilcoxon两两检验.csv", index=False, encoding="utf-8-sig")
    return res


# ---------------------------------------------------------------- 5) 胜负平矩阵

def win_loss_matrix(df, dims=None):
    """以“平均误差更低者胜”为判据，统计每种算法对每种算法在全部问题上的胜负平。"""
    dims = sorted(df["dim"].unique()) if dims is None else dims
    sheets = {}
    for dim in dims:
        sub = df[df["dim"] == dim]
        per = sub.groupby(["func", "algo"])["error"].mean().unstack()
        algos = list(per.columns)
        mat = pd.DataFrame(index=algos, columns=algos, dtype=object)
        for a in algos:
            for b in algos:
                if a == b:
                    mat.loc[a, b] = "-"
                    continue
                win = int(np.sum(per[a] < per[b]))
                lose = int(np.sum(per[a] > per[b]))
                tie = int(np.sum(per[a] == per[b]))
                mat.loc[a, b] = f"{win}胜/{tie}平/{lose}负"
        sheets[dim] = mat
        mat.to_csv(TABLE_DIR / f"胜负平矩阵_{dim}维.csv", encoding="utf-8-sig")
    return sheets


def main():
    df = load_runs()
    print("=" * 78)
    print("统计分析与显著性检验")
    print("=" * 78)
    print(f"读入运行记录：{len(df):,} 行（{df['dim'].nunique()} 个维度、"
          f"{df['func'].nunique()} 个函数、{df['algo'].nunique()} 种算法、"
          f"{df['run'].nunique()} 次重复）")

    summary = summarize(df)
    print(f"\n[输出] 汇总统计：{SUMMARY_FILE.name}（{len(summary)} 行）")

    ranks, _ = average_ranks(df)
    print(f"[输出] 平均排名：平均排名.csv")
    for dim in sorted(ranks["dim"].unique()):
        top = ranks[ranks["dim"] == dim].head(3)
        txt = "、".join(f"{r['algo']}({r['平均排名']:.2f})" for _, r in top.iterrows())
        print(f"        {DIM_LABEL.get(dim, dim)} 前三名：{txt}")

    fried = friedman_and_nemenyi(df)
    print(f"[输出] Friedman 检验：friedman检验.csv")
    for _, r in fried.iterrows():
        print(f"        {r['维度']:<8} chi2={r['Friedman统计量']:>9.3f}  p={r['p值']:.3e}  "
              f"{r['显著性']}  CD={r['Nemenyi临界差值CD']}")

    wil = wilcoxon_pairs(df)
    n_sig = (wil["校正后显著性"] != "不显著").sum()
    print(f"[输出] Wilcoxon 两两检验：wilcoxon两两检验.csv")
    print(f"        共 {len(wil)} 组两两比较，Holm 校正后显著 {n_sig} 组")

    win_loss_matrix(df)
    print(f"[输出] 胜负平矩阵：胜负平矩阵_*.csv")

    # 主实验（10 维）成功率排行，供报告直接引用
    main_dim = 10
    if main_dim in set(df["dim"]):
        sub = df[df["dim"] == main_dim]
        pool = sub.groupby("algo").agg(平均误差=("error", "mean"),
                                       成功率=("success", "mean"),
                                       平均耗时=("elapsed", "mean")).sort_values("平均误差")
        pool["成功率"] = (pool["成功率"] * 100).round(1)
        print(f"\n[{main_dim} 维主实验] 算法综合表现：")
        print(pool.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
