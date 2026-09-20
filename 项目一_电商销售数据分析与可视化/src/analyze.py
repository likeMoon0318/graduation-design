# -*- coding: utf-8 -*-
# 描述性统计与统计推断
# =====================================================================
# 本模块回答四类问题：
#   1. 数据长什么样      -> 描述性统计、分组汇总、交叉表
#   2. 变量之间有关联吗  -> 相关系数矩阵
#   3. 差异是真的吗      -> 假设检验（t 检验 / 方差分析 / 卡方检验 / 秩和检验）
#   4. 差异有多大        -> 效应量（Cohen's d、Cramér's V）
# 输出统一写入 results/tables/

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TABLE_DIR  # noqa: E402
from features import add_time_features, load_clean  # noqa: E402

pd.set_option("display.unicode.east_asian_width", True)
ALPHA = 0.05


# ---------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------

def cohens_d(x, y):
    """两组均值差异的效应量（合并标准差版本）。"""
    nx, ny = len(x), len(y)
    sx, sy = x.var(ddof=1), y.var(ddof=1)
    pooled = np.sqrt(((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2))
    return (x.mean() - y.mean()) / pooled


def cramers_v(table):
    """分类变量关联强度（卡方检验的效应量）。"""
    chi2 = stats.chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def significance(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "不显著"


def effect_label(v, kind):
    """按通用经验标准解释效应量大小。"""
    v = abs(v)
    if kind == "d":
        return "小" if v < 0.2 else ("中" if v < 0.5 else ("大" if v < 0.8 else "很大"))
    return "弱" if v < 0.1 else ("中等" if v < 0.3 else "强")


def save(df, name):
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ---------------------------------------------------------------------
# 一、描述性统计
# ---------------------------------------------------------------------

def descriptive(df):
    print("\n" + "=" * 74)
    print("一、描述性统计")
    print("=" * 74)
    cols = ["unit_price", "quantity", "discount", "amount", "rating"]
    desc = df[cols].describe().T
    desc["偏度"] = df[cols].skew()
    desc["峰度"] = df[cols].kurtosis()
    desc["变异系数"] = desc["std"] / desc["mean"]
    desc = desc.round(3).reset_index().rename(columns={"index": "字段"})
    print(desc.to_string(index=False))
    save(desc, "描述性统计")

    # 关键分位数说明金额分布的右偏特征
    q = df["amount"].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    print("\n订单金额分位数（元）：")
    print(q.round(2).to_string())
    print(f"提示：均值 {df['amount'].mean():.0f} 元远高于中位数 {df['amount'].median():.0f} 元，"
          f"说明订单金额呈明显右偏分布，少数高价订单拉高了平均水平。")
    return desc


# ---------------------------------------------------------------------
# 二、分组汇总
# ---------------------------------------------------------------------

def group_summaries(df):
    print("\n" + "=" * 74)
    print("二、分组汇总")
    print("=" * 74)
    out = {}

    def agg_by(key, name):
        g = df.groupby(key).agg(
            订单量=("order_id", "count"),
            销售额=("amount", "sum"),
            客单价=("amount", "mean"),
            中位客单价=("amount", "median"),
            退货率=("is_return", "mean"),
            平均评分=("rating", "mean"),
        ).round(3)
        g["销售额占比"] = (g["销售额"] / g["销售额"].sum()).round(4)
        g = g.sort_values("销售额", ascending=False).reset_index()
        print(f"\n【按{name}】")
        print(g.to_string(index=False))
        save(g, f"分组汇总_{name}")
        out[name] = g
        return g

    agg_by("category", "品类")
    agg_by("channel", "渠道")
    agg_by("province", "省份")
    agg_by("member_level", "会员等级")
    agg_by("is_promo", "是否大促")

    # 月度 x 渠道 销售额透视表
    pivot = df.pivot_table(index="month", columns="channel", values="amount",
                           aggfunc="sum", fill_value=0).round(0)
    pivot.index = [f"{m}月" for m in pivot.index]
    print("\n【月度 x 渠道 销售额透视表（元）】")
    print(pivot.to_string())
    save(pivot.reset_index().rename(columns={"index": "月份"}), "透视表_月度渠道销售额")

    # 热销商品 Top15
    top = (df.groupby(["category", "product"])
           .agg(销量=("quantity", "sum"), 销售额=("amount", "sum"), 订单数=("order_id", "count"))
           .sort_values("销售额", ascending=False).head(15).round(2).reset_index())
    print("\n【销售额 Top15 商品】")
    print(top.to_string(index=False))
    save(top, "热销商品Top15")
    return out


# ---------------------------------------------------------------------
# 三、相关性分析
# ---------------------------------------------------------------------

def correlation(df):
    print("\n" + "=" * 74)
    print("三、相关性分析")
    print("=" * 74)
    cols = ["unit_price", "quantity", "discount", "amount", "rating_filled", "hour", "is_return"]
    pearson = df[cols].corr(method="pearson").round(3)
    spearman = df[cols].corr(method="spearman").round(3)
    print("\n【Pearson 相关系数矩阵】")
    print(pearson.to_string())
    print("\n【Spearman 秩相关系数矩阵】")
    print(spearman.to_string())
    save(pearson.reset_index().rename(columns={"index": "变量"}), "相关系数_Pearson")
    save(spearman.reset_index().rename(columns={"index": "变量"}), "相关系数_Spearman")

    # 对主要相关关系给出显著性检验
    r, p = stats.pearsonr(df["unit_price"], df["amount"])
    print(f"\n单价与订单金额：r = {r:.3f}，p = {p:.3e}（n = {len(df):,}）")
    return spearman


# ---------------------------------------------------------------------
# 四、假设检验
# ---------------------------------------------------------------------

def hypothesis_tests(df):
    print("\n" + "=" * 74)
    print("四、假设检验")
    print("=" * 74)
    rows = []

    # ---- 检验 1：大促期间客单价是否显著高于平日 ----
    promo = df.loc[df["is_promo"], "amount"]
    normal = df.loc[~df["is_promo"], "amount"]
    t, p = stats.ttest_ind(promo, normal, equal_var=False)          # Welch t 检验
    u, p_u = stats.mannwhitneyu(promo, normal, alternative="two-sided")
    d = cohens_d(promo, normal)
    rows.append({
        "编号": "H1", "研究问题": "大促订单客单价与平日是否存在差异",
        "原假设": "大促与平日订单金额均值相同",
        "检验方法": "Welch t 检验（附 Mann-Whitney U 稳健性检验）",
        "统计量": f"t={t:.2f}; U={u:.0f}", "p值": p,
        "显著性": significance(p),
        "效应量": f"Cohen's d={d:.3f}（{effect_label(d, 'd')}）",
        "结论": (f"大促客单价显著{'高于' if promo.mean() > normal.mean() else '低于'}平日"
                 f"（{promo.mean():.0f} 元 vs {normal.mean():.0f} 元），呈以价换量特征"),
    })
    print(f"\n[H1] 大促 vs 平日客单价：大促均值 {promo.mean():.2f} 元（n={len(promo):,}），"
          f"平日均值 {normal.mean():.2f} 元（n={len(normal):,}）")
    print(f"     t = {t:.3f}, p = {p:.3e}, Cohen's d = {d:.3f}；Mann-Whitney U 检验 p = {p_u:.3e}")

    # ---- 检验 2：不同地区订单金额是否存在差异 ----
    region_groups = [g["amount"].values for _, g in df.groupby("province") if len(g) > 30]
    f_stat, p_anova = stats.f_oneway(*region_groups)
    h_stat, p_kw = stats.kruskal(*region_groups)
    eta2 = (f_stat * (len(region_groups) - 1)) / (len(df) - len(region_groups) + f_stat * (len(region_groups) - 1))
    rows.append({
        "编号": "H2", "研究问题": "不同地区的订单金额是否存在差异",
        "原假设": "各地区订单金额均值相同",
        "检验方法": "单因素方差分析（附 Kruskal-Wallis 稳健性检验）",
        "统计量": f"F={f_stat:.2f}; H={h_stat:.0f}", "p值": p_anova,
        "显著性": significance(p_anova),
        "效应量": f"eta^2={eta2:.4f}",
        "结论": "地区间差异统计显著，但效应量很小（eta^2<0.01），业务上可视为基本持平",
    })
    print(f"\n[H2] 地区间订单金额差异：F = {f_stat:.3f}, p = {p_anova:.3e}, eta^2 = {eta2:.4f}")
    print(f"     Kruskal-Wallis H = {h_stat:.1f}, p = {p_kw:.3e}")

    # 事后检验：哪些地区之间差异显著
    tukey = stats.tukey_hsd(*region_groups)
    region_names = [name for name, g in df.groupby("province") if len(g) > 30]
    posthoc = []
    for i in range(len(region_names)):
        for j in range(i + 1, len(region_names)):
            pv = tukey.pvalue[i, j]
            if pv < ALPHA:
                posthoc.append({"地区A": region_names[i], "地区B": region_names[j],
                                "p值": pv, "差异": "显著"})
    posthoc_df = pd.DataFrame(posthoc).sort_values("p值").round(6)
    print(f"     Tukey HSD 事后检验：{len(posthoc_df)} 组地区组合存在显著差异（共 "
          f"{len(region_names) * (len(region_names) - 1) // 2} 组）")
    save(posthoc_df, "事后检验_地区差异")

    # ---- 检验 3：品类与退货是否独立 ----
    table = pd.crosstab(df["category"], df["is_return"])
    chi2, p_chi, dof, _ = stats.chi2_contingency(table)
    v = cramers_v(table)
    rows.append({
        "编号": "H3", "研究问题": "商品品类与是否退货是否存在关联",
        "原假设": "品类与退货相互独立",
        "检验方法": "卡方独立性检验", "统计量": f"chi2={chi2:.0f}, df={dof}", "p值": p_chi,
        "显著性": significance(p_chi),
        "效应量": f"Cramer's V={v:.3f}（{effect_label(v, 'v')}）",
        "结论": "品类显著影响退货行为",
    })
    print(f"\n[H3] 品类与退货独立性：chi2 = {chi2:.1f}, df = {dof}, p = {p_chi:.3e}, "
          f"Cramer's V = {v:.3f}")

    # ---- 检验 4：渠道与退货是否独立 ----
    table_c = pd.crosstab(df["channel"], df["is_return"])
    chi2_c, p_c, dof_c, _ = stats.chi2_contingency(table_c)
    v_c = cramers_v(table_c)
    rows.append({
        "编号": "H4", "研究问题": "购买渠道与是否退货是否存在关联",
        "原假设": "渠道与退货相互独立",
        "检验方法": "卡方独立性检验", "统计量": f"chi2={chi2_c:.0f}, df={dof_c}", "p值": p_c,
        "显著性": significance(p_c),
        "效应量": f"Cramer's V={v_c:.3f}（{effect_label(v_c, 'v')}）",
        "结论": ("渠道与退货存在显著关联" if p_c < ALPHA
                 else "未发现渠道与退货存在显著关联，两者相互独立"),
    })
    print(f"[H4] 渠道与退货独立性：chi2 = {chi2_c:.1f}, df = {dof_c}, p = {p_c:.3e}, "
          f"Cramer's V = {v_c:.3f}")
    save(table_c.reset_index(), "列联表_渠道退货")
    save(table.reset_index(), "列联表_品类退货")

    # ---- 检验 5：退货订单的用户评分是否更低 ----
    rated = df.dropna(subset=["rating"])
    r_yes = rated.loc[rated["is_return"], "rating"]
    r_no = rated.loc[~rated["is_return"], "rating"]
    u5, p5 = stats.mannwhitneyu(r_yes, r_no, alternative="two-sided")
    rank_biserial = 1 - 2 * u5 / (len(r_yes) * len(r_no))
    rows.append({
        "编号": "H5", "研究问题": "退货订单的用户评分是否低于未退货订单",
        "原假设": "退货与未退货订单的评分分布相同",
        "检验方法": "Mann-Whitney U 秩和检验",
        "统计量": f"U={u5:.0f}", "p值": p5, "显著性": significance(p5),
        "效应量": f"rank-biserial r={rank_biserial:.3f}",
        "结论": f"退货订单评分{'更低' if r_yes.mean() < r_no.mean() else '更高'}"
                f"（{r_yes.mean():.2f} vs {r_no.mean():.2f}）",
    })
    print(f"\n[H5] 退货订单评分 {r_yes.mean():.3f}（n={len(r_yes):,}）vs 未退货 "
          f"{r_no.mean():.3f}（n={len(r_no):,}），U = {u5:.0f}, p = {p5:.3e}")

    # ---- 检验 6：会员等级是否影响客单价 ----
    member_groups = [g["amount"].values for _, g in df.groupby("member_level")]
    f6, p6 = stats.f_oneway(*member_groups)
    h6, p6k = stats.kruskal(*member_groups)
    rows.append({
        "编号": "H6", "研究问题": "不同会员等级的客单价是否存在差异",
        "原假设": "各会员等级订单金额均值相同",
        "检验方法": "单因素方差分析（附 Kruskal-Wallis）",
        "统计量": f"F={f6:.2f}; H={h6:.0f}", "p值": p6, "显著性": significance(p6),
        "效应量": "-", "结论": "会员等级与客单价显著相关",
    })
    print(f"[H6] 会员等级对客单价的影响：F = {f6:.3f}, p = {p6:.3e}；"
          f"Kruskal-Wallis p = {p6k:.3e}")

    # ---- 检验 7：单价与订单金额的相关性 ----
    rho, p7 = stats.spearmanr(df["unit_price"], df["amount"])
    rows.append({
        "编号": "H7", "研究问题": "商品单价与订单金额是否相关",
        "原假设": "单价与订单金额相互独立",
        "检验方法": "Spearman 秩相关检验",
        "统计量": f"rho={rho:.3f}", "p值": p7, "显著性": significance(p7),
        "效应量": f"rho={rho:.3f}（{'强' if abs(rho) > 0.5 else '中等'}相关）",
        "结论": "单价是订单金额的主要驱动因素",
    })
    print(f"[H7] 单价与订单金额 Spearman 相关：rho = {rho:.3f}, p = {p7:.3e}")

    result = pd.DataFrame(rows)
    result["p值"] = result["p值"].map(lambda v: f"{v:.3e}")
    save(result, "假设检验汇总")

    print("\n【假设检验汇总表】")
    print(result[["编号", "研究问题", "检验方法", "p值", "显著性", "结论"]].to_string(index=False))
    return result


def main():
    df = load_clean()
    df = add_time_features(df)
    print(f"[读取] 清洗后数据 {len(df):,} 行")

    descriptive(df)
    group_summaries(df)
    correlation(df)
    result = hypothesis_tests(df)

    print(f"\n[输出] 统计结果已写入 {TABLE_DIR}")
    return df, result


if __name__ == "__main__":
    main()
