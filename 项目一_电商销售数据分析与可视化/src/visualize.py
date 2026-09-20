# -*- coding: utf-8 -*-
# 可视化模块：生成报告所需的全部图表
# =====================================================================
# 输出：results/figures/*.png（300dpi，word 插图用）+ *.pdf（矢量图，打印用）
# 图表清单见 README 或项目规划书；每张图对应报告中的一个分析结论

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # 出图不需要交互窗口
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (CAT_COLORS, FIG_DIR, PALETTE, RAW_FILE, TABLE_DIR,  # noqa: E402
                    setup_chinese_font, save_figure)
from data_generator import PROMOTIONS  # noqa: E402
from features import add_time_features, build_customer_rfm, load_clean  # noqa: E402

FONT = setup_chinese_font()
sns.set_style("whitegrid", {"font.sans-serif": [FONT], "axes.unicode_minus": False})


def _money(x, _pos=None):
    """把金额（元）格式化为万元/亿元，便于图表阅读。"""
    if abs(x) >= 1e8:
        return f"{x / 1e8:.1f}亿"
    if abs(x) >= 1e4:
        return f"{x / 1e4:.0f}万"
    return f"{x:.0f}"


MONEY_FMT = FuncFormatter(_money)


# ---------------------------------------------------------------------
# 图 1：全年销售额趋势
# ---------------------------------------------------------------------

def fig01_daily_trend(df):
    daily = df.groupby("date").agg(销售额=("amount", "sum"), 订单量=("order_id", "count")).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["7日移动平均"] = daily["销售额"].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.plot(daily["date"], daily["销售额"], color=PALETTE["neutral"], lw=0.8, alpha=0.7, label="每日销售额")
    ax.plot(daily["date"], daily["7日移动平均"], color=PALETTE["main"], lw=2.4, label="7 日移动平均")

    # 标注大促区间
    for i, (name, start, end, _lo, _hi) in enumerate(PROMOTIONS):
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        ax.axvspan(s, e, color=PALETTE["accent"], alpha=0.13, lw=0)
        ax.text(s + (e - s) / 2, daily["销售额"].max() * (0.93 - 0.075 * (i % 2)),
                name, ha="center", fontsize=9, color=PALETTE["accent3"])

    ax.set_title("2025 年每日销售额趋势与促销活动分布", fontsize=15, pad=14)
    ax.set_xlabel("日期")
    ax.set_ylabel("销售额（元）")
    ax.yaxis.set_major_formatter(MONEY_FMT)
    ax.legend(loc="upper left", frameon=True)
    ax.text(0.99, 0.02,
            f"全年销售额 {daily['销售额'].sum() / 1e8:.2f} 亿元　"
            f"日均 {daily['销售额'].mean() / 1e4:.1f} 万元　"
            f"峰值 {daily['销售额'].max() / 1e4:.0f} 万元（{daily.loc[daily['销售额'].idxmax(), 'date']:%m月%d日}）",
            transform=ax.transAxes, ha="right", fontsize=10, color=PALETTE["main"],
            bbox=dict(boxstyle="round,pad=0.45", fc="#F2F6FA", ec=PALETTE["main"], alpha=0.9))
    return save_figure(fig, "图01_全年销售额趋势")


# ---------------------------------------------------------------------
# 图 2：下单时段热力图
# ---------------------------------------------------------------------

def fig02_hour_weekday_heatmap(df):
    order = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    pivot = df.pivot_table(index="weekday_name", columns="hour", values="order_id",
                           aggfunc="count").reindex(order)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.0), gridspec_kw={"width_ratios": [3, 1]})
    sns.heatmap(pivot, cmap="YlGnBu", ax=axes[0], cbar_kws={"label": "订单量（单）"})
    axes[0].set_title("下单时间分布：星期 x 小时", fontsize=14, pad=12)
    axes[0].set_xlabel("小时")
    axes[0].set_ylabel("")
    axes[0].set_xticks(np.arange(24)[::2])
    axes[0].set_xticklabels(np.arange(24)[::2])

    hourly = df.groupby("hour")["order_id"].count()
    axes[1].plot(hourly.values, hourly.index, color=PALETTE["accent"], lw=2.2, marker="o", ms=3)
    axes[1].fill_betweenx(hourly.index, 0, hourly.values, color=PALETTE["accent"], alpha=0.18)
    axes[1].set_title("各时段订单量", fontsize=13, pad=12)
    axes[1].set_ylim(23.5, -0.5)
    axes[1].set_xlabel("订单量（单）")
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:.0f}k"))
    peak = int(hourly.idxmax())
    axes[1].annotate(f"峰值 {peak}:00", xy=(hourly.max(), peak),
                     xytext=(hourly.max() * 0.45, peak - 4.5), fontsize=10,
                     color=PALETTE["accent3"],
                     arrowprops=dict(arrowstyle="->", color=PALETTE["accent3"]))
    return save_figure(fig, "图02_下单时段分布热力图")


# ---------------------------------------------------------------------
# 图 3：品类帕累托图
# ---------------------------------------------------------------------

def fig03_category_pareto(df):
    g = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    ratio = g / g.sum()
    cum = ratio.cumsum()

    fig, ax1 = plt.subplots(figsize=(11, 5.4))
    bars = ax1.bar(g.index, g.values, color=CAT_COLORS[:len(g)], width=0.62)
    ax1.set_ylabel("销售额（元）", color=PALETTE["main"])
    ax1.yaxis.set_major_formatter(MONEY_FMT)
    ax1.set_xlabel("")
    for b, v in zip(bars, g.values):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v / 1e4:.0f}万",
                 ha="center", va="bottom", fontsize=9.5, color="#333333")

    ax2 = ax1.twinx()
    ax2.plot(cum.index, cum.values, color=PALETTE["accent"], marker="o", lw=2.2, ms=6)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("累计占比", color=PALETTE["accent"])
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0%}"))
    ax2.grid(False)
    for x, y in zip(cum.index, cum.values):
        ax2.text(x, y + 0.035, f"{y:.1%}", ha="center", fontsize=9.5, color=PALETTE["accent"])

    top2 = cum.iloc[1]
    ax1.set_title(f"各品类销售额与累计占比（帕累托图）\n前 2 个品类贡献 {top2:.1%} 的销售额",
                  fontsize=14, pad=14)
    ax1.set_xticklabels(g.index, rotation=20, ha="right")
    return save_figure(fig, "图03_品类销售额帕累托图")


# ---------------------------------------------------------------------
# 图 4：渠道结构及其演变
# ---------------------------------------------------------------------

def fig04_channel_structure(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1.55]})

    ch = df.groupby("channel")["amount"].sum().sort_values(ascending=False)
    axes[0].pie(ch.values, labels=ch.index, autopct="%1.1f%%", startangle=90,
                colors=CAT_COLORS[:len(ch)], wedgeprops=dict(width=0.42, edgecolor="white"),
                textprops={"fontsize": 11})
    axes[0].set_title("全年渠道销售额占比", fontsize=14, pad=12)

    monthly = df.pivot_table(index="month", columns="channel", values="amount",
                             aggfunc="sum", fill_value=0)
    share = monthly.div(monthly.sum(axis=1), axis=0)
    axes[1].stackplot(share.index, [share[c] for c in share.columns],
                      labels=list(share.columns), colors=CAT_COLORS[:len(share.columns)], alpha=0.85)
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels([f"{m}月" for m in range(1, 13)])
    axes[1].set_ylim(0, 1)
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0%}"))
    axes[1].set_title("渠道销售额占比的月度演变", fontsize=14, pad=12)
    axes[1].legend(loc="lower left", ncol=4, fontsize=9.5, frameon=True)
    axes[1].set_xlabel("月份")
    axes[1].set_ylabel("销售额占比")

    first, last = share.iloc[0], share.iloc[-1]
    axes[1].text(0.99, 0.97,
                 f"APP 占比 {first['APP']:.0%} -> {last['APP']:.0%}；"
                 f"网页端 {first['网页']:.0%} -> {last['网页']:.0%}",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.4", fc="#FFF8F2", ec=PALETTE["accent"]))
    return save_figure(fig, "图04_渠道结构及演变")


# ---------------------------------------------------------------------
# 图 5：地区销售额与客单价
# ---------------------------------------------------------------------

def fig05_region(df):
    g = df.groupby("province").agg(销售额=("amount", "sum"), 客单价=("amount", "mean"),
                                   订单量=("order_id", "count")).sort_values("销售额")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), gridspec_kw={"width_ratios": [1.35, 1]})

    bars = axes[0].barh(g.index, g["销售额"], color=PALETTE["main"], alpha=0.85)
    for b, (v, city_n) in zip(bars, zip(g["销售额"], g["订单量"])):
        axes[0].text(v, b.get_y() + b.get_height() / 2, f" {v / 1e4:.0f}万 / {city_n / 1000:.1f}k单",
                     va="center", fontsize=9)
    axes[0].set_title("各省销售额与订单量排名", fontsize=14, pad=12)
    axes[0].set_xlabel("销售额（元）")
    axes[0].xaxis.set_major_formatter(MONEY_FMT)
    axes[0].set_xlim(0, g["销售额"].max() * 1.35)

    g2 = g.sort_values("客单价")
    axes[1].barh(g2.index, g2["客单价"], color=PALETTE["accent"], alpha=0.88)
    axes[1].axvline(df["amount"].mean(), color=PALETTE["accent3"], ls="--", lw=1.6,
                    label=f"全国均值 {df['amount'].mean():.0f} 元")
    axes[1].set_title("各省平均客单价", fontsize=14, pad=12)
    axes[1].set_xlabel("平均客单价（元）")
    axes[1].legend(loc="lower right", fontsize=10)
    return save_figure(fig, "图05_地区销售对比")


# ---------------------------------------------------------------------
# 图 6：订单金额分布
# ---------------------------------------------------------------------

def fig06_amount_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))

    data = np.log10(df["amount"].clip(lower=0.01))
    axes[0].hist(data, bins=90, color=PALETTE["main"], alpha=0.8, edgecolor="white", lw=0.3)
    axes[0].axvline(np.log10(df["amount"].mean()), color=PALETTE["accent3"], ls="--", lw=2,
                    label=f"均值 {df['amount'].mean():.0f} 元")
    axes[0].axvline(np.log10(df["amount"].median()), color=PALETTE["accent"], ls="-", lw=2,
                    label=f"中位数 {df['amount'].median():.0f} 元")
    axes[0].set_xticks([0, 1, 2, 3, 4, 5])
    axes[0].set_xticklabels(["1", "10", "100", "1000", "1万", "10万"])
    axes[0].set_xlabel("订单金额（元，对数刻度）")
    axes[0].set_ylabel("订单数")
    axes[0].set_title("订单金额分布（右偏长尾）", fontsize=14, pad=12)
    axes[0].legend(fontsize=10.5)

    # 箱线图：按订单金额分组展示消费结构
    bins = [0, 100, 300, 1000, 3000, 10000, np.inf]
    labels = ["100元以下", "100-300元", "300-1000元", "1000-3000元", "3000-1万元", "1万元以上"]
    band = pd.cut(df["amount"], bins=bins, labels=labels, right=False)
    counts = band.value_counts().reindex(labels)
    share = counts / counts.sum()
    bars = axes[1].bar(labels, counts.values, color=CAT_COLORS[:len(labels)], width=0.62)
    for b, c, s in zip(bars, counts.values, share.values):
        axes[1].text(b.get_x() + b.get_width() / 2, c, f"{c / 1000:.1f}k\n{s:.1%}",
                     ha="center", va="bottom", fontsize=9.5)
    axes[1].set_title("订单金额区间分布", fontsize=14, pad=12)
    axes[1].set_ylabel("订单数")
    axes[1].set_xticklabels(labels, rotation=22, ha="right")
    axes[1].set_ylim(0, counts.max() * 1.22)
    return save_figure(fig, "图06_订单金额分布")


# ---------------------------------------------------------------------
# 图 7：各品类客单价箱线图
# ---------------------------------------------------------------------

def fig07_category_boxplot(df):
    order = df.groupby("category")["amount"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    sns.boxplot(data=df, x="category", y="amount", order=order, ax=ax,
                palette=CAT_COLORS[:len(order)], width=0.55, fliersize=0.6, linewidth=0.9)
    ax.set_yscale("log")
    ax.set_yticks([10, 100, 1000, 10000, 100000])
    ax.set_yticklabels(["10", "100", "1000", "1万", "10万"])
    ax.set_xlabel("")
    ax.set_ylabel("订单金额（元，对数刻度）")
    ax.set_title("各品类订单金额分布（箱线图）", fontsize=14, pad=12)
    ax.set_xticklabels(order, rotation=18, ha="right")
    med = df.groupby("category")["amount"].median()
    for i, c in enumerate(order):
        ax.text(i, med[c] * 1.35, f"中位 {med[c]:.0f}元", ha="center", fontsize=9,
                color=PALETTE["main"])
    return save_figure(fig, "图07_各品类金额箱线图")


# ---------------------------------------------------------------------
# 图 8：相关性热力图
# ---------------------------------------------------------------------

def fig08_correlation(df):
    cols = ["unit_price", "quantity", "discount", "amount", "rating_filled", "hour", "is_return"]
    names = ["单价", "数量", "优惠金额", "订单金额", "评分", "下单小时", "是否退货"]
    corr = df[cols].corr(method="spearman")
    corr.index, corr.columns = names, names

    fig, ax = plt.subplots(figsize=(8.6, 7.0))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.6, cbar_kws={"label": "Spearman 相关系数", "shrink": 0.82}, ax=ax)
    ax.set_title("主要变量相关性（Spearman 秩相关）", fontsize=14, pad=14)
    return save_figure(fig, "图08_相关性热力图")


# ---------------------------------------------------------------------
# 图 9：单价与订单金额关系
# ---------------------------------------------------------------------

def fig09_price_amount(df):
    sample = df.sample(min(40000, len(df)), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    hb = axes[0].hexbin(np.log10(sample["unit_price"].clip(lower=0.1)),
                        np.log10(sample["amount"].clip(lower=0.1)),
                        gridsize=52, cmap="viridis", bins="log", mincnt=1)
    axes[0].set_xticks([1, 2, 3, 4]), axes[0].set_xticklabels(["10", "100", "1000", "1万"])
    axes[0].set_yticks([1, 2, 3, 4, 5]), axes[0].set_yticklabels(["10", "100", "1000", "1万", "10万"])
    axes[0].set_xlabel("商品单价（元，对数刻度）")
    axes[0].set_ylabel("订单金额（元，对数刻度）")
    axes[0].set_title("单价与订单金额的联合分布", fontsize=14, pad=12)
    fig.colorbar(hb, ax=axes[0], label="订单数（对数）")

    g = df.groupby("category").agg(单价=("unit_price", "mean"), 客单价=("amount", "mean"),
                                   退货率=("is_return", "mean"))
    sc = axes[1].scatter(g["单价"], g["客单价"], s=g["退货率"] * 4200 + 120,
                         c=CAT_COLORS[:len(g)], alpha=0.82, edgecolor="white", lw=1.2)
    for name, row in g.iterrows():
        axes[1].annotate(name, (row["单价"], row["客单价"]), fontsize=11,
                         xytext=(7, 5), textcoords="offset points")
    axes[1].set_xlabel("平均商品单价（元）")
    axes[1].set_ylabel("平均客单价（元）")
    axes[1].set_title("品类价格定位图\n（气泡大小代表退货率）", fontsize=14, pad=12)
    axes[1].text(0.97, 0.04, f"退货率范围 {g['退货率'].min():.1%} — {g['退货率'].max():.1%}",
                 transform=axes[1].transAxes, ha="right", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.4", fc="#F7F7F7", ec=PALETTE["neutral"]))
    return save_figure(fig, "图09_单价与金额关系")


# ---------------------------------------------------------------------
# 图 10：评分与退货的关系
# ---------------------------------------------------------------------

def fig10_rating_return(df):
    rated = df.dropna(subset=["rating"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))

    g = rated.groupby("rating").agg(订单数=("order_id", "count"), 退货率=("is_return", "mean"))
    bars = axes[0].bar(g.index.astype(int), g["退货率"], color=PALETTE["accent3"],
                       alpha=0.85, width=0.55)
    for b, (rate, cnt) in zip(bars, zip(g["退货率"], g["订单数"])):
        axes[0].text(b.get_x() + b.get_width() / 2, rate, f"{rate:.1%}\n({cnt / 1000:.0f}k单)",
                     ha="center", va="bottom", fontsize=9.5)
    axes[0].set_xlabel("用户评分（星）")
    axes[0].set_ylabel("退货率")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0%}"))
    axes[0].set_ylim(0, g["退货率"].max() * 1.25)
    axes[0].set_title("不同评分的退货率", fontsize=14, pad=12)

    cats = df.groupby("category")["is_return"].mean().sort_values(ascending=False)
    bars2 = axes[1].barh(cats.index, cats.values, color=PALETTE["main"], alpha=0.87)
    for b, v in zip(bars2, cats.values):
        axes[1].text(v, b.get_y() + b.get_height() / 2, f" {v:.1%}", va="center", fontsize=10)
    axes[1].axvline(df["is_return"].mean(), color=PALETTE["accent3"], ls="--", lw=1.6,
                    label=f"整体退货率 {df['is_return'].mean():.1%}")
    axes[1].set_title("各品类退货率对比", fontsize=14, pad=12)
    axes[1].set_xlabel("退货率")
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0%}"))
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].set_xlim(0, cats.max() * 1.22)
    return save_figure(fig, "图10_评分与退货率")


# ---------------------------------------------------------------------
# 图 11：大促效应
# ---------------------------------------------------------------------

def fig11_promotion_effect(df):
    daily = df.groupby(["date", "is_promo"]).agg(
        订单量=("order_id", "count"), 销售额=("amount", "sum"),
        客单价=("amount", "mean")).reset_index()
    g = daily.groupby("is_promo").agg(日均订单量=("订单量", "mean"),
                                      日均销售额=("销售额", "mean"),
                                      平均客单价=("客单价", "mean"))
    g.index = ["平日", "大促"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    titles = ["日均订单量（单）", "日均销售额（元）", "平均客单价（元）"]
    keys = ["日均订单量", "日均销售额", "平均客单价"]
    for ax, key, title in zip(axes, keys, titles):
        vals = g[key]
        bars = ax.bar(vals.index, vals.values, color=[PALETTE["neutral"], PALETTE["accent"]],
                      width=0.5, alpha=0.9)
        for b, v in zip(bars, vals.values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom",
                    fontsize=11.5, color="#333333")
        diff = (vals["大促"] - vals["平日"]) / vals["平日"]
        ax.set_title(f"{title}\n大促相对平日 {diff:+.1%}", fontsize=13, pad=12)
        ax.set_ylim(0, vals.max() * 1.22)
        if key == "日均销售额":
            ax.yaxis.set_major_formatter(MONEY_FMT)
    fig.suptitle("大促活动的经营效果对比：以价换量", fontsize=15, y=1.03)
    return save_figure(fig, "图11_大促效应对比")


# ---------------------------------------------------------------------
# 图 12：分地区月度趋势（小倍数图）
# ---------------------------------------------------------------------

def fig12_region_trend(df):
    pv = df.pivot_table(index="month", columns="province", values="amount",
                        aggfunc="sum", fill_value=0)
    provinces = df.groupby("province")["amount"].sum().sort_values(ascending=False).index[:12]

    fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharex=True)
    for ax, p in zip(axes.ravel(), provinces):
        ax.plot(pv.index, pv[p], color=PALETTE["main"], lw=1.9, marker="o", ms=3.4)
        ax.fill_between(pv.index, 0, pv[p], color=PALETTE["main"], alpha=0.13)
        peak = int(pv[p].idxmax())
        ax.scatter([peak], [pv.loc[peak, p]], color=PALETTE["accent"], zorder=5, s=42)
        ax.set_title(p, fontsize=12.5)
        ax.set_xticks([1, 4, 7, 10])
        ax.set_xticklabels(["1月", "4月", "7月", "10月"], fontsize=9)
        ax.yaxis.set_major_formatter(MONEY_FMT)
        ax.tick_params(axis="y", labelsize=9)
    fig.suptitle("分地区月度销售额趋势（橙点标记该地区峰值月份）", fontsize=15, y=0.995)
    fig.supxlabel("月份")
    fig.supylabel("销售额（元）")
    return save_figure(fig, "图12_分地区月度趋势")


# ---------------------------------------------------------------------
# 图 13：原始数据缺失情况
# ---------------------------------------------------------------------

def fig13_missing_pattern():
    raw = pd.read_csv(RAW_FILE, dtype=str, nrows=200000)
    sample = raw.sample(3000, random_state=7)
    miss = sample.isna() | (sample.astype(str).apply(lambda s: s.str.strip() == ""))
    cols = [c for c in miss.columns if miss[c].any()] + \
           [c for c in miss.columns if not miss[c].any()]
    miss = miss[cols]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), gridspec_kw={"width_ratios": [1.5, 1]})
    axes[0].imshow(miss.T.values, aspect="auto", cmap="gray_r", interpolation="nearest")
    axes[0].set_yticks(range(len(cols)))
    axes[0].set_yticklabels(cols, fontsize=10)
    axes[0].set_xlabel("随机抽取的 3000 条原始记录")
    axes[0].set_title("原始数据缺失分布图\n（黑色为缺失，白色为正常）", fontsize=13.5, pad=12)
    axes[0].grid(False)

    rate = (raw.isna() | (raw.astype(str).apply(lambda s: s.str.strip() == ""))).mean().sort_values(ascending=False)
    rate = rate[rate > 0]
    bars = axes[1].barh(rate.index[::-1], rate.values[::-1] * 100, color=PALETTE["accent3"], alpha=0.85)
    for b, v in zip(bars, rate.values[::-1] * 100):
        axes[1].text(v, b.get_y() + b.get_height() / 2, f" {v:.1f}%", va="center", fontsize=10)
    axes[1].set_xlabel("缺失率（%）")
    axes[1].set_title("各字段缺失率", fontsize=13.5, pad=12)
    axes[1].set_xlim(0, max(rate.max() * 100 * 1.25, 10))
    axes[1].text(0.97, 0.05,
                 "注：promotion 为空表示该订单不属于任何大促，\n属于业务含义而非数据缺陷",
                 transform=axes[1].transAxes, ha="right", fontsize=9.5, color=PALETTE["neutral"])
    return save_figure(fig, "图13_原始数据缺失分布")


# ---------------------------------------------------------------------
# 图 14：客户分群结果
# ---------------------------------------------------------------------

def fig14_customer_segments(df):
    seg = pd.read_csv(Path(TABLE_DIR).parent.parent / "data/processed/customer_segments.csv",
                      dtype={"customer_id": str})
    profile = pd.read_csv(TABLE_DIR / "客户分群画像.csv")
    name_map = dict(zip(profile["分群"], profile["客户群命名"]))
    seg["客户群"] = seg["分群"].map(name_map)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), gridspec_kw={"width_ratios": [1.4, 1, 1]})

    for i, (name, part) in enumerate(seg.groupby("客户群")):
        part = part.sample(min(len(part), 2500), random_state=1)
        axes[0].scatter(np.log10(part["R"].clip(lower=0.5)), np.log10(part["M"].clip(lower=1)),
                        s=9, alpha=0.55, color=CAT_COLORS[i % len(CAT_COLORS)], label=name)
    axes[0].set_xticks([0, 1, 2]), axes[0].set_xticklabels(["1天", "10天", "100天"])
    axes[0].set_yticks([1, 2, 3, 4, 5]), axes[0].set_yticklabels(["10元", "100元", "1000元", "1万元", "10万元"])
    axes[0].set_xlabel("最近一次购买间隔 R（对数刻度）")
    axes[0].set_ylabel("消费总额 M（对数刻度）")
    axes[0].set_title("客户分群分布（R—M 平面）", fontsize=13.5, pad=12)
    axes[0].legend(fontsize=9, markerscale=2.2, loc="upper right")

    sizes = profile.set_index("客户群命名")["客户数"]
    axes[1].pie(sizes.values, labels=sizes.index, autopct="%1.1f%%", startangle=110,
                colors=CAT_COLORS[:len(sizes)], wedgeprops=dict(width=0.45, edgecolor="white"),
                textprops={"fontsize": 10})
    axes[1].set_title("各客户群人数占比", fontsize=13.5, pad=12)

    m = profile.set_index("客户群命名")["平均消费金额M"]
    bars = axes[2].bar(range(len(m)), m.values, color=CAT_COLORS[:len(m)], width=0.6)
    axes[2].set_xticks(range(len(m)))
    axes[2].set_xticklabels(m.index, rotation=20, ha="right", fontsize=10)
    for b, v in zip(bars, m.values):
        axes[2].text(b.get_x() + b.get_width() / 2, v, f"{v / 1e4:.1f}万", ha="center",
                     va="bottom", fontsize=10)
    axes[2].set_ylabel("人均年消费金额（元）")
    axes[2].yaxis.set_major_formatter(MONEY_FMT)
    axes[2].set_title("各客户群人均消费", fontsize=13.5, pad=12)
    axes[2].set_ylim(0, m.max() * 1.2)
    return save_figure(fig, "图14_客户分群结果")


# ---------------------------------------------------------------------
# 图 15：退货预测模型效果
# ---------------------------------------------------------------------

def fig15_model_roc():
    roc = pd.read_csv(TABLE_DIR / "退货预测_ROC曲线数据.csv")
    cm = pd.read_csv(TABLE_DIR / "退货预测_混淆矩阵.csv", index_col=0)
    metrics = pd.read_csv(TABLE_DIR / "退货预测_模型评价.csv")
    auc = metrics["AUC"].iat[0]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    axes[0].plot(roc["假正率FPR"], roc["真正率TPR"], color=PALETTE["main"], lw=2.4,
                 label=f"逻辑回归（AUC = {auc:.3f}）")
    axes[0].plot([0, 1], [0, 1], ls="--", color=PALETTE["neutral"], lw=1.4, label="随机猜测（AUC = 0.5）")
    axes[0].fill_between(roc["假正率FPR"], roc["真正率TPR"], roc["假正率FPR"],
                         color=PALETTE["main"], alpha=0.12)
    axes[0].set_xlabel("假正率 FPR")
    axes[0].set_ylabel("真正率 TPR")
    axes[0].set_title("退货预测模型的 ROC 曲线", fontsize=14, pad=12)
    axes[0].legend(loc="lower right", fontsize=10.5)

    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=axes[1],
                annot_kws={"fontsize": 13})
    axes[1].set_title("混淆矩阵（测试集）", fontsize=14, pad=12)
    axes[1].set_xlabel("模型预测")
    axes[1].set_ylabel("实际情况")
    return save_figure(fig, "图15_退货预测模型效果")


# ---------------------------------------------------------------------
# 图 16：模型特征重要性
# ---------------------------------------------------------------------

def fig16_feature_importance():
    ret_coef = pd.read_csv(TABLE_DIR / "退货预测_特征系数.csv").head(12).sort_values("影响强度")
    amt_coef = pd.read_csv(TABLE_DIR / "订单金额影响因素_系数.csv").head(12).sort_values("影响强度")
    rename = {"category_": "品类：", "channel_": "渠道：", "member_level_": "会员：",
              "province_": "省份：", "is_promo_": "是否大促："}

    def pretty(s):
        for k, v in rename.items():
            if s.startswith(k):
                return v + s[len(k):]
        return {"单价对数": "商品单价", "rating_filled": "用户评分", "quantity": "购买数量",
                "折扣率": "折扣率", "hour": "下单时段"}.get(s, s)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.0))
    labels = [pretty(x) for x in ret_coef["特征"]]
    colors = [PALETTE["accent3"] if v > 0 else PALETTE["accent2"] for v in ret_coef["标准化系数"]]
    axes[0].barh(labels, ret_coef["标准化系数"], color=colors, alpha=0.88)
    axes[0].axvline(0, color="#333333", lw=1)
    axes[0].set_title("退货预测模型：特征影响方向\n（红色提高退货概率，绿色降低）", fontsize=13.5, pad=12)
    axes[0].set_xlabel("标准化回归系数")

    labels2 = [pretty(x) for x in amt_coef["特征"]]
    axes[1].barh(labels2, amt_coef["标准化系数"], color=PALETTE["main"], alpha=0.88)
    axes[1].axvline(0, color="#333333", lw=1)
    axes[1].set_title("订单金额影响因素（岭回归标准化系数）", fontsize=13.5, pad=12)
    axes[1].set_xlabel("标准化回归系数")
    fig.tight_layout()
    return save_figure(fig, "图16_模型特征重要性")


# ---------------------------------------------------------------------
# 图 17：聚类簇数评估
# ---------------------------------------------------------------------

def fig17_cluster_evaluation():
    m = pd.read_csv(TABLE_DIR / "客户分群_簇数评估.csv")
    fig, ax1 = plt.subplots(figsize=(9.5, 5.2))
    ax1.plot(m["簇数k"], m["组内平方和SSE"], marker="o", color=PALETTE["main"], lw=2.2, label="组内平方和 SSE")
    ax1.set_xlabel("聚类簇数 k")
    ax1.set_ylabel("组内平方和 SSE", color=PALETTE["main"])
    ax1.axvline(4, color=PALETTE["accent"], ls="--", lw=1.6, label="选定 k = 4")

    ax2 = ax1.twinx()
    ax2.plot(m["簇数k"], m["轮廓系数"], marker="s", color=PALETTE["accent2"], lw=2.2,
             label="轮廓系数")
    ax2.set_ylabel("轮廓系数", color=PALETTE["accent2"])
    ax2.grid(False)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=10)
    ax1.set_title("客户分群簇数选择：肘部法与轮廓系数", fontsize=14, pad=12)
    ax1.text(0.03, 0.06,
             "轮廓系数在 k=2 时最高，但肘部法显示 k=4 之后\n"
             "SSE 下降明显放缓，且 4 类客户更利于差异化运营，\n故综合选定 k=4",
             transform=ax1.transAxes, fontsize=10, va="bottom",
             bbox=dict(boxstyle="round,pad=0.5", fc="#F7F9FB", ec=PALETTE["neutral"]))
    return save_figure(fig, "图17_聚类簇数评估")


# ---------------------------------------------------------------------
# 图 18：综合经营看板
# ---------------------------------------------------------------------

def fig18_dashboard(df):
    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.32)

    # 指标卡
    kpis = [
        ("总销售额", f"{df['amount'].sum() / 1e8:.2f} 亿元", PALETTE["main"]),
        ("总订单量", f"{len(df) / 1e4:.2f} 万单", PALETTE["accent"]),
        ("平均客单价", f"{df['amount'].mean():.0f} 元", PALETTE["accent2"]),
        ("整体退货率", f"{df['is_return'].mean():.2%}", PALETTE["accent3"]),
    ]
    for i, (title, value, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i])
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.9, transform=ax.transAxes,
                                   fc="#F7F9FB", ec=color, lw=1.6, zorder=1))
        ax.text(0.5, 0.63, title, ha="center", va="center", fontsize=12.5,
                color="#555555", transform=ax.transAxes, zorder=2)
        ax.text(0.5, 0.3, value, ha="center", va="center", fontsize=19,
                color=color, fontweight="bold", transform=ax.transAxes, zorder=2)

    # 月度销售额
    ax1 = fig.add_subplot(gs[1, :2])
    monthly = df.groupby("month_name", sort=False)["amount"].sum()
    ax1.bar(range(len(monthly)), monthly.values, color=PALETTE["main"], alpha=0.88, width=0.65)
    ax1.set_xticks(range(len(monthly)))
    ax1.set_xticklabels(monthly.index, fontsize=9)
    ax1.yaxis.set_major_formatter(MONEY_FMT)
    ax1.set_title("月度销售额", fontsize=13)
    peak = int(np.argmax(monthly.values))
    ax1.text(peak, monthly.values[peak], f"{monthly.values[peak] / 1e4:.0f}万", ha="center",
             va="bottom", fontsize=10, color=PALETTE["accent3"])

    # 渠道占比
    ax2 = fig.add_subplot(gs[1, 2])
    ch = df.groupby("channel")["amount"].sum()
    ax2.pie(ch.values, labels=ch.index, autopct="%1.0f%%", startangle=95,
            colors=CAT_COLORS[:len(ch)], textprops={"fontsize": 9})
    ax2.set_title("渠道销售额占比", fontsize=13)

    # 品类 TOP
    ax3 = fig.add_subplot(gs[1, 3])
    cg = df.groupby("category")["amount"].sum().sort_values().tail(6)
    ax3.barh(cg.index, cg.values, color=CAT_COLORS[:len(cg)], alpha=0.88)
    ax3.xaxis.set_major_formatter(MONEY_FMT)
    ax3.tick_params(axis="y", labelsize=9.5)
    ax3.set_title("品类销售额 TOP6", fontsize=13)

    # 星期效应
    ax4 = fig.add_subplot(gs[2, :2])
    daily = df.groupby("date").agg(销售额=("amount", "sum"), 星期=("weekday_name", "first"),
                                   周末=("is_weekend", "first")).reset_index()
    w = daily.groupby("星期")["销售额"].mean().reindex(
        ["周一", "周二", "周三", "周四", "周五", "周六", "周日"])
    ax4.bar(w.index, w.values, color=[PALETTE["neutral"]] * 5 + [PALETTE["accent"]] * 2, width=0.6, alpha=0.9)
    ax4.yaxis.set_major_formatter(MONEY_FMT)
    ax4.set_title("日均销售额的星期效应", fontsize=13)
    ax4.text(0.985, 0.9, f"周末较工作日高 "
                         f"{(daily[daily['周末']]['销售额'].mean() / daily[~daily['周末']]['销售额'].mean() - 1):.1%}",
             transform=ax4.transAxes, ha="right", fontsize=10, color=PALETTE["accent3"])

    # 金额分布
    ax5 = fig.add_subplot(gs[2, 2:])
    ax5.hist(np.log10(df["amount"].clip(lower=0.1)), bins=70, color=PALETTE["main"], alpha=0.85)
    ax5.set_xticks([0, 1, 2, 3, 4, 5])
    ax5.set_xticklabels(["1", "10", "100", "1000", "1万", "10万"], fontsize=9)
    ax5.set_title("订单金额分布（对数刻度）", fontsize=13)

    fig.suptitle("2025 年电商经营数据综合看板", fontsize=17, y=0.985)
    return save_figure(fig, "图18_综合经营看板")


# ---------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------

def main():
    print(f"[字体] 使用中文字体：{FONT}")
    df = add_time_features(load_clean())
    print(f"[读取] 数据 {len(df):,} 行，开始生成图表……\n")

    figures = [
        ("图01 全年销售额趋势", lambda: fig01_daily_trend(df)),
        ("图02 下单时段热力图", lambda: fig02_hour_weekday_heatmap(df)),
        ("图03 品类帕累托图", lambda: fig03_category_pareto(df)),
        ("图04 渠道结构与演变", lambda: fig04_channel_structure(df)),
        ("图05 地区销售对比", lambda: fig05_region(df)),
        ("图06 订单金额分布", lambda: fig06_amount_distribution(df)),
        ("图07 各品类金额箱线图", lambda: fig07_category_boxplot(df)),
        ("图08 相关性热力图", lambda: fig08_correlation(df)),
        ("图09 单价与金额关系", lambda: fig09_price_amount(df)),
        ("图10 评分与退货率", lambda: fig10_rating_return(df)),
        ("图11 大促效应对比", lambda: fig11_promotion_effect(df)),
        ("图12 分地区月度趋势", lambda: fig12_region_trend(df)),
        ("图13 原始数据缺失分布", lambda: fig13_missing_pattern()),
        ("图14 客户分群结果", lambda: fig14_customer_segments(df)),
        ("图15 退货预测模型效果", lambda: fig15_model_roc()),
        ("图16 模型特征重要性", lambda: fig16_feature_importance()),
        ("图17 聚类簇数评估", lambda: fig17_cluster_evaluation()),
        ("图18 综合经营看板", lambda: fig18_dashboard(df)),
    ]
    for name, fn in figures:
        path = fn()
        plt.close("all")
        print(f"[出图] {name:<22} -> {path.name}")

    print(f"\n[完成] 共 {len(figures)} 张图，输出目录：{FIG_DIR}")


if __name__ == "__main__":
    main()
