# -*- coding: utf-8 -*-
# 交互式数据看板（Tkinter + matplotlib 嵌入式绘图）
# =====================================================================
# 运行： python src/dashboard.py
# 功能： 多维筛选（日期 / 地区 / 渠道 / 品类 / 会员等级）
#        指标卡实时刷新 -> 16 张图表下拉切换 -> 明细数据分页预览
#        当前图表导出 PNG、筛选结果导出 CSV

import os
import queue
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

# matplotlib 需要可写的缓存目录，避免在只读主目录下报字体缓存警告
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_cache_dashboard"))

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("TkAgg")          # 嵌入式绘图必须使用 TkAgg 后端
from matplotlib.backends.backend_tkagg import (  # noqa: E402
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (CAT_COLORS, CLEAN_FILE, PALETTE, REPORT_DIR,  # noqa: E402
                    setup_chinese_font)
from data_generator import PROMOTIONS  # noqa: E402
from features import TIME_SLOT_LABELS, add_time_features  # noqa: E402

FONT = setup_chinese_font()

# 界面配色
BG = "#F4F6F9"
CARD_BG = "#FFFFFF"
TEXT_MAIN = "#22303F"
TEXT_SUB = "#6B7A8C"
LINE = "#D9E1EA"
BTN_BG = PALETTE["main"]
BTN_FG = "#FFFFFF"
BTN_ACTIVE = "#3E7599"
ACCENT = PALETTE["accent"]
ACCENT2 = PALETTE["accent2"]
ACCENT3 = PALETTE["accent3"]

TABLE_COLUMNS = [
    ("order_id", "订单号", 130),
    ("order_time", "下单时间", 140),
    ("customer_id", "客户ID", 80),
    ("province", "省份", 60),
    ("city", "城市", 60),
    ("channel", "渠道", 70),
    ("category", "品类", 80),
    ("product", "商品", 110),
    ("unit_price", "单价", 60),
    ("quantity", "数量", 50),
    ("amount", "实付金额", 70),
    ("rating", "评分", 50),
    ("is_return", "退货", 50),
    ("member_level", "会员等级", 70),
]


def money_fmt(x, _pos=None):
    """金额轴格式化为万元 / 亿元。"""
    if abs(x) >= 1e8:
        return f"{x / 1e8:.1f}亿"
    if abs(x) >= 1e4:
        return f"{x / 1e4:.0f}万"
    return f"{x:.0f}"


MONEY = FuncFormatter(money_fmt)


# =====================================================================
# 图表库：每个函数接收筛选后的数据与一个 Axes，绘制完成后由画布渲染
# =====================================================================

def chart_trend(df, ax):
    daily = df.groupby("date").agg(销售额=("amount", "sum")).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["ma7"] = daily["销售额"].rolling(7, min_periods=1).mean()
    ax.plot(daily["date"], daily["销售额"], color=PALETTE["neutral"], lw=0.7, alpha=0.75,
            label="每日销售额")
    ax.plot(daily["date"], daily["ma7"], color=PALETTE["main"], lw=2.2, label="7 日移动平均")
    ax.fill_between(daily["date"], daily["ma7"], color=PALETTE["main"], alpha=0.10)
    ax.set_title("每日销售额趋势", fontsize=13, pad=10)
    ax.set_ylabel("销售额（元）")
    ax.yaxis.set_major_formatter(MONEY)
    ax.legend(loc="upper left", fontsize=9)


def chart_hour_weekday(df, ax):
    order = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    pivot = (df.pivot_table(index="weekday_name", columns="hour", values="amount", aggfunc="sum")
             .reindex(order).fillna(0) / 1e4)
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("下单小时")
    ax.set_title("下单时段 x 星期 销售额热力图（单位：万元）", fontsize=13, pad=10)
    cb = ax.figure.colorbar(im, ax=ax, pad=0.01)
    cb.ax.tick_params(labelsize=8)
    ax.grid(False)


def chart_pareto(df, ax):
    g = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    share = g / g.sum()
    ax.bar(range(len(g)), g.values / 1e4, color=CAT_COLORS[:len(g)])
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g.index, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("销售额（万元）")
    ax2 = ax.twinx()
    ax2.plot(range(len(g)), share.cumsum().values * 100, color=ACCENT, marker="o", lw=1.8)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("累计占比（%）", color=ACCENT)
    ax2.grid(False)
    ax2.axhline(80, color=PALETTE["neutral"], ls="--", lw=1)
    ax.set_title("品类销售额帕累托图", fontsize=13, pad=10)


def chart_channel_donut(df, ax):
    g = df.groupby("channel")["amount"].sum().sort_values(ascending=False)
    ax.pie(g.values, labels=g.index, autopct="%1.1f%%", startangle=90,
           colors=CAT_COLORS[:len(g)], pctdistance=0.78,
           wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.4),
           textprops=dict(fontsize=9))
    ax.text(0, 0, f"总销售额\n{g.sum() / 1e8:.2f} 亿元", ha="center", va="center",
            fontsize=11, color=PALETTE["main"])
    ax.set_title("渠道销售额结构", fontsize=13, pad=10)


def chart_channel_stack(df, ax):
    pv = df.pivot_table(index="month", columns="channel", values="amount", aggfunc="sum").fillna(0)
    ax.stackplot(pv.index, [pv[c].values / 1e4 for c in pv.columns], labels=pv.columns,
                 colors=CAT_COLORS[:pv.shape[1]], alpha=0.9)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([f"{m}月" for m in range(1, 13)], fontsize=9)
    ax.set_ylabel("销售额（万元）")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.set_title("各渠道销售额月度演变", fontsize=13, pad=10)


def chart_region_bar(df, ax):
    g = df.groupby("province")["amount"].sum().sort_values(ascending=False).head(12)
    bars = ax.barh(range(len(g))[::-1], g.values / 1e4,
                   color=[PALETTE["main"] if i < 3 else PALETTE["neutral"]
                          for i in range(len(g))])
    ax.set_yticks(range(len(g))[::-1])
    ax.set_yticklabels(g.index, fontsize=9)
    ax.set_xlabel("销售额（万元）")
    for b, v in zip(bars, g.values / 1e4):
        ax.text(b.get_width() + g.max() / 1e4 * 0.01, b.get_y() + b.get_height() / 2,
                f"{v:.0f}", va="center", fontsize=8, color=TEXT_SUB)
    ax.set_title("地区销售额排行（Top 12）", fontsize=13, pad=10)


def chart_city_bar(df, ax):
    g = df.groupby("city")["amount"].sum().sort_values(ascending=False).head(10)
    ax.bar(range(len(g)), g.values / 1e4, color=PALETTE["accent2"])
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("销售额（万元）")
    ax.set_title("城市销售额排行（Top 10）", fontsize=13, pad=10)


def chart_amount_hist(df, ax):
    data = df.loc[df["amount"] <= df["amount"].quantile(0.995), "amount"]
    ax.hist(data, bins=60, color=PALETTE["main"], alpha=0.72, density=True, edgecolor="white")
    if len(data) > 10:
        from scipy import stats
        kde = stats.gaussian_kde(data)
        xs = np.linspace(data.min(), data.max(), 300)
        ax.plot(xs, kde(xs), color=ACCENT, lw=2, label="核密度估计")
    ax.axvline(data.mean(), color=ACCENT3, ls="--", lw=1.6, label=f"均值 {data.mean():.1f} 元")
    ax.axvline(data.median(), color=PALETTE["accent2"], ls=":", lw=1.6,
               label=f"中位数 {data.median():.1f} 元")
    ax.set_xlabel("订单实付金额（元）")
    ax.set_ylabel("密度")
    ax.legend(fontsize=9)
    ax.set_title("订单金额分布（截尾至 99.5% 分位）", fontsize=13, pad=10)


def chart_category_box(df, ax):
    cats = df.groupby("category")["amount"].median().sort_values(ascending=False).index
    q99 = df["amount"].quantile(0.99)
    data = [df.loc[(df["category"] == c) & (df["amount"] <= q99), "amount"].values for c in cats]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
    for patch, color in zip(bp["boxes"], CAT_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for med in bp["medians"]:
        med.set_color("white")
        med.set_linewidth(1.6)
    ax.set_xticks(range(1, len(cats) + 1))
    ax.set_xticklabels(cats, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("订单金额（元）")
    ax.set_title("各品类订单金额分布", fontsize=13, pad=10)


def chart_corr(df, ax):
    cols = {"unit_price": "单价", "quantity": "数量", "discount": "优惠",
            "amount": "订单金额", "rating": "评分", "is_return": "退货"}
    sub = df[list(cols)].copy()
    sub["is_return"] = sub["is_return"].astype(int)
    corr = sub.corr(method="pearson")
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [cols[c] for c in corr.columns]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(corr.values[i, j]) > 0.55 else TEXT_MAIN)
    ax.grid(False)
    ax.figure.colorbar(im, ax=ax, pad=0.01)
    ax.set_title("核心指标相关系数矩阵", fontsize=13, pad=10)


def chart_price_scatter(df, ax):
    sub = df[df["amount"] <= df["amount"].quantile(0.99)]
    ax.scatter(sub["unit_price"], sub["amount"], s=5, alpha=0.18,
               color=PALETTE["main"], edgecolors="none")
    if len(sub) > 10:
        k, b = np.polyfit(sub["unit_price"], sub["amount"], 1)
        xs = np.linspace(sub["unit_price"].min(), sub["unit_price"].max(), 50)
        ax.plot(xs, k * xs + b, color=ACCENT, lw=2, label=f"拟合：y = {k:.2f}x + {b:.1f}")
        ax.legend(fontsize=9)
    ax.set_xlabel("商品单价（元）")
    ax.set_ylabel("订单金额（元）")
    ax.set_title("单价与订单金额关系", fontsize=13, pad=10)


def chart_rating_return(df, ax):
    tmp = df.dropna(subset=["rating"]).copy()
    if tmp.empty:
        raise ValueError("当前筛选下没有评分数据")
    tmp["rating_int"] = tmp["rating"].round().astype(int)
    g = tmp.groupby("rating_int").agg(退货率=("is_return", "mean"), 订单量=("order_id", "count"))
    ax.bar(g.index, g["退货率"].values * 100, color=ACCENT3, width=0.6)
    ax.set_xlabel("用户评分（分）")
    ax.set_ylabel("退货率（%）")
    ax.set_xticks(g.index)
    for x, y, n in zip(g.index, g["退货率"].values * 100, g["订单量"].values):
        ax.text(x, y + 0.15, f"{y:.1f}%\nn={n:,}", ha="center", fontsize=8, color=TEXT_SUB)
    ax.set_title("用户评分与退货率的关系", fontsize=13, pad=10)


def chart_promo_compare(df, ax):
    promo = df[df["is_promo"]]
    normal = df[~df["is_promo"]]
    avg = [promo["amount"].mean() if len(promo) else 0.0,
           normal["amount"].mean() if len(normal) else 0.0]
    cnt = [len(promo), len(normal)]
    x = np.arange(2)
    ax.bar(x - 0.2, avg, width=0.38, color=[ACCENT, PALETTE["main"]])
    ax.set_xticks(x)
    ax.set_xticklabels([f"大促期间\n({cnt[0]:,} 单)", f"平日\n({cnt[1]:,} 单)"])
    ax.set_ylabel("客单价（元）")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, [c / 1e4 for c in cnt], width=0.38, color=PALETTE["neutral"], alpha=0.65)
    ax2.set_ylabel("订单量（万单）", color=PALETTE["neutral"])
    ax2.grid(False)
    for xi, v in zip(x, avg):
        ax.text(xi - 0.2, v * 1.01, f"{v:.0f} 元", ha="center", fontsize=9, color=TEXT_MAIN)
    for xi, c in zip(x, cnt):
        ax2.text(xi + 0.2, c / 1e4 * 1.01, f"{c / 1e4:.1f} 万单", ha="center",
                 fontsize=9, color=TEXT_SUB)
    ax.set_title("大促与平日的客单价 / 订单量对比", fontsize=13, pad=10)


def chart_region_trend(df, ax):
    top = df.groupby("province")["amount"].sum().sort_values(ascending=False).head(6).index
    sub = df[df["province"].isin(top)]
    pv = sub.pivot_table(index="month", columns="province", values="amount", aggfunc="sum").fillna(0)
    pv = pv.reindex(range(1, 13)).fillna(0)
    for i, col in enumerate(pv.columns):
        ax.plot(pv.index, pv[col].values / 1e4, marker="o", ms=4, lw=1.8,
                color=CAT_COLORS[i % len(CAT_COLORS)], label=col)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([f"{m}月" for m in range(1, 13)], fontsize=9)
    ax.set_ylabel("销售额（万元）")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("重点地区销售额月度趋势", fontsize=13, pad=10)


def chart_member_bar(df, ax):
    order = ["普通会员", "银牌会员", "金牌会员", "钻石会员"]
    g = df.groupby("member_level").agg(客单价=("amount", "mean"), 订单量=("order_id", "count"))
    g = g.reindex([o for o in order if o in g.index])
    x = np.arange(len(g))
    ax.bar(x, g["客单价"].values, color=CAT_COLORS[:len(g)], width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(g.index, fontsize=9)
    ax.set_ylabel("客单价（元）")
    for xi, v, n in zip(x, g["客单价"].values, g["订单量"].values):
        ax.text(xi, v * 1.01, f"{v:.0f} 元\n{n:,} 单", ha="center", fontsize=8, color=TEXT_SUB)
    ax.set_title("各会员等级客单价对比", fontsize=13, pad=10)


def chart_time_slot(df, ax):
    g = (df.groupby("time_slot", observed=False)["amount"].sum()
         .reindex(TIME_SLOT_LABELS).fillna(0))
    ax.bar(range(len(g)), g.values / 1e4, color=ACCENT2, width=0.6)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g.index, fontsize=9)
    ax.set_ylabel("销售额（万元）")
    total = g.sum()
    for xi, v in zip(range(len(g)), g.values / 1e4):
        ax.text(xi, v * 1.01, f"{v:.0f}\n({v * 1e4 / total:.1%})", ha="center",
                fontsize=8, color=TEXT_SUB)
    ax.set_title("各时段销售额分布", fontsize=13, pad=10)


CHARTS = {
    "销售额趋势": chart_trend,
    "下单时段热力图": chart_hour_weekday,
    "品类帕累托图": chart_pareto,
    "渠道结构环形图": chart_channel_donut,
    "渠道趋势堆叠面积图": chart_channel_stack,
    "地区销售额排行": chart_region_bar,
    "城市销售额排行": chart_city_bar,
    "订单金额分布": chart_amount_hist,
    "各品类金额箱线图": chart_category_box,
    "相关性热力图": chart_corr,
    "单价与金额散点图": chart_price_scatter,
    "评分与退货率": chart_rating_return,
    "大促效应对比": chart_promo_compare,
    "重点地区月度趋势": chart_region_trend,
    "会员等级客单价": chart_member_bar,
    "时段销售分布": chart_time_slot,
}


# =====================================================================
# 主界面
# =====================================================================

class DashboardApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("电商销售数据分析与可视化系统 —— 交互式看板")
        self.geometry("1440x920")
        self.minsize(1200, 760)
        self.configure(bg=BG)

        self.df = None            # 全量数据
        self.view = None          # 筛选后的数据
        self.page = 0
        self.page_size = 50
        self.current_chart = tk.StringVar(value=list(CHARTS)[0])
        self._queue = queue.Queue()      # 后台线程 -> 主线程 的通信队列

        self._build_style()
        self._build_layout()
        self._load_data_async()
        self.after(80, self._poll_queue)

    # -------------------------------------------------- 样式
    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=TEXT_MAIN)
        style.configure("TButton", font=("PingFang SC", 10), padding=6)
        style.configure("Treeview", font=("PingFang SC", 9), rowheight=22,
                        background="white", fieldbackground="white")
        style.configure("Treeview.Heading", font=("PingFang SC", 9, "bold"))
        style.configure("TLabelframe", background=BG)
        style.configure("TLabelframe.Label", background=BG, foreground=PALETTE["main"],
                        font=("PingFang SC", 10, "bold"))

    # -------------------------------------------------- 布局
    def _build_layout(self):
        header = tk.Frame(self, bg=BTN_BG, height=54)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)
        tk.Label(header, text="  电商销售数据分析与可视化系统", bg=BTN_BG, fg="white",
                 font=("PingFang SC", 16, "bold")).pack(side="left")
        self.status = tk.Label(header, text="正在加载数据…", bg=BTN_BG, fg="#D5E3EF",
                               font=("PingFang SC", 10))
        self.status.pack(side="right", padx=16)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        self.sidebar = tk.Frame(body, bg=BG, width=232)
        self.sidebar.pack(side="left", fill="y", padx=(10, 4), pady=10)
        self.sidebar.pack_propagate(False)

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)

        self._build_sidebar()
        self._build_kpi(right)
        self._build_chart(right)
        self._build_table(right)

    def _build_sidebar(self):
        tk.Label(self.sidebar, text="数据筛选", bg=BG, fg=PALETTE["main"],
                 font=("PingFang SC", 12, "bold")).pack(anchor="w", pady=(0, 6))

        box = ttk.LabelFrame(self.sidebar, text="日期范围")
        box.pack(fill="x", pady=4)
        self.date_start = tk.StringVar()
        self.date_end = tk.StringVar()
        row = tk.Frame(box, bg=BG)
        row.pack(fill="x", padx=6, pady=4)
        self.cb_start = ttk.Combobox(row, textvariable=self.date_start, state="readonly",
                                     width=12, font=("PingFang SC", 9))
        self.cb_end = ttk.Combobox(row, textvariable=self.date_end, state="readonly",
                                   width=12, font=("PingFang SC", 9))
        self.cb_start.pack(side="left")
        tk.Label(row, text="→", bg=BG).pack(side="left", padx=3)
        self.cb_end.pack(side="left")
        tk.Label(box, text="提示：按住 Ctrl / Cmd 可多选", bg=BG, fg=TEXT_SUB,
                 font=("PingFang SC", 8)).pack(anchor="w", padx=8, pady=(0, 4))

        self.listboxes = {}
        for key, label, height in [("province", "地区（省份）", 6),
                                   ("channel", "购买渠道", 4),
                                   ("category", "商品品类", 6),
                                   ("member_level", "会员等级", 4)]:
            f = ttk.LabelFrame(self.sidebar, text=label)
            f.pack(fill="x", pady=4)
            lb = tk.Listbox(f, selectmode="extended", height=height, exportselection=False,
                            font=("PingFang SC", 9), activestyle="none", highlightthickness=0,
                            bg="white", selectbackground=PALETTE["main"], selectforeground="white")
            sb = ttk.Scrollbar(f, orient="vertical", command=lb.yview)
            lb.configure(yscrollcommand=sb.set)
            lb.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=4)
            sb.pack(side="right", fill="y", pady=4)
            self.listboxes[key] = lb

        btns = tk.Frame(self.sidebar, bg=BG)
        btns.pack(fill="x", pady=(8, 2))
        tk.Button(btns, text="应用筛选", command=self.apply_filters, bg=BTN_BG, fg=BTN_FG,
                  activebackground=BTN_ACTIVE, activeforeground="white", relief="flat",
                  font=("PingFang SC", 10, "bold"), pady=6).pack(fill="x", pady=2)
        tk.Button(btns, text="重置筛选", command=self.reset_filters, bg="#E3E9F0", fg=TEXT_MAIN,
                  activebackground="#D2DBE5", relief="flat",
                  font=("PingFang SC", 10), pady=5).pack(fill="x", pady=2)
        tk.Button(btns, text="导出筛选结果 CSV", command=self.export_csv, bg=ACCENT2, fg="white",
                  activebackground="#3E8A5F", relief="flat",
                  font=("PingFang SC", 10), pady=5).pack(fill="x", pady=2)
        tk.Button(btns, text="导出当前图表 PNG", command=self.export_png, bg=ACCENT, fg="white",
                  activebackground="#C9612C", relief="flat",
                  font=("PingFang SC", 10), pady=5).pack(fill="x", pady=2)

        # 大促信息（与数据生成模块共用同一套活动定义，保证口径一致）
        info = ttk.LabelFrame(self.sidebar, text="数据说明")
        info.pack(fill="both", expand=True, pady=(8, 0))
        txt = tk.Text(info, height=7, font=("PingFang SC", 8), bg="white", relief="flat",
                      wrap="word", fg=TEXT_SUB)
        txt.pack(fill="both", expand=True, padx=6, pady=4)
        txt.insert("1.0", "数据来源：程序仿真生成（随机种子固定，结果可复现）。\n"
                          f"促销活动共 {len(PROMOTIONS)} 个档期，"
                          "大促期间客单价与订单量均发生显著变化。\n"
                          "退货率、评分、缺失值等质量问题是刻意注入的，用于演示清洗流程。")
        txt.configure(state="disabled")

    def _build_kpi(self, parent):
        bar = tk.Frame(parent, bg=BG)
        bar.pack(fill="x")
        self.kpi_labels = {}
        specs = [("订单量", PALETTE["main"]), ("销售额", ACCENT),
                 ("客单价", ACCENT2), ("退货率", ACCENT3), ("消费客户数", "#8E6C9E")]
        for title, color in specs:
            card = tk.Frame(bar, bg=CARD_BG, highlightbackground=LINE, highlightthickness=1)
            card.pack(side="left", fill="both", expand=True, padx=4, pady=(0, 8))
            tk.Label(card, text=title, bg=CARD_BG, fg=TEXT_SUB,
                     font=("PingFang SC", 10)).pack(anchor="w", padx=12, pady=(8, 0))
            lab = tk.Label(card, text="—", bg=CARD_BG, fg=color,
                           font=("PingFang SC", 18, "bold"))
            lab.pack(anchor="w", padx=12, pady=(0, 8))
            self.kpi_labels[title] = lab

    def _build_chart(self, parent):
        head = tk.Frame(parent, bg=BG)
        head.pack(fill="x", pady=(2, 4))
        tk.Label(head, text="图表选择：", bg=BG, fg=TEXT_MAIN,
                 font=("PingFang SC", 10)).pack(side="left")
        cb = ttk.Combobox(head, textvariable=self.current_chart, values=list(CHARTS),
                          state="readonly", width=22, font=("PingFang SC", 10))
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda _e: self.draw_chart())

        wrap = tk.Frame(parent, bg=CARD_BG, highlightbackground=LINE, highlightthickness=1)
        wrap.pack(fill="both", expand=True)
        self.fig = Figure(figsize=(10.2, 4.9), dpi=100, facecolor="white")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=wrap)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, wrap, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")

    def _build_table(self, parent):
        box = ttk.LabelFrame(parent, text="明细数据预览")
        box.pack(fill="both", expand=False, pady=(8, 0))
        wrap = tk.Frame(box, bg=BG)
        wrap.pack(fill="both", expand=True, padx=4, pady=4)

        self.tree = ttk.Treeview(wrap, columns=[c[0] for c in TABLE_COLUMNS], show="headings",
                                 height=8)
        for key, title, width in TABLE_COLUMNS:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, anchor="center", stretch=False)
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(wrap, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)

        pager = tk.Frame(box, bg=BG)
        pager.pack(fill="x", pady=(0, 4))
        tk.Button(pager, text="◀ 上一页", command=lambda: self.change_page(-1),
                  relief="flat", bg="#E3E9F0", font=("PingFang SC", 9)).pack(side="left", padx=6)
        self.page_label = tk.Label(pager, text="第 0 / 0 页", bg=BG, fg=TEXT_SUB,
                                   font=("PingFang SC", 9))
        self.page_label.pack(side="left", padx=8)
        tk.Button(pager, text="下一页 ▶", command=lambda: self.change_page(1),
                  relief="flat", bg="#E3E9F0", font=("PingFang SC", 9)).pack(side="left")

    # -------------------------------------------------- 数据加载
    def _poll_queue(self):
        """在主线程中消费后台线程的消息（Tkinter 不允许跨线程操作控件）。"""
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "data":
                    self.df = payload
                    self._after_load()
                elif kind == "error":
                    messagebox.showerror("数据加载失败", payload)
                    self.status.configure(text="数据加载失败")
        except queue.Empty:
            pass
        self.after(80, self._poll_queue)

    def _load_data_async(self):
        def worker():
            try:
                df = pd.read_csv(CLEAN_FILE, dtype={"order_id": str, "customer_id": str},
                                 parse_dates=["order_time"])
                df = add_time_features(df)
                self._queue.put(("data", df))
            except Exception as exc:      # noqa: BLE001
                self._queue.put(("error", f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _after_load(self):
        df = self.df
        date_strs = [str(d) for d in sorted(df["date"].unique())]
        self.cb_start["values"] = date_strs
        self.cb_end["values"] = date_strs
        self.date_start.set(date_strs[0])
        self.date_end.set(date_strs[-1])
        for key, lb in self.listboxes.items():
            if key == "province":
                vals = df["province"].value_counts().index.tolist()
            elif key == "member_level":
                vals = ["普通会员", "银牌会员", "金牌会员", "钻石会员"]
            else:
                vals = sorted(df[key].unique())
            lb.delete(0, "end")
            for v in vals:
                lb.insert("end", v)
            lb.selection_set(0, "end")
        self.view = df
        self.refresh_all()

    # -------------------------------------------------- 筛选逻辑
    def _selected(self, key):
        lb = self.listboxes[key]
        return [lb.get(i) for i in lb.curselection()]

    def apply_filters(self):
        if self.df is None:
            return
        df = self.df
        s, e = pd.Timestamp(self.date_start.get()), pd.Timestamp(self.date_end.get())
        if s > e:
            s, e = e, s
            self.date_start.set(str(s.date()))
            self.date_end.set(str(e.date()))
        mask = (df["order_time"] >= s) & (df["order_time"] < e + pd.Timedelta(days=1))
        for key in self.listboxes:
            vals = self._selected(key)
            if vals:
                mask &= df[key].isin(vals)
        self.view = df[mask]
        self.page = 0
        self.refresh_all()

    def reset_filters(self):
        if self.df is None:
            return
        date_strs = [str(d) for d in sorted(self.df["date"].unique())]
        self.date_start.set(date_strs[0])
        self.date_end.set(date_strs[-1])
        for lb in self.listboxes.values():
            lb.selection_set(0, "end")
        self.view = self.df
        self.page = 0
        self.refresh_all()

    # -------------------------------------------------- 刷新
    def refresh_all(self):
        self.update_kpi()
        self.draw_chart()
        self.fill_table()

    def update_kpi(self):
        df = self.view
        if df is None or df.empty:
            for lab in self.kpi_labels.values():
                lab.configure(text="—")
            self.status.configure(text="当前筛选结果为空，请放宽筛选条件")
            return
        amount = df["amount"].sum()
        self.kpi_labels["订单量"].configure(text=f"{len(df):,}")
        self.kpi_labels["销售额"].configure(
            text=f"{amount / 1e8:.2f} 亿" if amount >= 1e8 else f"{amount / 1e4:.1f} 万")
        self.kpi_labels["客单价"].configure(text=f"{df['amount'].mean():.0f} 元")
        self.kpi_labels["退货率"].configure(text=f"{df['is_return'].mean():.2%}")
        self.kpi_labels["消费客户数"].configure(text=f"{df['customer_id'].nunique():,}")
        self.status.configure(text=f"当前筛选：{len(df):,} 条订单（全量 {len(self.df):,} 条）"
                                   f"　销售额 {amount / 1e8:.2f} 亿元")

    def draw_chart(self):
        if self.view is None:
            return
        name = self.current_chart.get()
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        if self.view.empty:
            ax.text(0.5, 0.5, "当前筛选条件下没有数据\n请放宽筛选范围",
                    ha="center", va="center", fontsize=14, color=TEXT_SUB)
            ax.axis("off")
        else:
            try:
                CHARTS[name](self.view, ax)
            except Exception as exc:      # noqa: BLE001
                ax.axis("off")
                ax.text(0.5, 0.5, f"该图表在当前筛选下无法绘制：\n{exc}",
                        ha="center", va="center", fontsize=11, color=ACCENT3)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # -------------------------------------------------- 明细表
    def fill_table(self):
        self.tree.delete(*self.tree.get_children())
        df = self.view
        if df is None or df.empty:
            self.page_label.configure(text="第 0 / 0 页")
            return
        total_pages = (len(df) - 1) // self.page_size + 1
        self.page = max(0, min(self.page, total_pages - 1))
        chunk = df.iloc[self.page * self.page_size:(self.page + 1) * self.page_size]
        for _, r in chunk.iterrows():
            self.tree.insert("", "end", values=(
                r["order_id"], f"{r['order_time']:%Y-%m-%d %H:%M}", r["customer_id"],
                r["province"], r["city"], r["channel"], r["category"], r["product"],
                f"{r['unit_price']:.2f}", int(r["quantity"]), f"{r['amount']:.2f}",
                "—" if pd.isna(r["rating"]) else f"{r['rating']:.1f}",
                "是" if r["is_return"] else "否", r["member_level"]))
        self.page_label.configure(text=f"第 {self.page + 1} / {total_pages} 页"
                                       f"（共 {len(df):,} 条，每页 {self.page_size} 条）")

    def change_page(self, delta):
        if self.view is None or self.view.empty:
            return
        self.page += delta
        self.fill_table()

    # -------------------------------------------------- 导出
    def export_png(self):
        if self.view is None:
            return
        name = self.current_chart.get()
        path = filedialog.asksaveasfilename(
            title="导出当前图表", defaultextension=".png",
            initialfile=f"{name}.png", filetypes=[("PNG 图片", "*.png")],
            initialdir=str(REPORT_DIR))
        if not path:
            return
        self.fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        messagebox.showinfo("导出成功", f"图表已保存至：\n{path}")

    def export_csv(self):
        if self.view is None or self.view.empty:
            messagebox.showwarning("无数据", "当前筛选结果为空，无法导出。")
            return
        path = filedialog.asksaveasfilename(
            title="导出筛选结果", defaultextension=".csv",
            initialfile="筛选结果.csv", filetypes=[("CSV 文件", "*.csv")],
            initialdir=str(REPORT_DIR))
        if not path:
            return
        out = self.view.drop(columns=["date", "time_slot"], errors="ignore")
        out.to_csv(path, index=False, encoding="utf-8-sig")
        messagebox.showinfo("导出成功", f"已导出 {len(out):,} 条记录至：\n{path}")


def main():
    app = DashboardApp()
    app.mainloop()


if __name__ == "__main__":
    main()
