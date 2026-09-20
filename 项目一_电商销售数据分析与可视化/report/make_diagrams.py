# -*- coding: utf-8 -*-
# 生成报告用的流程示意图（技术路线图、数据质量处理流程图）
# 运行： python report/make_diagrams.py

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_cache_report1"))

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


def box(ax, x, y, w, h, text, fc, ec, fs=10, tc="white", bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=3, fontweight="bold" if bold else "normal", linespacing=1.5)


def arrow(ax, p1, p2, color="#6B7A8C", style="-|>", lw=1.3, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12,
                                 linewidth=lw, color=color, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))


def fig_roadmap():
    fig, ax = plt.subplots(figsize=(11.2, 7.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    main, accent, green, red = (PALETTE["main"], PALETTE["accent"],
                                PALETTE["accent2"], PALETTE["accent3"])
    rows = [
        ("数据层", accent, [
            "业务规则设计",
            "全年订单仿真生成\n(29.1 万条)",
            "刻意注入数据质量问题\n(重复/缺失/异常/格式混乱)",
            "原始数据落盘\norders_raw.csv"]),
        ("处理层", main, [
            "去重与类型修正",
            "缺失值与异常值处理\n(IQR / 业务勾稽)",
            "特征工程\n(时间特征 + RFM)",
            "数据质量报告\n(清洗前后对比)"]),
        ("分析层", green, [
            "描述性统计与分组汇总",
            "7 项假设检验\n(t / 方差分析 / 卡方 / 秩和)",
            "客户分群 KMeans\n订单金额回归 退货预测",
            "18 张图表\n(300dpi PNG + 矢量 PDF)"]),
        ("应用层", red, [
            "交互式看板\n(多维筛选)",
            "16 种图表实时重绘",
            "明细数据分页预览",
            "导出 PNG / CSV"]),
    ]

    y = 0.845
    for name, color, items in rows:
        ax.text(0.012, y + 0.055, name, fontsize=12, color=color, fontweight="bold",
                rotation=90, va="center", ha="center")
        xs = [0.075, 0.305, 0.535, 0.765]
        for i, (x, text) in enumerate(zip(xs, items)):
            box(ax, x, y, 0.20, 0.115, text, color, color, fs=9.5)
            if i < 3:
                arrow(ax, (x + 0.20, y + 0.0575), (xs[i + 1], y + 0.0575))
        y -= 0.208

    # 行间下行箭头
    for y0 in (0.845 - 0.0575, 0.637 - 0.0575 + 0.115, 0.429 - 0.0575 + 0.115):
        pass
    arrow(ax, (0.865, 0.845), (0.865, 0.752), rad=-0.35, color="#8C8C8C")
    arrow(ax, (0.178, 0.637), (0.178, 0.544), rad=0.35, color="#8C8C8C")
    arrow(ax, (0.178, 0.429), (0.178, 0.336), rad=0.35, color="#8C8C8C")

    ax.text(0.5, 0.045,
            "全流程由 run_all.py 一键复现；随机种子固定为 20250101，所有结果可重现",
            ha="center", fontsize=10, color="#4A4A4A")
    ax.set_title("电商销售数据分析与可视化系统 技术路线", fontsize=14, pad=10)
    fig.tight_layout()
    path = OUT / "技术路线图.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_quality_flow():
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    main, accent, green = PALETTE["main"], PALETTE["accent"], PALETTE["accent2"]

    box(ax, 0.02, 0.62, 0.20, 0.22,
        "原始数据 291,045 条\n重复 1,735 行\n缺失/异常/格式混乱", accent, accent, fs=9.5)
    steps = [
        ("① 去重与规范化", "删除完全重复行\n文本去空格、统一大小写", 0.26),
        ("② 类型与格式修复", "order_time 解析\namount 剥离货币符号", 0.44),
        ("③ 缺失值处理", "地区按业务规则补全\n评分缺失保留并标记", 0.62),
        ("④ 异常值与勾稽", "IQR 识别异常金额\n数量按金额反推修正", 0.80),
    ]
    for title, desc, x in steps:
        box(ax, x, 0.62, 0.155, 0.22, f"{title}\n\n{desc}", main, main, fs=8.6)
    for x0 in (0.22, 0.415, 0.595, 0.775):
        arrow(ax, (x0, 0.73), (x0 + 0.04, 0.73))

    box(ax, 0.30, 0.20, 0.40, 0.20,
        "清洗后数据 289,310 条\n列出「问题数量 → 处理方式」对照表\n"
        "校验项全部通过（重复 0 / 缺失 0 / 勾稽不符 0）",
        green, green, fs=10)
    arrow(ax, (0.5, 0.62), (0.5, 0.41))
    arrow(ax, (0.12, 0.62), (0.12, 0.30), rad=0.25)
    ax.plot([0.12, 0.30], [0.30, 0.30], color="#6B7A8C", lw=1.3, zorder=1)
    ax.text(0.115, 0.36, "质量报告\n(清洗前后对比)", fontsize=8, color="#4A4A4A",
            ha="center", va="bottom")

    ax.set_title("数据质量问题识别与处理流程", fontsize=14, pad=10)
    fig.tight_layout()
    path = OUT / "数据质量处理流程.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main():
    for fn in (fig_roadmap, fig_quality_flow):
        print("[生成]", fn().relative_to(ROOT))


if __name__ == "__main__":
    main()
