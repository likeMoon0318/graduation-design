# -*- coding: utf-8 -*-
# 项目一：基于 Python 的电商销售数据分析与可视化系统
# 统一配置：路径、随机种子、中文字体、配色方案

from pathlib import Path

# ---------------------------------------------------------------- 路径配置
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULT_DIR = ROOT / "results"
FIG_DIR = RESULT_DIR / "figures"
TABLE_DIR = RESULT_DIR / "tables"
REPORT_DIR = ROOT / "report"

for _d in (RAW_DIR, PROCESSED_DIR, FIG_DIR, TABLE_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 文件路径
RAW_FILE = RAW_DIR / "orders_raw.csv"
CLEAN_FILE = PROCESSED_DIR / "orders_clean.csv"
CUSTOMER_FILE = PROCESSED_DIR / "customer_rfm.csv"
QUALITY_FILE = TABLE_DIR / "data_quality_report.csv"

# ---------------------------------------------------------------- 随机种子
# 全流程固定随机种子，保证结果完全可复现
RANDOM_SEED = 20250101

# ---------------------------------------------------------------- 绘图配置
FONT_CANDIDATES = [
    "Hiragino Sans GB",   # 冬青黑体简体中文
    "Heiti TC",
    "Songti SC",
    "Arial Unicode MS",
    "STHeiti",
]

# 图表统一配色，保证所有图与报告视觉一致
PALETTE = {
    "main": "#2E5C8A",      # 主色：深蓝
    "accent": "#E8743B",    # 强调色：橙
    "accent2": "#4C9F70",   # 辅助色：绿
    "accent3": "#B5484A",   # 辅助色：红
    "neutral": "#8C8C8C",   # 中性灰
}

# 分类色板（品类 / 渠道 / 分群通用）
CAT_COLORS = [
    "#2E5C8A", "#E8743B", "#4C9F70", "#B5484A",
    "#8E6C9E", "#C7A008", "#3F8FA8", "#A2725E",
]


def setup_chinese_font():
    """配置 matplotlib 中文字体，避免图中中文显示为方块。返回实际生效的字体名。"""
    import matplotlib
    from matplotlib import font_manager

    matplotlib.rcParams["font.sans-serif"] = FONT_CANDIDATES
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False   # 负号正常显示
    matplotlib.rcParams["figure.dpi"] = 110
    matplotlib.rcParams["savefig.dpi"] = 300
    matplotlib.rcParams["axes.edgecolor"] = "#4A4A4A"
    matplotlib.rcParams["axes.linewidth"] = 0.9
    matplotlib.rcParams["axes.grid"] = True
    matplotlib.rcParams["grid.alpha"] = 0.25
    matplotlib.rcParams["grid.linestyle"] = "--"
    matplotlib.rcParams["axes.axisbelow"] = True

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in FONT_CANDIDATES:
        if name in available:
            return name
    return "unknown"


def save_figure(fig, name, also_pdf=True):
    """统一保存图片：300dpi PNG + 矢量 PDF，供报告插图使用。"""
    fig.tight_layout()
    png = FIG_DIR / f"{name}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    if also_pdf:
        fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    return png
