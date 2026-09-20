# -*- coding: utf-8 -*-
# 项目二：基于 Python 的群智能优化算法函数极值寻优系统
# 统一配置：路径、随机种子、中文字体、配色

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PART_DIR = RAW_DIR / "parts"
PROCESSED_DIR = DATA_DIR / "processed"
RESULT_DIR = ROOT / "results"
FIG_DIR = RESULT_DIR / "figures"
TABLE_DIR = RESULT_DIR / "tables"
REPORT_DIR = ROOT / "report"

for _d in (RAW_DIR, PART_DIR, PROCESSED_DIR, FIG_DIR, TABLE_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RUNS_FINAL = PROCESSED_DIR / "runs_final.csv"
RUNS_HISTORY = RAW_DIR / "runs_history.csv"
SUMMARY_FILE = PROCESSED_DIR / "summary.csv"

SEEDS = list(range(1, 31))
DIMS = [2, 10, 30]
MAX_ITER = 300
POP_SIZE = 50
HISTORY_STRIDE = {2: 2, 10: 5, 30: 10}
SUCCESS_TOL = 1e-2

# 绘图字体候选（macOS 自带中文字体）
FONT_CANDIDATES = [
    "Hiragino Sans GB", "Heiti TC", "Songti SC", "Arial Unicode MS", "STHeiti",
    "DejaVu Sans",       # 兜底：补全数学符号（如对数刻度中的负号）
]

PALETTE = {
    "main": "#2E5C8A",
    "accent": "#E8743B",
    "accent2": "#4C9F70",
    "accent3": "#B5484A",
    "neutral": "#8C8C8C",
}

# 8 种算法统一配色：所有图表中同一算法颜色保持一致
ALGO_COLORS = {
    "GA": "#2E5C8A",
    "PSO": "#E8743B",
    "APSO": "#C7A008",
    "DE": "#4C9F70",
    "GWO": "#8E6C9E",
    "SA": "#B5484A",
    "BFGS": "#3F8FA8",
    "NM": "#A2725E",
}

ALGO_CN = {
    "GA": "遗传算法",
    "PSO": "粒子群算法",
    "APSO": "自适应权重粒子群",
    "DE": "差分进化",
    "GWO": "灰狼优化",
    "SA": "模拟退火",
    "BFGS": "拟牛顿法 BFGS",
    "NM": "单纯形法 Nelder-Mead",
}

MAIN_ALGOS = ["GA", "PSO", "APSO", "DE", "GWO", "SA", "BFGS", "NM"]


def setup_chinese_font():
    """配置 matplotlib 中文字体，返回实际生效的字体名。"""
    import matplotlib
    from matplotlib import font_manager

    matplotlib.rcParams["font.sans-serif"] = FONT_CANDIDATES
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 110
    matplotlib.rcParams["savefig.dpi"] = 300
    matplotlib.rcParams["axes.edgecolor"] = "#4A4A4A"
    matplotlib.rcParams["axes.linewidth"] = 0.9
    matplotlib.rcParams["axes.grid"] = True
    matplotlib.rcParams["grid.alpha"] = 0.25
    matplotlib.rcParams["grid.linestyle"] = "--"
    matplotlib.rcParams["axes.axisbelow"] = True
    matplotlib.rcParams["mathtext.fontset"] = "dejavusans"

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in FONT_CANDIDATES:
        if name in available:
            return name
    return "unknown"


def save_figure(fig, name, also_pdf=True):
    """统一保存图片：300dpi PNG（报告插图）+ 矢量 PDF（打印）。"""
    fig.tight_layout()
    png = FIG_DIR / f"{name}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    if also_pdf:
        fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    return png
