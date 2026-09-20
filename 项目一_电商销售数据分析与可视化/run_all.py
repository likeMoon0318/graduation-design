# -*- coding: utf-8 -*-
# 一键复现全流程：数据生成 -> 清洗 -> 特征工程 -> 统计分析 -> 建模 -> 可视化
# =====================================================================
# 用法：
#   python run_all.py               # 完整跑一遍（含重新生成原始数据）
#   python run_all.py --from clean  # 从指定阶段开始（复用已有上游产物）
#   python run_all.py --skip generate,clean   # 跳过指定阶段
#   python run_all.py --list        # 查看阶段列表

import argparse
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

# matplotlib / 字体缓存写到临时目录，避免污染用户主目录
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_cache_project1"))

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import CLEAN_FILE, FIG_DIR, RAW_FILE, TABLE_DIR  # noqa: E402


def run_generate():
    import data_generator
    data_generator.main()


def run_clean():
    import clean
    clean.main()


def run_features():
    import features
    features.main()


def run_analyze():
    import analyze
    analyze.main()


def run_model():
    import model
    model.main()


def run_visualize():
    import visualize
    visualize.main()


# 阶段定义：(键名, 说明, 前置产物检查, 执行函数)
STAGES = [
    ("generate", "生成全年订单原始数据（含刻意注入的数据质量问题）", RAW_FILE, run_generate),
    ("clean", "数据清洗与质量报告", None, run_clean),
    ("features", "特征工程（时间特征 / RFM / 日粒度指标）", CLEAN_FILE, run_features),
    ("analyze", "描述统计、分组汇总与假设检验", None, run_analyze),
    ("model", "客户分群（KMeans）与订单金额回归、退货预测", None, run_model),
    ("visualize", "绘制全部图表（300dpi PNG + 矢量 PDF）", None, run_visualize),
]
STAGE_KEYS = [s[0] for s in STAGES]


def banner(text, char="="):
    print("\n" + char * 78)
    print(text)
    print(char * 78)


def summary():
    banner("产物清单", "-")
    figs = sorted(FIG_DIR.glob("*.png"))
    pdfs = sorted(FIG_DIR.glob("*.pdf"))
    tables = sorted(TABLE_DIR.glob("*.csv"))
    print(f"图表 PNG：{len(figs):>3} 个   矢量 PDF：{len(pdfs):>3} 个")
    print(f"结果表格：{len(tables):>3} 个")
    for p in (RAW_FILE, CLEAN_FILE):
        if p.exists():
            print(f"{p.relative_to(ROOT)}  ->  {p.stat().st_size / 1024 / 1024:.1f} MB")
    print("\n下一步：python src/dashboard.py  启动交互式看板")


def main():
    ap = argparse.ArgumentParser(description="电商销售数据分析项目 —— 一键复现")
    ap.add_argument("--from", dest="start", choices=STAGE_KEYS, default=STAGE_KEYS[0],
                    help="从该阶段开始执行（其后的阶段全部执行）")
    ap.add_argument("--skip", default="", help="要跳过的阶段，逗号分隔")
    ap.add_argument("--list", action="store_true", help="列出所有阶段后退出")
    args = ap.parse_args()

    if args.list:
        for i, (key, desc, _need, _fn) in enumerate(STAGES, 1):
            print(f"{i}. {key:<10} {desc}")
        return 0

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    unknown = skip - set(STAGE_KEYS)
    if unknown:
        print(f"[错误] 未知阶段：{', '.join(sorted(unknown))}")
        return 2

    banner("电商销售数据分析与可视化系统 —— 全流程复现")
    print(f"项目根目录：{ROOT}")
    print(f"Python：{sys.version.split()[0]}")
    print(f"计划执行：{' -> '.join(k for k in STAGE_KEYS if k not in skip)}")

    started = time.time()
    timings = []
    for key, desc, need, fn in STAGES:
        if key in skip:
            print(f"\n[跳过] {key}：{desc}")
            continue
        if key == args.start or STAGE_KEYS.index(key) > STAGE_KEYS.index(args.start):
            pass
        else:
            print(f"\n[跳过] {key}：早于起始阶段 --from {args.start}")
            continue
        if need is not None and not Path(need).exists():
            print(f"\n[中止] 阶段 {key} 需要前置文件 {Path(need).name}，请先执行上游阶段。")
            return 1
        banner(f"[阶段 {STAGE_KEYS.index(key) + 1}/{len(STAGES)}] {key} —— {desc}")
        t0 = time.time()
        try:
            fn()
        except Exception:                     # noqa: BLE001
            print(f"\n[失败] 阶段 {key} 执行出错：")
            traceback.print_exc()
            return 1
        dt = time.time() - t0
        timings.append((key, dt))
        print(f"[完成] {key} 用时 {dt:.1f} 秒")

    banner("执行汇总", "-")
    for key, dt in timings:
        print(f"  {key:<10} {dt:>7.1f} 秒")
    print(f"  {'合计':<10} {time.time() - started:>7.1f} 秒")
    summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
