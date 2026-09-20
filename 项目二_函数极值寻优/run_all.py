# -*- coding: utf-8 -*-
# 一键复现全流程：批量实验 -> 统计分析 -> 参数敏感性 -> 可视化
# =====================================================================
# 用法：
#   python run_all.py                     # 完整复现（实验部分自动并行）
#   python run_all.py --from analyze      # 复用已有实验数据，只做后面的步骤
#   python run_all.py --force             # 忽略断点，重新跑全部实验
#   python run_all.py --list              # 查看阶段列表

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_cache_project2"))

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from config import FIG_DIR, RUNS_FINAL, SUMMARY_FILE, TABLE_DIR  # noqa: E402

# 阶段：(键名, 说明, 依赖产物, 执行的模块, 附加参数)
STAGES = [
    ("experiment", "批量实验（12 函数 x 8~10 算法 x 30 次独立运行）",
     None, "experiment.py", []),
    ("analyze", "统计分析与显著性检验（汇总 / Friedman / Wilcoxon）",
     RUNS_FINAL, "stats_analysis.py", []),
    ("sensitivity", "参数敏感性实验（交叉率、变异率、惯性权重等）",
     None, "sensitivity.py", []),
    ("visualize", "绘制全部图表（300dpi PNG + 矢量 PDF）",
     RUNS_FINAL, "visualize.py", []),
]
STAGE_KEYS = [s[0] for s in STAGES]


def banner(text, char="="):
    print("\n" + char * 78)
    print(text)
    print(char * 78)


def run_stage(module, extra):
    cmd = [sys.executable, str(SRC / module)] + extra
    print(f"$ {' '.join(str(c) for c in cmd)}")
    return subprocess.call(cmd, cwd=str(ROOT))


def summary():
    banner("产物清单", "-")
    pngs = sorted(FIG_DIR.glob("*.png"))
    pdfs = sorted(FIG_DIR.glob("*.pdf"))
    tables = sorted(TABLE_DIR.glob("*.csv"))
    print(f"图表 PNG：{len(pngs):>3} 个    矢量 PDF：{len(pdfs):>3} 个")
    print(f"结果表格：{len(tables):>3} 个")
    for p in (RUNS_FINAL, SUMMARY_FILE):
        if p.exists():
            print(f"{p.relative_to(ROOT)}  ->  {p.stat().st_size / 1024 / 1024:.1f} MB")
    print("\n下一步：python src/simulator.py   启动寻优过程动画演示器")


def main():
    ap = argparse.ArgumentParser(description="函数极值寻优项目 —— 一键复现")
    ap.add_argument("--from", dest="start", choices=STAGE_KEYS, default=STAGE_KEYS[0],
                    help="从该阶段开始执行（其后的阶段全部执行）")
    ap.add_argument("--skip", default="", help="要跳过的阶段，逗号分隔")
    ap.add_argument("--force", action="store_true", help="实验阶段忽略断点，全部重跑")
    ap.add_argument("--jobs", type=int, default=0, help="批量实验的并行进程数")
    ap.add_argument("--list", action="store_true", help="列出阶段后退出")
    args = ap.parse_args()

    if args.list:
        for i, (key, desc, _need, _m, _e) in enumerate(STAGES, 1):
            print(f"{i}. {key:<12} {desc}")
        return 0

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    if skip - set(STAGE_KEYS):
        print(f"[错误] 未知阶段：{', '.join(sorted(skip - set(STAGE_KEYS)))}")
        return 2

    banner("群智能优化算法函数极值寻优系统 —— 全流程复现")
    print(f"项目根目录：{ROOT}")
    print(f"Python：{sys.version.split()[0]}")
    print(f"计划执行：{' -> '.join(k for k in STAGE_KEYS if k not in skip)}")

    timings = []
    started = time.time()
    for key, desc, need, module, _extra in STAGES:
        idx = STAGE_KEYS.index(key)
        if key in skip:
            print(f"\n[跳过] {key}：{desc}")
            continue
        if idx < STAGE_KEYS.index(args.start):
            print(f"\n[跳过] {key}：早于起始阶段 --from {args.start}")
            continue
        if need is not None and not Path(need).exists():
            print(f"\n[中止] 阶段 {key} 需要前置文件 {Path(need).name}，请先执行上游阶段。")
            return 1
        banner(f"[阶段 {idx + 1}/{len(STAGES)}] {key} —— {desc}")
        extra = []
        if key == "experiment":
            if args.force:
                extra.append("--force")
            if args.jobs > 0:
                extra += ["--jobs", str(args.jobs)]
        t0 = time.time()
        code = run_stage(module, extra)
        if code != 0:
            print(f"\n[失败] 阶段 {key} 返回码 {code}，已中止。")
            return code
        dt = time.time() - t0
        timings.append((key, dt))
        print(f"[完成] {key} 用时 {dt:.1f} 秒")

    banner("执行汇总", "-")
    for key, dt in timings:
        print(f"  {key:<12} {dt:>7.1f} 秒")
    print(f"  {'合计':<12} {time.time() - started:>7.1f} 秒")
    summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
