# -*- coding: utf-8 -*-
# 寻优过程演示器（Tkinter + matplotlib 嵌入式动画）
# =====================================================================
# 运行： python src/simulator.py
# 功能：
#   1) 选择测试函数与维度，实时显示对应的等高线 / 曲面切片与理论最优位置
#   2) 选择算法并现场调整参数（种群规模、迭代次数、交叉率、变异率、惯性权重等）
#   3) 点击「开始寻优」后以动画方式演示群体在搜索空间中的移动轨迹，
#      同步绘制收敛曲线（对数误差）
#   4) 显示本次运行的最优解、最优值、与理论最优的误差、评价次数与耗时
#   5) 多次运行结果可叠加对比（多算法收敛曲线同图 + 结果对比表 + 导出）

import os
import sys
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_cache_simulator"))

import numpy as np

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (  # noqa: E402
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import ALGO_CN, ALGO_COLORS, PALETTE, REPORT_DIR, setup_chinese_font  # noqa: E402
from algorithms import ALGORITHMS, DEFAULT_PARAMS  # noqa: E402
from functions import BENCHMARKS, grid_surface  # noqa: E402

FONT = setup_chinese_font()

BG = "#F4F6F9"
CARD_BG = "#FFFFFF"
TEXT_MAIN = "#22303F"
TEXT_SUB = "#6B7A8C"
LINE = "#D9E1EA"
BTN_BG = PALETTE["main"]
ACCENT = PALETTE["accent"]
ACCENT2 = PALETTE["accent2"]
ACCENT3 = PALETTE["accent3"]

# 各算法在界面上暴露的可调参数： (参数名, 中文标签, 最小值, 最大值, 小数位)
ALGO_PARAMS = {
    "GA": [("pc", "交叉概率 pc", 0.0, 1.0, 2),
           ("pm", "变异概率 pm", 0.0, 0.5, 3),
           ("eta_c", "交叉分布指数", 1.0, 30.0, 1),
           ("eta_m", "变异分布指数", 1.0, 40.0, 1),
           ("elite", "精英个体数", 0, 6, 0)],
    "PSO": [("w", "惯性权重 w", 0.0, 1.2, 3),
            ("c1", "个体学习因子 c1", 0.0, 3.0, 3),
            ("c2", "社会学习因子 c2", 0.0, 3.0, 3)],
    "APSO": [("w0", "初始惯性权重", 0.0, 1.2, 2),
             ("w1", "终止惯性权重", 0.0, 1.2, 2),
             ("c1_0", "初始 c1", 0.0, 3.0, 2),
             ("c1_1", "终止 c1", 0.0, 3.0, 2)],
    "DE": [("F", "缩放因子 F", 0.0, 1.5, 2),
           ("CR", "交叉概率 CR", 0.0, 1.0, 2)],
    "GWO": [],
    "SA": [("T1_ratio", "终止温度比例", 1e-8, 1e-1, 8),
           ("step_ratio", "邻域步长比例", 0.001, 0.5, 3),
           ("cool_power", "降温半径指数", 0.05, 1.0, 2)],
    "BFGS": [],
    "NM": [],
}


def log_fmt(v, _pos=None):
    """对数刻度标签使用 ASCII 科学计数法，避免字体缺少数学负号。"""
    if v <= 0:
        return ""
    e = np.log10(v)
    if abs(e - round(e)) > 1e-9:
        return ""
    return f"1e{int(round(e))}"


class SimulatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("群智能优化算法函数极值寻优演示器")
        self.geometry("1520x950")
        self.minsize(1280, 820)
        self.configure(bg=BG)

        self.result = None            # 最近一次运行结果
        self.frames = []              # 动画帧
        self.frame_idx = 0
        self.playing = False
        self.compare = []             # 叠加对比的结果列表
        self.surface_cache = {}

        self.algo = tk.StringVar(value="PSO")
        self.func_key = tk.StringVar(value="rastrigin")
        self.dim = tk.IntVar(value=2)
        self.pop_size = tk.IntVar(value=50)
        self.max_iter = tk.IntVar(value=150)
        self.seed = tk.IntVar(value=1)
        self.speed = tk.IntVar(value=40)
        self.param_vars = {}

        self._build_style()
        self._build_layout()
        self._on_func_change()
        self._on_algo_change()
        self.after(200, self.redraw_space)

    # -------------------------------------------------- 样式
    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=TEXT_MAIN)
        style.configure("TCheckbutton", background=BG)
        style.configure("TLabelframe", background=BG)
        style.configure("TLabelframe.Label", background=BG, foreground=BTN_BG,
                        font=("PingFang SC", 10, "bold"))
        style.configure("Treeview", font=("PingFang SC", 9), rowheight=21,
                        background="white", fieldbackground="white")
        style.configure("Treeview.Heading", font=("PingFang SC", 9, "bold"))

    # -------------------------------------------------- 布局
    def _build_layout(self):
        header = tk.Frame(self, bg=BTN_BG, height=54)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="  群智能优化算法函数极值寻优演示器", bg=BTN_BG, fg="white",
                 font=("PingFang SC", 16, "bold")).pack(side="left")
        self.status = tk.Label(header, text="就绪：选择函数与算法后点击「开始寻优」",
                               bg=BTN_BG, fg="#D5E3EF", font=("PingFang SC", 10))
        self.status.pack(side="right", padx=16)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        side = tk.Frame(body, bg=BG, width=300)
        side.pack(side="left", fill="y", padx=(10, 4), pady=10)
        side.pack_propagate(False)
        self._build_sidebar(side)

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)
        self._build_canvas(right)
        self._build_compare_table(right)

    def _build_sidebar(self, side):
        # ---- 问题设置
        f1 = ttk.LabelFrame(side, text="问题设置")
        f1.pack(fill="x", pady=4)
        tk.Label(f1, text="测试函数", bg=BG, font=("PingFang SC", 9)).grid(
            row=0, column=0, sticky="w", padx=6, pady=3)
        self.func_box = ttk.Combobox(f1, textvariable=self.func_key, state="readonly",
                                     width=17, font=("PingFang SC", 9),
                                     values=[f"{k} | {v.cn}" for k, v in BENCHMARKS.items()])
        self.func_box.grid(row=0, column=1, padx=4, pady=3)
        self.func_box.set(f"rastrigin | {BENCHMARKS['rastrigin'].cn}")
        self.func_box.bind("<<ComboboxSelected>>", lambda _e: self._on_func_change())

        tk.Label(f1, text="维度", bg=BG, font=("PingFang SC", 9)).grid(
            row=1, column=0, sticky="w", padx=6, pady=3)
        self.dim_box = ttk.Combobox(f1, textvariable=self.dim, state="readonly", width=17,
                                    font=("PingFang SC", 9), values=[2, 10, 30])
        self.dim_box.grid(row=1, column=1, padx=4, pady=3)
        self.func_info = tk.Label(f1, text="", bg=BG, fg=TEXT_SUB, wraplength=270,
                                  justify="left", font=("PingFang SC", 8))
        self.func_info.grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 6))

        # ---- 算法设置
        f2 = ttk.LabelFrame(side, text="算法设置")
        f2.pack(fill="x", pady=4)
        tk.Label(f2, text="算法", bg=BG, font=("PingFang SC", 9)).grid(
            row=0, column=0, sticky="w", padx=6, pady=3)
        self.algo_box = ttk.Combobox(f2, textvariable=self.algo, state="readonly", width=17,
                                     font=("PingFang SC", 9),
                                     values=[f"{a} | {ALGO_CN[a]}" for a in ALGORITHMS])
        self.algo_box.grid(row=0, column=1, padx=4, pady=3)
        self.algo_box.set(f"PSO | {ALGO_CN['PSO']}")
        self.algo_box.bind("<<ComboboxSelected>>", lambda _e: self._on_algo_change())

        self.common_frame = tk.Frame(f2, bg=BG)
        self.common_frame.grid(row=1, column=0, columnspan=2, sticky="we")
        for i, (label, var, lo, hi) in enumerate([
                ("种群规模", self.pop_size, 4, 300),
                ("迭代次数", self.max_iter, 10, 1000),
                ("随机种子", self.seed, 1, 9999)]):
            tk.Label(self.common_frame, text=label, bg=BG,
                     font=("PingFang SC", 9)).grid(row=i, column=0, sticky="w", padx=6, pady=2)
            tk.Spinbox(self.common_frame, from_=lo, to=hi, textvariable=var, width=8,
                       font=("PingFang SC", 9)).grid(row=i, column=1, padx=4, pady=2)

        self.param_frame = ttk.LabelFrame(side, text="算法参数")
        self.param_frame.pack(fill="x", pady=4)

        # ---- 运行控制
        f3 = ttk.LabelFrame(side, text="运行控制")
        f3.pack(fill="x", pady=4)
        tk.Button(f3, text="▶  开始寻优（动画）", command=self.start_run, bg=ACCENT2,
                  fg="white", relief="flat", font=("PingFang SC", 11, "bold"),
                  pady=7).pack(fill="x", padx=6, pady=(6, 3))
        row = tk.Frame(f3, bg=BG)
        row.pack(fill="x", padx=6, pady=3)
        tk.Button(row, text="暂停/继续", command=self.toggle_play, relief="flat",
                  bg="#E3E9F0", font=("PingFang SC", 9), width=10).pack(side="left")
        tk.Button(row, text="重播动画", command=self.replay, relief="flat",
                  bg="#E3E9F0", font=("PingFang SC", 9), width=10).pack(side="left", padx=4)
        tk.Label(f3, text="动画速度（帧间隔 ms，越小越快）", bg=BG, fg=TEXT_SUB,
                 font=("PingFang SC", 8)).pack(anchor="w", padx=8, pady=(6, 0))
        tk.Scale(f3, from_=1, to=120, orient="horizontal", variable=self.speed,
                 bg=BG, highlightthickness=0, showvalue=True, length=260).pack(
            fill="x", padx=6)
        tk.Button(f3, text="＋ 加入对比（多算法同图）", command=self.add_compare,
                  bg=BTN_BG, fg="white", relief="flat", font=("PingFang SC", 9),
                  pady=5).pack(fill="x", padx=6, pady=3)
        tk.Button(f3, text="清空对比与轨迹", command=self.clear_all, relief="flat",
                  bg="#E3E9F0", font=("PingFang SC", 9), pady=4).pack(fill="x", padx=6, pady=3)
        tk.Button(f3, text="导出当前画面 PNG", command=self.export_png, bg=ACCENT,
                  fg="white", relief="flat", font=("PingFang SC", 9),
                  pady=5).pack(fill="x", padx=6, pady=(3, 8))

        # ---- 本次运行信息
        self.info = ttk.LabelFrame(side, text="本次运行结果")
        self.info.pack(fill="both", expand=True, pady=4)
        self.info_text = tk.Text(self.info, height=9, font=("PingFang SC", 9),
                                 bg="white", relief="flat", wrap="word", fg=TEXT_MAIN)
        self.info_text.pack(fill="both", expand=True, padx=4, pady=4)
        self.info_text.insert("1.0", "尚无运行结果。\n点击「开始寻优（动画）」开始。")
        self.info_text.configure(state="disabled")

    def _build_canvas(self, parent):
        wrap = tk.Frame(parent, bg=CARD_BG, highlightbackground=LINE, highlightthickness=1)
        wrap.pack(fill="both", expand=True)
        self.fig = Figure(figsize=(11.8, 5.9), dpi=100, facecolor="white")
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.22)
        self.ax_space = self.fig.add_subplot(gs[0, 0])
        self.ax_conv = self.fig.add_subplot(gs[0, 1])
        self.canvas = FigureCanvasTkAgg(self.fig, master=wrap)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, wrap, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")

    def _build_compare_table(self, parent):
        box = ttk.LabelFrame(parent, text="多算法对比结果（点击「加入对比」累积）")
        box.pack(fill="x", pady=(8, 0))
        cols = ("algo", "dim", "func", "best", "error", "evals", "time", "success")
        heads = ("算法", "维度", "函数", "最优值", "误差", "评价次数", "耗时(s)", "是否达标")
        widths = (70, 50, 130, 130, 110, 90, 80, 80)
        self.tree = ttk.Treeview(box, columns=cols, show="headings", height=6)
        for c, h, w in zip(cols, heads, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill="x", padx=4, pady=4)

    # -------------------------------------------------- 交互回调
    def _current_func(self):
        return self.func_box.get().split(" | ")[0].strip()

    def _current_algo(self):
        return self.algo_box.get().split(" | ")[0].strip()

    def _on_func_change(self):
        key = self._current_func()
        b = BENCHMARKS[key]
        dims = list(b.dims)
        self.dim_box.configure(values=dims)
        if self.dim.get() not in dims:
            self.dim.set(dims[0])
        self.surface_cache.clear()
        self.func_info.configure(
            text=f"类型：{b.category}｜理论最优 f* = {b.fmin:.6g}\n"
                 f"搜索范围：[{b.lb:g}, {b.ub:g}]｜可用维度 {dims}\n"
                 f"说明：{b.note}")
        self.redraw_space()

    def _on_algo_change(self):
        for w in self.param_frame.winfo_children():
            w.destroy()
        algo = self._current_algo()
        self.param_vars.clear()
        spec = ALGO_PARAMS.get(algo, [])
        if not spec:
            tk.Label(self.param_frame, text="该算法无可调参数（使用标准设置）",
                     bg=BG, fg=TEXT_SUB, font=("PingFang SC", 9)).pack(padx=6, pady=8)
        defaults = DEFAULT_PARAMS.get(algo, {})
        for i, (name, label, lo, hi, nd) in enumerate(spec):
            tk.Label(self.param_frame, text=label, bg=BG,
                     font=("PingFang SC", 9)).grid(row=i, column=0, sticky="w", padx=6, pady=2)
            default = defaults.get(name, lo)
            if default is None:
                default = 1.0 / max(2, int(self.dim.get()))    # pm 默认取 1/维度
            var = tk.StringVar(value=f"{float(default):.{nd}f}" if nd else str(int(default)))
            tk.Spinbox(self.param_frame, from_=lo, to=hi, increment=10 ** (-nd) if nd else 1,
                       textvariable=var, width=9, font=("PingFang SC", 9)).grid(
                row=i, column=1, padx=4, pady=2)
            self.param_vars[name] = (var, nd)
        self.redraw_space()

    def _collect_params(self):
        params = {}
        for name, (var, nd) in self.param_vars.items():
            try:
                params[name] = float(var.get()) if nd else int(float(var.get()))
            except ValueError:
                continue
        return params

    # -------------------------------------------------- 搜索空间绘制
    def redraw_space(self):
        key = self._current_func()
        b = BENCHMARKS[key]
        dim = self.dim.get()
        ax = self.ax_space
        ax.clear()
        ax.grid(False)

        lb, ub = b.bounds(2)
        lb_d, ub_d = b.bounds(dim)
        cache_key = (key, "full")
        if cache_key not in self.surface_cache:
            self.surface_cache[cache_key] = grid_surface(key, n=240)
        XX, YY, ZZ = self.surface_cache[cache_key]

        if dim == 2:
            Z = ZZ
            note = ""
        else:                                          # 高维：固定其余维度取中值的切片
            fixed = (lb_d + ub_d) / 2.0
            pts = np.column_stack([XX.ravel(), YY.ravel(),
                                   np.tile(fixed[2:], (XX.size, 1))])
            Z = b.fn(pts).reshape(XX.shape)
            note = f"（切片：其余 {dim - 2} 维固定在搜索区间中点）"

        # 统一在对数尺度上绘制；若函数存在负值（如 Six-Hump Camel）则先整体平移
        shift = max(0.0, 1e-3 - float(Z.min()))
        Zs = np.log10(Z + shift + 1e-12)
        levels = np.linspace(Zs.min(), Zs.max(), 26)
        ax.contourf(XX, YY, Zs, levels=levels, cmap="viridis", alpha=0.85)
        ax.contour(XX, YY, Zs, levels=levels[::3], colors="white",
                   linewidths=0.35, alpha=0.45)

        xopt = np.asarray(b.xmin, dtype=float)
        if xopt.size == 1:
            xopt = np.repeat(xopt, 2)
        if np.all(xopt >= lb) and np.all(xopt <= ub):
            ax.scatter([xopt[0]], [xopt[1]], marker="*", s=230, color=ACCENT3,
                       edgecolor="white", lw=1.2, zorder=7, label="理论最优位置")
            ax.legend(fontsize=8, loc="upper right")

        self.pop_artist = ax.scatter([], [], s=14, color="#FFFFFF", alpha=0.55,
                                     edgecolor="none", zorder=5)
        self.best_artist = ax.scatter([], [], marker="o", s=95, facecolor="none",
                                      edgecolor=ACCENT, lw=2.0, zorder=8)
        self.traj_artist, = ax.plot([], [], "-", color=ACCENT, lw=1.5, alpha=0.85, zorder=6,
                                    label="历代最优位置")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel("x1", fontsize=10)
        ax.set_ylabel("x2", fontsize=10)
        ax.set_title(f"{b.cn} 的搜索空间{note}", fontsize=12)

        self.ax_conv.clear()
        self.ax_conv.set_xlabel("迭代代数", fontsize=10)
        self.ax_conv.set_ylabel("最优误差 |f - f*|", fontsize=10)
        self.ax_conv.set_yscale("log")
        self.ax_conv.yaxis.set_major_locator(LogLocator(base=10.0))
        self.ax_conv.yaxis.set_major_formatter(FuncFormatter(log_fmt))
        self.ax_conv.yaxis.set_minor_formatter(NullFormatter())
        self.ax_conv.axhline(b.tol, color=PALETTE["neutral"], ls=":", lw=1.2,
                             label=f"成功判据 {b.tol:g}")
        self.conv_line, = self.ax_conv.plot([], [], lw=2.2, color=BTN_BG, label="本次运行")
        self.ax_conv.legend(fontsize=8, loc="upper right")
        self.ax_conv.set_title("收敛曲线（对数误差）", fontsize=12)
        self._draw_compare_curves()
        self.fig.subplots_adjust(left=0.075, right=0.975, top=0.90, bottom=0.115, wspace=0.24)
        self.canvas.draw_idle()

    def _draw_compare_curves(self):
        """把已加入对比的历史结果画到收敛图上（多算法同图比较）。"""
        for i, r in enumerate(self.compare):
            color = ALGO_COLORS.get(r["algo"], PALETTE["neutral"])
            self.ax_conv.plot(r["history_gen"], np.clip(r["history_best"] - r["fmin"], 1e-12, None),
                              lw=1.5, ls="--", color=color, alpha=0.85,
                              label=f"{r['algo']}#{i + 1}")
        if self.compare:
            self.ax_conv.legend(fontsize=7, loc="upper right", ncol=1)

    # -------------------------------------------------- 运行与动画
    def start_run(self):
        key = self._current_func()
        algo = self._current_algo()
        b = BENCHMARKS[key]
        dim = self.dim.get()
        if dim not in b.dims:
            messagebox.showwarning("维度不支持", f"{b.cn} 仅支持 {b.dims} 维。")
            return
        params = self._collect_params()
        if algo == "SA":
            params.pop("T0_ratio", None)
        try:
            res = ALGORITHMS[algo](
                b, dim,
                max_iter=int(self.max_iter.get()),
                pop_size=int(self.pop_size.get()),
                seed=int(self.seed.get()),
                stride=1, keep_traj=True, **params)
        except Exception as exc:                       # noqa: BLE001
            messagebox.showerror("运行失败", f"{type(exc).__name__}: {exc}")
            return

        res.update({"algo": algo, "func": key, "func_cn": b.cn, "fmin": b.fmin})
        self.result = res
        self._build_frames()
        self.redraw_space()
        self.frame_idx = 0
        self.playing = True
        self.status.configure(text=f"运行完成：{algo} 在 {b.cn}（{dim} 维）上"
                                   f"耗时 {res['elapsed'] * 1000:.1f} 毫秒，"
                                   f"评价 {res['n_evals']:,} 次 —— 正在播放动画")
        self._update_info()
        self._anim_step()

    def _build_frames(self):
        """把轨迹按帧率抽样，最多约 70 帧，保证动画流畅。"""
        res = self.result
        traj = res.get("trajectory")
        if traj:
            step = max(1, len(traj) // 70)
            self.frames = [traj[i] for i in range(0, len(traj), step)]
            if self.frames[-1][0] != traj[-1][0]:
                self.frames.append(traj[-1])
        else:
            gen = res["history_gen"]
            self.frames = [(int(g), res["best_x"], None) for g in gen]

    def toggle_play(self):
        if self.result is None:
            return
        self.playing = not self.playing
        if self.playing:
            self._anim_step()
        self.status.configure(text="动画播放中…" if self.playing else "动画已暂停")

    def replay(self):
        if self.result is None:
            return
        self.frame_idx = 0
        self.traj_artist.set_data([], [])
        self.playing = True
        self._anim_step()

    def _anim_step(self):
        if not self.playing or self.result is None:
            return
        if self.frame_idx >= len(self.frames):
            self.playing = False
            self.status.configure(text="动画播放结束")
            return
        gen, best_x, pop = self.frames[self.frame_idx]
        if pop is not None:
            self.pop_artist.set_offsets(np.asarray(pop)[:, :2])
        self.best_artist.set_offsets(np.asarray(best_x[:2]).reshape(1, 2))

        traj = np.array([f[1][:2] for f in self.frames[:self.frame_idx + 1]])
        self.traj_artist.set_data(traj[:, 0], traj[:, 1])

        res = self.result
        mask = res["history_gen"] <= gen
        err = np.clip(res["history_best"][mask] - res["fmin"], 1e-12, None)
        self.conv_line.set_data(res["history_gen"][mask], err)
        self.ax_conv.relim()
        self.ax_conv.autoscale_view(scalex=True, scaley=False)
        if err.size:
            lo = max(err.min() * 0.3, 1e-16)
            hi = max(err.max() * 3.0, lo * 100)
            self.ax_conv.set_ylim(lo, hi)

        self.frame_idx += 1
        self.canvas.draw_idle()
        self.after(max(1, int(self.speed.get())), self._anim_step)

    # -------------------------------------------------- 结果信息
    def _update_info(self):
        res = self.result
        b = BENCHMARKS[res["func"]]
        text = (
            f"算法：{res['algo']}（{ALGO_CN.get(res['algo'], '')}）\n"
            f"函数：{b.cn}｜维度：{res['dim']}\n"
            f"理论最优值 f* = {b.fmin:.6g}\n"
            f"本次最优值 f_best = {res['best_value']:.6g}\n"
            f"绝对误差 |f_best - f*| = {res['error']:.3e}\n"
            f"是否达到成功判据({b.tol:g})：{'是' if res['success'] else '否'}\n"
            f"函数评价次数：{res['n_evals']:,}\n"
            f"运行耗时：{res['elapsed'] * 1000:.2f} 毫秒\n"
            f"收敛代数：{res['convergence_gen'] if res['convergence_gen'] is not None else '未收敛'}\n"
            f"\n最优解向量（前 10 维）：\n"
            f"{np.array2string(np.asarray(res['best_x'])[:10], precision=4, max_line_width=40)}"
        )
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", text)
        self.info_text.configure(state="disabled")

    def add_compare(self):
        if self.result is None:
            messagebox.showinfo("提示", "请先运行一次寻优。")
            return
        self.compare.append(self.result)
        res = self.result
        self.tree.insert("", "end", values=(
            res["algo"], f"{res['dim']}D", res["func_cn"], f"{res['best_value']:.6g}",
            f"{res['error']:.3e}", f"{res['n_evals']:,}", f"{res['elapsed'] * 1000:.2f}",
            "达标" if res["success"] else "未达标"))
        self.redraw_space()
        self.status.configure(text=f"已加入对比：{res['algo']}（当前共 {len(self.compare)} 条）")

    def clear_all(self):
        self.compare.clear()
        self.tree.delete(*self.tree.get_children())
        self.result = None
        self.frames = []
        self.frame_idx = 0
        self.redraw_space()
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", "尚无运行结果。\n点击「开始寻优（动画）」开始。")
        self.info_text.configure(state="disabled")
        self.status.configure(text="已清空对比与轨迹")

    def export_png(self):
        path = filedialog.asksaveasfilename(
            title="导出当前画面", defaultextension=".png",
            initialfile="寻优演示.png", filetypes=[("PNG 图片", "*.png")],
            initialdir=str(REPORT_DIR))
        if not path:
            return
        self.fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        messagebox.showinfo("导出成功", f"画面已保存至：\n{path}")


def main():
    app = SimulatorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
