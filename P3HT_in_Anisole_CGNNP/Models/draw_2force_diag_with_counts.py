# -*- coding: utf-8 -*-
"""
绘制“对角线散点 + 右侧 counts 分布”的组合图。

典型用途：
1. 先运行 force_export_and_check.py，得到 .npz 文件；
2. 再运行本脚本，从 .npz 中读取两个一维数组；
3. 生成类似文献中的验证图：
   - 左图：x-y 散点 + y=x 对角线
   - 右图：counts 分布（与左图共用 y 轴）

运行方式：
    python draw_force_diag_with_counts.py
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# =========================
# 用户需要修改的参数
# =========================

# 选择要画哪个数据集：
# 可选："train", "val", "test"
DATA_SPLIT = "train"

# 输入的 npz 文件。
# 推荐直接使用 force_export_and_check.py 导出的：
# 1. OUTPUT_PREFIX + ".npz"
# 2. OUTPUT_PREFIX + "_all_arrays.npz"
NPZ_PATH = f"test3_force_scatter/{DATA_SPLIT}_all_arrays.npz"

# 要拿来画横坐标 / 纵坐标的数组名。
# train.npz 里常见的是：
#   predicted_force, target_force
# train_all_arrays.npz 里常见的是：
#   predicted_delta_force, true_delta_force,
#   prior_force, aa_mapped_force, total_predicted_force
X_KEY = "true_delta_force"
Y_KEY = "predicted_delta_force"

# 输出图片路径
OUTPUT_FIG = f"test3_force_scatter/{DATA_SPLIT}_true_vs_pred_delta_diag_counts.png"

# 图标题和坐标轴标题
# SHOW_TITLE = True  时，显示 FIG_TITLE
# SHOW_TITLE = False 时，不显示上方标题
SHOW_TITLE = False
FIG_TITLE = f"{DATA_SPLIT.capitalize()}"
X_LABEL = "True $\Delta F$ (eV Å$^{-1}$)"
Y_LABEL = "Predicted $\Delta F$ (eV Å$^{-1}$)"

# ============================================================
# 用户选择图的模式：这里改
# ------------------------------------------------------------
# True  : 生成“双图”，即 左侧对角线散点 + 右侧 counts 分布
# False : 只生成“单个对角线图”，不画右侧 counts 面板
# ============================================================
SHOW_COUNTS_PANEL = True

# 右侧 counts 面板画哪一个分布：
# - "y"    : 只画纵坐标数据分布；由于右图与左图共享纵坐标，这是最常用设置
# - "x"    : 只画横坐标数据分布
# - "both" : 两者都画在右侧，便于比较
# 只有在 SHOW_COUNTS_PANEL = True 时，这个设置才会生效。
HIST_SOURCE = "y"

# 如果数据点太多，散点图只随机抽样这么多个点来画。
# 注意：右侧 counts 直方图仍使用全部数据统计。
MAX_POINTS_TO_PLOT = 300000

# 右侧直方图 bin 数
HIST_BINS = 120

# 右侧 Counts 轴是否使用对数坐标
# True  表示使用对数坐标
# False 表示使用普通线性坐标
USE_LOG_COUNT_AXIS = False

# 当右侧 Counts 轴使用线性坐标时，是否把 100、1000、10000 这类整数 10 的幂
# 显示成 10^2、10^3、10^4 的数学格式。
# 这样即使不是对数坐标，也能保留论文里常见的幂次写法。
USE_POWER_OF_TEN_COUNT_LABELS = True

# ============================================================
# 用户控制右图 counts 横坐标 tick：这里改
# ------------------------------------------------------------
# True  : 右图 counts 横坐标只显示一个 tick
# False : 使用 matplotlib 默认的多个 tick
#
# 例如：
# - train 时可设 SINGLE_COUNT_TICK_VALUE = 1e4
#   SINGLE_COUNT_TICK_LABEL = r"$10^4$"
# - val/test 时可设 SINGLE_COUNT_TICK_VALUE = 1e3
#   SINGLE_COUNT_TICK_LABEL = r"$10^3$"
#
# 如果 SINGLE_COUNT_TICK_LABEL = None，则程序会尽量自动格式化。
# ============================================================
SHOW_ONLY_ONE_COUNT_TICK = True
SINGLE_COUNT_TICK_VALUE = 1e4
SINGLE_COUNT_TICK_LABEL = r"$10^4$"

# 右侧 counts 分布图样式
# True  : 用“轮廓线 + 浅色填充”的方式显示，更平滑、更像论文图
# False : 用原始直方图块状显示
USE_SMOOTH_COUNT_PROFILE = False
COUNT_PROFILE_LINE_WIDTH = 3
COUNT_PROFILE_ALPHA = 0.22

# 散点图点样式
POINT_SIZE = 8
POINT_ALPHA = 0.25
POINT_COLOR = "tab:red"

# 是否在图中标注 MAE / RMSE
SHOW_METRICS = True

# ============================================================
# 用户修改图的横纵坐标范围：这里改
# ------------------------------------------------------------
# 本脚本会让左图 x/y 使用同一套范围，这样对角线 y=x 才是标准 45 度。
# 右图与左图共享“纵坐标”，因此这里也会同时控制右图的纵坐标显示范围。
#
# 用法：
# - AXIS_LIMITS = None
#   表示根据数据自动决定范围
# - AXIS_LIMITS = (-5.5, 5.5)
#   表示手动指定横纵坐标都显示为 -5.5 到 5.5
# ============================================================
AXIS_LIMITS =(-120, 120)

# ============================================================
# 用户修改字体、字号、线条样式：这里改
# ------------------------------------------------------------
# FONT_FAMILY：整张图默认字体类型
# 可尝试：
# - "Arial"
# - "Helvetica"
# - "Times New Roman"
# - "DejaVu Sans"
#
# 各字号含义：
# - TITLE_FONT_SIZE：图标题字号
# - LABEL_FONT_SIZE：横纵坐标标题字号
# - TICK_FONT_SIZE：坐标刻度数字字号
# - METRICS_FONT_SIZE：RMSE / MAE 文字字号
# - HIST_LEGEND_FONT_SIZE：右图图例字号（仅 HIST_SOURCE="both" 时使用）
# ============================================================
FONT_FAMILY = "Arial"
TITLE_FONT_SIZE = 32
LABEL_FONT_SIZE = 32
TICK_FONT_SIZE = 28
METRICS_FONT_SIZE = 28
HIST_LEGEND_FONT_SIZE = 28

# ============================================================
# 用户修改左图指标文字位置：这里改
# ------------------------------------------------------------
# 采用坐标轴相对坐标：
# - (0, 0) 是左下角
# - (1, 1) 是右上角
# 例如：
# - METRICS_TEXT_X = 0.05 表示更靠左
# - METRICS_TEXT_Y = 0.95 表示更靠上
# ============================================================
METRICS_TEXT_X = 0.05
METRICS_TEXT_Y = 0.95

# ============================================================
# 用户修改整张图的长宽：这里改
# ------------------------------------------------------------
# FIG_WIDTH：整张图总宽度
# FIG_HEIGHT：整张图总高度
# 单位都是英寸（inch）
# 例如：
# - FIG_WIDTH = 7.2
# - FIG_HEIGHT = 5.8
# ============================================================
FIG_WIDTH = 8
FIG_HEIGHT = 6

# ============================================================
# 用户修改左右两个图所占宽度比例：这里改
# ------------------------------------------------------------
# MAIN_PANEL_WIDTH：左侧散点图宽度比例
# HIST_PANEL_WIDTH：右侧 counts 图宽度比例
# PANEL_WSPACE：两张图之间的间距
# 例如 MAIN_PANEL_WIDTH=4.6、HIST_PANEL_WIDTH=1.2 表示左图明显更宽
# ============================================================
MAIN_PANEL_WIDTH = 4.6
HIST_PANEL_WIDTH = 1.2
PANEL_WSPACE = 0.12

# 线条和边框样式 斜线；外框；刻度宽度；刻度长度
DIAGONAL_LINE_WIDTH = 3
SPINE_LINE_WIDTH = 3
TICK_WIDTH = 4
TICK_LENGTH = 8

# 图像参数
FIG_DPI = 900

plt.rcParams["font.family"] = FONT_FAMILY

_VALID_SPLITS = {"train", "val", "test"}
if DATA_SPLIT not in _VALID_SPLITS:
    raise ValueError(f"DATA_SPLIT must be one of {_VALID_SPLITS}, got: {DATA_SPLIT}")


# =========================
# 内部函数
# =========================

def ensure_output_dir(output_path):
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def load_arrays(npz_path, x_key, y_key):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path)
    available_keys = list(data.keys())

    if x_key not in data:
        raise KeyError(f"X_KEY '{x_key}' not found. Available keys: {available_keys}")
    if y_key not in data:
        raise KeyError(f"Y_KEY '{y_key}' not found. Available keys: {available_keys}")

    x = np.asarray(data[x_key]).reshape(-1)
    y = np.asarray(data[y_key]).reshape(-1)

    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")

    return x, y, available_keys


def compute_metrics(x, y):
    diff = x - y
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    ss_res = np.sum((y - x) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return mae, rmse, r2


def sample_for_scatter(x, y, max_points):
    n = len(x)
    if n <= max_points:
        return x, y

    rng = np.random.default_rng(2026)
    idx = rng.choice(n, size=max_points, replace=False)
    return x[idx], y[idx]


def get_axis_limits(x, y, axis_limits):
    if axis_limits is not None:
        return axis_limits

    vmin = min(np.min(x), np.min(y))
    vmax = max(np.max(x), np.max(y))
    padding = 0.03 * (vmax - vmin) if vmax > vmin else 1.0
    return vmin - padding, vmax + padding


def format_count_tick_as_power_of_ten(value, _pos):
    if value <= 0:
        return ""

    rounded = int(round(value))
    if not np.isclose(value, rounded):
        return ""

    exponent = np.log10(rounded)
    if np.isclose(exponent, round(exponent)):
        return rf"$10^{{{int(round(exponent))}}}$"

    return f"{rounded}"


def smooth_1d(values, window=7):
    if window <= 1:
        return np.asarray(values, dtype=float)
    if window % 2 == 0:
        window += 1
    arr = np.asarray(values, dtype=float)
    pad = window // 2
    arr_pad = np.pad(arr, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr_pad, kernel, mode="valid")


def draw_count_profile(ax, values, bins, color, alpha, label=None):
    counts, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # 对数坐标下不能显示 0，把它抬到 1 附近
    profile = counts.astype(float)
    if USE_LOG_COUNT_AXIS:
        profile = np.maximum(profile, 1.0)

    if USE_SMOOTH_COUNT_PROFILE:
        profile = smooth_1d(profile, window=7)

    left_base = 1.0 if USE_LOG_COUNT_AXIS else 0.0
    ax.fill_betweenx(
        centers,
        left_base,
        profile,
        color=color,
        alpha=alpha,
        linewidth=0.0,
    )
    ax.plot(profile, centers, color=color, linewidth=COUNT_PROFILE_LINE_WIDTH, label=label)


def plot_diag_with_counts(x, y, output_fig):
    plot_x, plot_y = sample_for_scatter(x, y, MAX_POINTS_TO_PLOT)
    mae, rmse, r2 = compute_metrics(x, y)
    vmin, vmax = get_axis_limits(x, y, AXIS_LIMITS)

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    if SHOW_COUNTS_PANEL:
        gs = fig.add_gridspec(
            1,
            2,
            width_ratios=[MAIN_PANEL_WIDTH, HIST_PANEL_WIDTH],
            wspace=PANEL_WSPACE,
        )
        ax_scatter = fig.add_subplot(gs[0, 0])
        ax_hist = fig.add_subplot(gs[0, 1], sharey=ax_scatter)
    else:
        ax_scatter = fig.add_subplot(111)
        ax_hist = None

    ax_scatter.scatter(
        plot_x,
        plot_y,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        color=POINT_COLOR,
        edgecolors="none",
    )
    ax_scatter.plot(
        [vmin, vmax],
        [vmin, vmax],
        color="black",
        linewidth=DIAGONAL_LINE_WIDTH,
    )

    ax_scatter.set_xlim(vmin, vmax)
    ax_scatter.set_ylim(vmin, vmax)
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.set_xlabel(X_LABEL, fontsize=LABEL_FONT_SIZE)
    ax_scatter.set_ylabel(Y_LABEL, fontsize=LABEL_FONT_SIZE)
    if SHOW_TITLE:
        ax_scatter.set_title(FIG_TITLE, fontsize=TITLE_FONT_SIZE)

    if SHOW_METRICS:
        text = (
            f"RMSE={rmse:.4f} eV/Å\n"
            f"MAE={mae:.4f} eV/Å\n"
            f"$R^2$={r2:.4f}"
        )
        ax_scatter.text(
            METRICS_TEXT_X,
            METRICS_TEXT_Y,
            text,
            transform=ax_scatter.transAxes,
            ha="left",
            va="top",
            multialignment="left",
            fontsize=METRICS_FONT_SIZE,
        )

    if SHOW_COUNTS_PANEL:
        bins = np.linspace(vmin, vmax, HIST_BINS + 1)

        if HIST_SOURCE == "y":
            draw_count_profile(
                ax_hist,
                y,
                bins=bins,
                color=POINT_COLOR,
                alpha=COUNT_PROFILE_ALPHA,
            )
        elif HIST_SOURCE == "x":
            draw_count_profile(
                ax_hist,
                x,
                bins=bins,
                color=POINT_COLOR,
                alpha=COUNT_PROFILE_ALPHA,
            )
        elif HIST_SOURCE == "both":
            draw_count_profile(
                ax_hist,
                y,
                bins=bins,
                color="orangered",
                alpha=0.20,
                label="Y counts",
            )
            draw_count_profile(
                ax_hist,
                x,
                bins=bins,
                color="royalblue",
                alpha=0.14,
                label="X counts",
            )
            ax_hist.legend(loc="upper right", fontsize=HIST_LEGEND_FONT_SIZE, framealpha=0.9)
        else:
            raise ValueError("HIST_SOURCE must be 'x', 'y', or 'both'")

        ax_hist.set_xlabel("Counts", fontsize=LABEL_FONT_SIZE)
        ax_hist.tick_params(axis="y", labelleft=False)

        if USE_LOG_COUNT_AXIS:
            ax_hist.set_xscale("log")
            ax_hist.set_xlim(left=1.0)
        elif USE_POWER_OF_TEN_COUNT_LABELS:
            ax_hist.xaxis.set_major_formatter(FuncFormatter(format_count_tick_as_power_of_ten))

        if SHOW_ONLY_ONE_COUNT_TICK:
            ax_hist.set_xticks([SINGLE_COUNT_TICK_VALUE])
            if SINGLE_COUNT_TICK_LABEL is None:
                auto_label = format_count_tick_as_power_of_ten(SINGLE_COUNT_TICK_VALUE, None)
                if auto_label == "":
                    auto_label = f"{SINGLE_COUNT_TICK_VALUE:g}"
                ax_hist.set_xticklabels([auto_label])
            else:
                ax_hist.set_xticklabels([SINGLE_COUNT_TICK_LABEL])

    axes = (ax_scatter, ax_hist) if SHOW_COUNTS_PANEL else (ax_scatter,)
    for ax in axes:
        ax.tick_params(
            direction="in",
            top=True,
            right=True,
            width=TICK_WIDTH,
            length=TICK_LENGTH,
            labelsize=TICK_FONT_SIZE,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(SPINE_LINE_WIDTH)

    fig.tight_layout()
    fig.savefig(output_fig, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_output_dir(OUTPUT_FIG)

    print(f"Loading NPZ: {NPZ_PATH}")
    x, y, available_keys = load_arrays(NPZ_PATH, X_KEY, Y_KEY)

    print(f"Available keys: {available_keys}")
    print(f"X_KEY = {X_KEY}")
    print(f"Y_KEY = {Y_KEY}")
    print(f"Number of force components = {len(x)}")
    print(f"HIST_SOURCE = {HIST_SOURCE}")

    plot_diag_with_counts(x, y, OUTPUT_FIG)

    mae, rmse, r2 = compute_metrics(x, y)
    print(f"Saved figure: {OUTPUT_FIG}")
    print(f"RMSE = {rmse:.8f} eV/Å")
    print(f"MAE  = {mae:.8f} eV/Å")
    print(f"R^2  = {r2:.8f}")


if __name__ == "__main__":
    main()
