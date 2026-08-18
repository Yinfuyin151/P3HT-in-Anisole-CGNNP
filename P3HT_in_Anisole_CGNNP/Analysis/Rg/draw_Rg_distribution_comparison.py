#!/usr/bin/env python3
import os
import numpy as np
from matplotlib.ticker import MultipleLocator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 用户需要修改的参数
# ============================================================

AA_FILE = "AA_Rg_dist.xvg"
CG_FILE = "CG_Rg_dist.xvg"
M3_FILE = "M3_Rg_dist.xvg"

OUTPUT_FIG = "AA_CG_M3_Rg_distribution_comparison.svg"

# 如果想手动限制横纵坐标范围，在这里改
# 例如：
# X_AXIS_LIMITS = (0.0, 8.0)
# Y_AXIS_LIMITS = (0.0, 1.2)
X_AXIS_LIMITS = None
Y_AXIS_LIMITS = None
# 如果想手动设置 x/y 轴主刻度间隔，在这里改。None 表示不手动设置
X_MAJOR_TICK_STEP = None
Y_MAJOR_TICK_STEP = None

# ============================================================
# 曲线平滑设置：这里改
# ------------------------------------------------------------
# SMOOTH_WINDOW_POINTS：
# - 平滑窗口大小，单位是“数据点个数”
# - 数值越大，曲线越平滑，但细节会丢失更多
# - 建议先试 5、7、9、11
#
# SHOW_FLUCTUATION_SHADE：
# - True  表示把原始波动转化为半透明阴影
# - False 表示只画平滑后的主曲线
# ============================================================
SMOOTH_WINDOW_POINTS = 5
SHOW_FLUCTUATION_SHADE = False
SHADE_ALPHA = 0.20

# 图像样式
FONT_FAMILY = "Arial"
TITLE_FONT_SIZE = 38
LABEL_FONT_SIZE = 38
TICK_FONT_SIZE = 34
LEGEND_FONT_SIZE = 34
FIG_DPI = 600
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

# 线条和边框样式
# 用户可在这里修改刻度线宽度和长度
# TICK_WIDTH：刻度线粗细
# TICK_LENGTH：刻度线长度
# X_TICK_PAD：横坐标数字离坐标轴的距离
# Y_TICK_PAD：纵坐标数字离坐标轴的距离
SPINE_LINE_WIDTH = 3
TICK_WIDTH = 4
TICK_LENGTH = 8
X_TICK_PAD = 6
Y_TICK_PAD = 8

# 配色
AA_COLOR = "tab:grey"
CG_COLOR = "tab:red"
M3_COLOR = "tab:blue"
LINE_WIDTH = 5
AA_BAR_ALPHA = 0.4
AA_BAR_WIDTH_SCALE = 0.90

FIG_TITLE = "Radius of Gyration Distribution"
X_LABEL = "Radius of gyration (nm)"
Y_LABEL = "Distribution"
# 是否显示图片上方标题。False 表示不显示
SHOW_TITLE = False
# 是否显示图例。True 表示显示，False 表示不显示
SHOW_LEGEND = True


plt.rcParams["font.family"] = FONT_FAMILY


def read_xvg_xy(xvg_path):
    if not os.path.exists(xvg_path):
        raise FileNotFoundError(f"File not found: {xvg_path}")

    x = []
    y = []
    with open(xvg_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            x.append(float(parts[0]))
            y.append(float(parts[1]))

    if not x:
        raise ValueError(f"No numeric data found in {xvg_path}")

    return x, y


def moving_average(y, window):
    if window <= 1:
        return np.asarray(y, dtype=float)

    if window % 2 == 0:
        raise ValueError("SMOOTH_WINDOW_POINTS must be an odd integer, e.g. 5, 7, 9.")

    y = np.asarray(y, dtype=float)
    pad = window // 2
    y_pad = np.pad(y, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y_pad, kernel, mode="valid")


def moving_std_around_curve(y, y_smooth, window):
    if window <= 1:
        return np.zeros_like(y_smooth)

    residual = np.asarray(y, dtype=float) - np.asarray(y_smooth, dtype=float)
    pad = window // 2
    res_pad = np.pad(residual, pad_width=pad, mode="edge")
    std_values = np.zeros_like(y_smooth)

    for i in range(len(y_smooth)):
        local = res_pad[i:i + window]
        std_values[i] = np.std(local)

    return std_values


def plot_smoothed_distribution(ax, x, y, color, label, zorder_line=3, zorder_shade=2):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    y_smooth = moving_average(y_arr, SMOOTH_WINDOW_POINTS)

    if SHOW_FLUCTUATION_SHADE:
        y_std = moving_std_around_curve(y_arr, y_smooth, SMOOTH_WINDOW_POINTS)
        lower = np.clip(y_smooth - y_std, a_min=0.0, a_max=None)
        upper = y_smooth + y_std
        ax.fill_between(
            x_arr,
            lower,
            upper,
            color=color,
            alpha=SHADE_ALPHA,
            linewidth=0.0,
            zorder=zorder_shade,
        )

    ax.plot(
        x_arr,
        y_smooth,
        color=color,
        linewidth=LINE_WIDTH,
        label=label,
        zorder=zorder_line,
    )


def plot_aa_distribution_as_bars(ax, x, y, color, label):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if len(x_arr) > 1:
        dx = np.diff(x_arr)
        bar_width = np.median(dx) * AA_BAR_WIDTH_SCALE
    else:
        bar_width = 0.02

    ax.bar(
        x_arr,
        y_arr,
        width=bar_width,
        color=color,
        alpha=AA_BAR_ALPHA,
        edgecolor=color,
        linewidth=0.0,
        align="center",
        label=label,
        zorder=1,
    )


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    aa_path = os.path.join(script_dir, AA_FILE)
    cg_path = os.path.join(script_dir, CG_FILE)
    m3_path = os.path.join(script_dir, M3_FILE)
    output_fig_path = os.path.join(script_dir, OUTPUT_FIG)

    print(f"Reading: {aa_path}")
    aa_x, aa_y = read_xvg_xy(aa_path)

    print(f"Reading: {cg_path}")
    cg_x, cg_y = read_xvg_xy(cg_path)

    print(f"Reading: {m3_path}")
    m3_x, m3_y = read_xvg_xy(m3_path)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    plot_aa_distribution_as_bars(ax, aa_x, aa_y, AA_COLOR, "AA")
    plot_smoothed_distribution(ax, m3_x, m3_y, M3_COLOR, "M3", zorder_line=3, zorder_shade=2)
    plot_smoothed_distribution(ax, cg_x, cg_y, CG_COLOR, "CG", zorder_line=5, zorder_shade=4)

    if SHOW_TITLE:
        ax.set_title(FIG_TITLE, fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel(X_LABEL, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(Y_LABEL, fontsize=LABEL_FONT_SIZE)

    if X_AXIS_LIMITS is not None:
        ax.set_xlim(X_AXIS_LIMITS)
    if Y_AXIS_LIMITS is not None:
        ax.set_ylim(Y_AXIS_LIMITS)

    if X_MAJOR_TICK_STEP is not None:
        ax.xaxis.set_major_locator(MultipleLocator(X_MAJOR_TICK_STEP))
    if Y_MAJOR_TICK_STEP is not None:
        ax.yaxis.set_major_locator(MultipleLocator(Y_MAJOR_TICK_STEP))

    ax.tick_params(
        axis="both",
        labelsize=TICK_FONT_SIZE,
        width=TICK_WIDTH,
        length=TICK_LENGTH,
        direction="in",
        top=True,
        right=True,
    )
    ax.tick_params(axis="x", pad=X_TICK_PAD)
    ax.tick_params(axis="y", pad=Y_TICK_PAD)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LINE_WIDTH)

    if SHOW_LEGEND:
        handles, labels = ax.get_legend_handles_labels()
        desired_order = ["AA", "CG", "M3"]
        label_to_handle = dict(zip(labels, handles))
        ordered_handles = [label_to_handle[label] for label in desired_order if label in label_to_handle]
        ordered_labels = [label for label in desired_order if label in label_to_handle]
        ax.legend(ordered_handles, ordered_labels, fontsize=LEGEND_FONT_SIZE, frameon=False)

    fig.tight_layout()
    fig.savefig(output_fig_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {output_fig_path}")


if __name__ == "__main__":
    main()
