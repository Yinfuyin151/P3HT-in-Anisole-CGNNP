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

ANGLE_TYPES = ["RRR", "RDE", "DRRright", "DRRleft"]

# 如果想手动限制横纵坐标范围，在这里改。None or number
X_AXIS_LIMITS = None
Y_AXIS_LIMITS = None
# 如果想手动设置 x/y 轴主刻度间隔，在这里改。None 表示不手动设置
X_MAJOR_TICK_STEP = None
Y_MAJOR_TICK_STEP = None

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
LINE_WIDTH = 4
AA_BAR_ALPHA = 0.4
AA_BAR_WIDTH_SCALE = 0.90

X_LABEL = "Angle (degree)"
Y_LABEL = "Distribution"
# 是否显示图片上方标题。False 表示不显示
SHOW_TITLE = False
# 是否显示图例。True 表示显示，False 表示不显示
SHOW_LEGEND = False


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


def plot_aa_distribution_as_bars(ax, x, y, color, label):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if len(x_arr) > 1:
        dx = np.diff(x_arr)
        bar_width = np.median(dx) * AA_BAR_WIDTH_SCALE
    else:
        bar_width = 1.0

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


def plot_line_distribution(ax, x, y, color, label, zorder_line):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    ax.plot(
        x_arr,
        y_arr,
        color=color,
        linewidth=LINE_WIDTH,
        label=label,
        zorder=zorder_line,
    )


def draw_one_angle_type(script_dir, angle_type):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    aa_file = os.path.join(script_dir, f"AA_angle_dist{angle_type}.xvg")
    print(f"Reading: {aa_file}")
    aa_x, aa_y = read_xvg_xy(aa_file)
    plot_aa_distribution_as_bars(ax, aa_x, aa_y, AA_COLOR, "AA")

    m3_file = os.path.join(script_dir, f"M3_angle_dist{angle_type}.xvg")
    print(f"Reading: {m3_file}")
    m3_x, m3_y = read_xvg_xy(m3_file)
    plot_line_distribution(ax, m3_x, m3_y, M3_COLOR, "M3", zorder_line=3)

    cg_file = os.path.join(script_dir, f"CG_angle_dist{angle_type}.xvg")
    print(f"Reading: {cg_file}")
    cg_x, cg_y = read_xvg_xy(cg_file)
    plot_line_distribution(ax, cg_x, cg_y, CG_COLOR, "CG", zorder_line=5)

    if SHOW_TITLE:
        ax.set_title(f"Angle {angle_type} Distribution", fontsize=TITLE_FONT_SIZE)
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
    output_fig = os.path.join(script_dir, f"angle_{angle_type}_distribution_comparison.svg")
    fig.savefig(output_fig, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_fig}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for angle_type in ANGLE_TYPES:
        draw_one_angle_type(script_dir, angle_type)


if __name__ == "__main__":
    main()
