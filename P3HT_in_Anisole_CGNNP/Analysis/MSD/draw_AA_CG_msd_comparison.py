#!/usr/bin/env python3
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ============================================================
# 用户需要修改的参数
# ============================================================

# 输入文件
AA_CSV_FILE = "AAmonomer_ring3beads.csv"
CG_CSV_FILE = "CGmonomer_ring3beads.csv"

# 输出图片
OUTPUT_FIG = "AA_CG_msd_comparison.svg"

# CG 时间修正因子
# 当前 CG 轨迹时间把 ns 错写成了 ps，因此真实时间 = 文件中的时间 × 1000
CG_TIME_SCALE_FACTOR = 1000.0

# 使用哪个时间列作图
# 可选："time_ps" 或 "time_ns"
TIME_COLUMN = "time_ns"

# ============================================================
# 用户修改坐标范围和主刻度：这里改
# ------------------------------------------------------------
# - X_AXIS_LIMITS / Y_AXIS_LIMITS = None 表示自动
# - 也可以手动指定范围，例如 (0.0, 8.0)
# - X_MAJOR_TICK_STEP / Y_MAJOR_TICK_STEP = None 表示不手动设置主刻度间隔
# ============================================================
X_AXIS_LIMITS = (0.0, 8.0)
Y_AXIS_LIMITS = None
X_MAJOR_TICK_STEP = None
Y_MAJOR_TICK_STEP = None

# 列名
MSD_COLUMN = "msd_nm2"
SEM_COLUMN = "sem_nm2"

# 是否画 SEM 阴影
SHOW_SEM_SHADE = True
SEM_ALPHA = 0.3

# 图像样式
FONT_FAMILY = "Arial"
TITLE_FONT_SIZE = 38
LABEL_FONT_SIZE = 38
TICK_FONT_SIZE = 34
LEGEND_FONT_SIZE = 34
FIG_DPI = 600
FIG_WIDTH = 8
FIG_HEIGHT = 6
SPINE_LINE_WIDTH = 3
TICK_WIDTH = 4
TICK_LENGTH = 8
X_TICK_PAD = 6
Y_TICK_PAD = 8

AA_COLOR = "tab:blue"
CG_COLOR = "tab:red"
LINE_WIDTH = 5

FIG_TITLE = "AA vs CG MSD comparison"
X_LABEL = "Time (ns)"
Y_LABEL = "MSD (nm$^2$)"
SHOW_TITLE = False
SHOW_LEGEND = True


plt.rcParams["font.family"] = FONT_FAMILY


def read_msd_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    time_values = []
    msd_values = []
    sem_values = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = [TIME_COLUMN, MSD_COLUMN, SEM_COLUMN]
        missing = [col for col in required if col not in reader.fieldnames]
        if missing:
            raise KeyError(
                f"Missing columns in {csv_path}: {missing}. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            time_values.append(float(row[TIME_COLUMN]))
            msd_values.append(float(row[MSD_COLUMN]))
            sem_values.append(float(row[SEM_COLUMN]))

    return time_values, msd_values, sem_values


def plot_series(ax, time_values, msd_values, sem_values, color, label):
    ax.plot(
        time_values,
        msd_values,
        color=color,
        linewidth=LINE_WIDTH,
        label=label,
    )

    if SHOW_SEM_SHADE and any(v > 0 for v in sem_values):
        lower = [m - s for m, s in zip(msd_values, sem_values)]
        upper = [m + s for m, s in zip(msd_values, sem_values)]
        ax.fill_between(
            time_values,
            lower,
            upper,
            color=color,
            alpha=SEM_ALPHA,
            linewidth=0.0,
        )


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    aa_csv_path = os.path.join(script_dir, AA_CSV_FILE)
    cg_csv_path = os.path.join(script_dir, CG_CSV_FILE)
    output_fig_path = os.path.join(script_dir, OUTPUT_FIG)

    print(f"Reading AA CSV: {aa_csv_path}")
    aa_time, aa_msd, aa_sem = read_msd_csv(aa_csv_path)

    print(f"Reading CG CSV: {cg_csv_path}")
    cg_time, cg_msd, cg_sem = read_msd_csv(cg_csv_path)

    cg_time_corrected = [t * CG_TIME_SCALE_FACTOR for t in cg_time]

    print(f"CG time scale factor = {CG_TIME_SCALE_FACTOR}")
    print(f"Saving figure: {output_fig_path}")

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    plot_series(ax, aa_time, aa_msd, aa_sem, AA_COLOR, "AA")
    plot_series(ax, cg_time_corrected, cg_msd, cg_sem, CG_COLOR, "CG")

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

    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    if SHOW_LEGEND:
        ax.legend(fontsize=LEGEND_FONT_SIZE, frameon=False)

    plt.tight_layout()
    plt.savefig(output_fig_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()
