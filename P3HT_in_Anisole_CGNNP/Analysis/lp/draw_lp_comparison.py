#!/usr/bin/env python3
import csv
import math
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    MultipleLocator = None
    HAS_MATPLOTLIB = False


# ============================================================
# 用户需要修改的参数
# ============================================================

AA_CSV_FILE = "AA_lp_result.csv"
CG_CSV_FILE = "CG_lp_result.csv"
M3_CSV_FILE = "M3_lp_result.csv"
AA_REF_CSV_FILE = "T6_32mer_digitized_from_image.csv"

OUTPUT_FIG = "AA_CG_M3_lp_comparison.svg"

# 如果想手动限制横纵坐标范围，在这里改。None or number
# 当前先默认把 x 轴设为 0 到 20
X_AXIS_LIMITS = (0.0, 20.0)
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
FIG_WIDTH = 15
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
AA_REF_COLOR = "tab:orange"
LINE_WIDTH = 7
AA_LINE_WIDTH = 7
CG_LINE_WIDTH = 7
M3_LINE_WIDTH = 7
AA_REF_LINE_WIDTH = 7
POINT_SIZE = 180
AA_POINT_SIZE = 180
CG_POINT_SIZE = 180
M3_POINT_SIZE = 180
AA_REF_POINT_SIZE = 180
POINT_EDGE_WIDTH = 5
AA_POINT_EDGE_WIDTH = 5
CG_POINT_EDGE_WIDTH = 5
M3_POINT_EDGE_WIDTH = 5
AA_REF_POINT_EDGE_WIDTH = 5
FIT_LINESTYLE = "--"
AA_FIT_LINESTYLE = "-"
CG_FIT_LINESTYLE = "--"
M3_FIT_LINESTYLE = "--"
AA_REF_FIT_LINESTYLE = "--"
AA_LINE_ALPHA = 0.95
CG_LINE_ALPHA = 0.95
M3_LINE_ALPHA = 0.95
AA_REF_LINE_ALPHA = 0.85
HLINE_COLOR = "0.65"
GUIDE_LINESTYLE = "-"
SHOW_M3_VERTICAL_GUIDE = True
SHOW_AA_REF_VERTICAL_GUIDE = True
# 是否把 Martini 3 的拟合曲线沿 x 轴正方向外推到超过 e^-1 辅助线的位置
EXTEND_M3_FIT_TO_CROSSING = True
# 当 Martini 3 外推后，x 轴在 crossing 点右侧额外保留的空白
M3_X_MARGIN = 1.0

#箭头的参数设置 还有个mutation_scale
ARROW_LINE_WIDTH = 5
ARROW_HEAD_WIDTH = 0.8
ARROW_HEAD_LENGTH = 0.12
ARROW_BASE_Y = 0.15
ARROW_TIP_Y = 0.02
AA_ARROW_BASE_Y = 0.28
AA_ARROW_TIP_Y = 0.15

# 参考文献数据在图例中的名称：这里可由用户自行修改
AA_REF_LABEL = "Previous work"

FIG_TITLE = "Persistence Length Correlation Comparison"
X_LABEL = "Monomer index n"
Y_LABEL = r"$\langle \cos \theta_n \rangle$"
# 是否显示图片上方标题。False 表示不显示
SHOW_TITLE = False
# 是否显示图例。True 表示显示，False 表示不显示
SHOW_LEGEND = True

if HAS_MATPLOTLIB:
    plt.rcParams["font.family"] = FONT_FAMILY


def read_lp_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    n_vals = []
    cos_vals = []
    fit_vals = []
    fit_used = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = ["n", "cos_avg", "fit_exp_decay", "fit_used"]
        missing = [col for col in required if col not in reader.fieldnames]
        if missing:
            raise KeyError(
                f"Missing columns in {csv_path}: {missing}. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            n_vals.append(int(row["n"]))
            cos_vals.append(float(row["cos_avg"]))
            fit_vals.append(float(row["fit_exp_decay"]))
            fit_used.append(int(row["fit_used"]))

    return n_vals, cos_vals, fit_vals, fit_used


def estimate_np_from_fit(n_vals, fit_vals, fit_used):
    pairs = [(n, y) for n, y, used in zip(n_vals, fit_vals, fit_used) if used == 1 and y > 0]

    if len(pairs) < 2:
        return None

    x = np.asarray([p[0] for p in pairs], dtype=float)
    logy = np.log(np.asarray([p[1] for p in pairs], dtype=float))
    slope, intercept = np.polyfit(x, logy, 1)
    if slope >= 0:
        return None
    return -1.0 / slope


def fit_exp_decay_params(n_vals, fit_vals, fit_used):
    pairs = [(n, y) for n, y, used in zip(n_vals, fit_vals, fit_used) if used == 1 and y > 0]
    if len(pairs) < 2:
        return None

    x = np.asarray([p[0] for p in pairs], dtype=float)
    logy = np.log(np.asarray([p[1] for p in pairs], dtype=float))
    slope, intercept = np.polyfit(x, logy, 1)
    return slope, intercept


def plot_one_system(
    ax,
    csv_path,
    color,
    label,
    marker,
    draw_vertical_guide,
    extend_fit_to_crossing=False,
    arrow_base_y=ARROW_BASE_Y,
    arrow_tip_y=ARROW_TIP_Y,
    point_size=POINT_SIZE,
    point_edge_width=POINT_EDGE_WIDTH,
    line_width=LINE_WIDTH,
    fit_linestyle=FIT_LINESTYLE,
    line_alpha=1.0,
    scatter_zorder=3,
    line_zorder=2,
):
    n_vals, cos_vals, fit_vals, fit_used = read_lp_csv(csv_path)

    ax.scatter(
        n_vals,
        cos_vals,
        s=point_size,
        facecolors="none",
        edgecolors=color,
        linewidths=point_edge_width,
        marker=marker,
        label=label,
        zorder=scatter_zorder,
    )

    npers = estimate_np_from_fit(n_vals, fit_vals, fit_used)
    fit_x = np.asarray(n_vals, dtype=float)
    fit_y = np.asarray(fit_vals, dtype=float)

    if extend_fit_to_crossing and npers is not None:
        fit_params = fit_exp_decay_params(n_vals, fit_vals, fit_used)
        if fit_params is not None:
            slope, intercept = fit_params
            fit_x_end = max(float(max(n_vals)), npers + M3_X_MARGIN)
            fit_x = np.linspace(float(min(n_vals)), fit_x_end, 400)
            fit_y = np.exp(intercept + slope * fit_x)

    ax.plot(
        fit_x,
        fit_y,
        color=color,
        linewidth=line_width,
        linestyle=fit_linestyle,
        alpha=line_alpha,
        label="_nolegend_",
        zorder=line_zorder,
    )

    if draw_vertical_guide and npers is not None:
        ax.annotate(
            "",
            xy=(npers, arrow_tip_y),
            xytext=(npers, arrow_base_y),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=ARROW_LINE_WIDTH,
                mutation_scale=23,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=1,
        )

    fit_x_max = float(np.max(fit_x)) if len(fit_x) > 0 else float(max(n_vals))
    return npers, fit_x_max


def print_arrow_indices(arrow_indices):
    print("Arrow-indicated monomer indices:")
    for label, npers in arrow_indices:
        if npers is None:
            print(f"  {label}: unavailable")
            continue

        nearest_index = int(round(npers))
        print(f"  {label}: n = {npers:.3f} (nearest integer index = {nearest_index})")


def get_default_y_limits():
    return 0.0, 1.02


def collect_arrow_indices(script_dir):
    datasets = [
        ("AA", os.path.join(script_dir, AA_CSV_FILE)),
        ("CG", os.path.join(script_dir, CG_CSV_FILE)),
        ("Martini 3", os.path.join(script_dir, M3_CSV_FILE)),
        (AA_REF_LABEL, os.path.join(script_dir, AA_REF_CSV_FILE)),
    ]

    arrow_indices = []
    for label, csv_path in datasets:
        n_vals, _, fit_vals, fit_used = read_lp_csv(csv_path)
        npers = estimate_np_from_fit(n_vals, fit_vals, fit_used)
        arrow_indices.append((label, npers))
    return arrow_indices


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    aa_csv_path = os.path.join(script_dir, AA_CSV_FILE)
    cg_csv_path = os.path.join(script_dir, CG_CSV_FILE)
    m3_csv_path = os.path.join(script_dir, M3_CSV_FILE)
    aa_ref_csv_path = os.path.join(script_dir, AA_REF_CSV_FILE)
    output_fig_path = os.path.join(script_dir, OUTPUT_FIG)

    if not HAS_MATPLOTLIB:
        print("matplotlib is not installed in the current Python environment.")
        print("Skipping figure generation and printing arrow-indicated indices only.")
        print_arrow_indices(collect_arrow_indices(script_dir))
        return

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax.axhline(
        1.0 / math.e,
        color=HLINE_COLOR,
        linewidth=1.4,
        linestyle=GUIDE_LINESTYLE,
        zorder=0,
    )

    aa_npers, aa_fit_xmax = plot_one_system(
        ax,
        aa_csv_path,
        AA_COLOR,
        "AA",
        marker="o",
        draw_vertical_guide=True,
        arrow_base_y=AA_ARROW_BASE_Y,
        arrow_tip_y=AA_ARROW_TIP_Y,
        point_size=AA_POINT_SIZE,
        point_edge_width=AA_POINT_EDGE_WIDTH,
        line_width=AA_LINE_WIDTH,
        fit_linestyle=AA_FIT_LINESTYLE,
        line_alpha=AA_LINE_ALPHA,
        scatter_zorder=5,
        line_zorder=1,
    )
    cg_npers, cg_fit_xmax = plot_one_system(
        ax,
        cg_csv_path,
        CG_COLOR,
        "CG",
        marker="s",
        draw_vertical_guide=True,
        point_size=CG_POINT_SIZE,
        point_edge_width=CG_POINT_EDGE_WIDTH,
        line_width=CG_LINE_WIDTH,
        fit_linestyle=CG_FIT_LINESTYLE,
        line_alpha=CG_LINE_ALPHA,
        scatter_zorder=3,
        line_zorder=4,
    )
    m3_npers, m3_fit_xmax = plot_one_system(
        ax,
        m3_csv_path,
        M3_COLOR,
        "M3",
        marker="^",
        draw_vertical_guide=SHOW_M3_VERTICAL_GUIDE,
        extend_fit_to_crossing=EXTEND_M3_FIT_TO_CROSSING,
        point_size=M3_POINT_SIZE,
        point_edge_width=M3_POINT_EDGE_WIDTH,
        line_width=M3_LINE_WIDTH,
        fit_linestyle=M3_FIT_LINESTYLE,
        line_alpha=M3_LINE_ALPHA,
    )
    aa_ref_npers, aa_ref_fit_xmax = plot_one_system(
        ax,
        aa_ref_csv_path,
        AA_REF_COLOR,
        AA_REF_LABEL,
        marker="D",
        draw_vertical_guide=SHOW_AA_REF_VERTICAL_GUIDE,
        point_size=AA_REF_POINT_SIZE,
        point_edge_width=AA_REF_POINT_EDGE_WIDTH,
        line_width=AA_REF_LINE_WIDTH,
        fit_linestyle=AA_REF_FIT_LINESTYLE,
        line_alpha=AA_REF_LINE_ALPHA,
    )

    if SHOW_TITLE:
        ax.set_title(FIG_TITLE, fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel(X_LABEL, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(Y_LABEL, fontsize=LABEL_FONT_SIZE)

    if X_AXIS_LIMITS is not None:
        x_min, x_max = X_AXIS_LIMITS
        auto_xmax = max(aa_fit_xmax, cg_fit_xmax, m3_fit_xmax, aa_ref_fit_xmax)
        if m3_npers is not None:
            auto_xmax = max(auto_xmax, m3_npers + M3_X_MARGIN)
        ax.set_xlim((x_min, max(x_max, auto_xmax)))
    else:
        auto_xmax = max(aa_fit_xmax, cg_fit_xmax, m3_fit_xmax, aa_ref_fit_xmax)
        if m3_npers is not None:
            auto_xmax = max(auto_xmax, m3_npers + M3_X_MARGIN)
        ax.set_xlim(0.0, auto_xmax)
    if Y_AXIS_LIMITS is not None:
        ax.set_ylim(Y_AXIS_LIMITS)
    else:
        ax.set_ylim(get_default_y_limits())

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
        desired_order = ["AA", "CG", "M3", AA_REF_LABEL]
        label_to_handle = dict(zip(labels, handles))
        ordered_handles = [label_to_handle[label] for label in desired_order if label in label_to_handle]
        ordered_labels = [label for label in desired_order if label in label_to_handle]
        ax.legend(ordered_handles, ordered_labels, fontsize=LEGEND_FONT_SIZE, frameon=False)

    fig.tight_layout()
    fig.savefig(output_fig_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {output_fig_path}")
    print_arrow_indices(
        [
            ("AA", aa_npers),
            ("CG", cg_npers),
            ("M3", m3_npers),
            (AA_REF_LABEL, aa_ref_npers),
        ]
    )


if __name__ == "__main__":
    main()
