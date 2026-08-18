#!/usr/bin/env python3
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 用户需要修改的参数
# ============================================================

CSV_FILE = "metrics.csv"
OUTPUT_FIG = "lr_train_val_total_mse_vs_epoch.svg"

# 列名
EPOCH_COLUMN = "epoch"
LR_COLUMN = "lr"
TRAIN_LOSS_COLUMN = "train_total_mse_loss"
VAL_LOSS_COLUMN = "val_total_mse_loss"

# 图例显示名称
LR_LABEL = "LR"
TRAIN_LOSS_LABEL = "Train MSE"
VAL_LOSS_LABEL = "Val MSE"

# 图像样式
FONT_FAMILY = "Arial"
TITLE_FONT_SIZE = 34
LABEL_FONT_SIZE = 34
TICK_FONT_SIZE = 30
LEGEND_FONT_SIZE = 30
FIG_DPI = 600

LR_COLOR = "tab:grey"
TRAIN_LOSS_COLOR = "tab:red"
VAL_LOSS_COLOR = "tab:blue"

LR_LINEWIDTH = 4
LOSS_LINEWIDTH = 4

# 在 Val MSE 曲线上标出最终选用的 epoch：这里改
HIGHLIGHT_EPOCH = 883
HIGHLIGHT_MARKER = "*"
HIGHLIGHT_COLOR = "tab:blue"
HIGHLIGHT_MARKER_SIZE = 350
HIGHLIGHT_EDGE_COLOR = "none"
HIGHLIGHT_EDGE_WIDTH = 0.0

# 在 x=HIGHLIGHT_EPOCH 的位置添加一根向下箭头，并标注文字：这里改
HIGHLIGHT_ARROW_TEXT = "epoch=883"
HIGHLIGHT_ARROW_COLOR = "tab:blue"
HIGHLIGHT_ARROW_TEXT_SIZE = 30
HIGHLIGHT_ARROW_LINE_WIDTH = 1.8
# 箭头顶部相对星星再向上移动多少，采用右侧 y 轴数据单位
HIGHLIGHT_ARROW_TOP_OFFSET = 0.1
# 文字相对箭头顶部的水平偏移量，单位是 x 轴的 epoch
# 负值表示向左移，正值表示向右移
HIGHLIGHT_TEXT_X_OFFSET = -100
# 文字相对箭头顶部的竖直偏移量，采用右侧 y 轴数据单位
# 正值表示向上移，负值表示向下移
HIGHLIGHT_TEXT_Y_OFFSET = 0.015

FIG_TITLE = " "
X_LABEL = "Epoch"
LEFT_Y_LABEL = "Learning rate"
RIGHT_Y_LABEL = "Total MSE loss"
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
FIG_WIDTH = 9
FIG_HEIGHT = 6

# 线条和边框样式
# 用户可在这里修改刻度线宽度和长度
# TICK_WIDTH：刻度线粗细
# TICK_LENGTH：刻度线长度
SPINE_LINE_WIDTH = 3
TICK_WIDTH = 4
TICK_LENGTH = 8

plt.rcParams["font.family"] = FONT_FAMILY


def load_metrics(csv_file):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    epochs = []
    lrs = []
    train_losses = []
    val_losses = []

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_columns = [
            EPOCH_COLUMN,
            LR_COLUMN,
            TRAIN_LOSS_COLUMN,
            VAL_LOSS_COLUMN,
        ]
        missing = [col for col in required_columns if col not in reader.fieldnames]
        if missing:
            raise KeyError(
                f"Missing columns in CSV: {missing}. Available columns: {reader.fieldnames}"
            )

        for row in reader:
            epoch_str = row[EPOCH_COLUMN].strip()
            lr_str = row[LR_COLUMN].strip()
            train_loss_str = row[TRAIN_LOSS_COLUMN].strip()
            val_loss_str = row[VAL_LOSS_COLUMN].strip()

            if not epoch_str or not lr_str:
                continue

            epochs.append(float(epoch_str))
            lrs.append(float(lr_str))

            train_losses.append(float(train_loss_str) if train_loss_str else float("nan"))
            val_losses.append(float(val_loss_str) if val_loss_str else float("nan"))

    if not epochs:
        raise RuntimeError("No valid rows were read from the CSV file.")

    return epochs, lrs, train_losses, val_losses


def draw_plot(epochs, lrs, train_losses, val_losses, output_fig):
    fig, ax_lr = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    ax_loss = ax_lr.twinx()
    # 把 loss 这一层放到更上面，保证高亮星星不会被曲线或另一组坐标轴遮住
    ax_lr.set_zorder(2)
    ax_loss.set_zorder(3)
    ax_lr.patch.set_facecolor("white")
    ax_lr.patch.set_alpha(1.0)
    ax_loss.patch.set_alpha(0.0)

    lr_line = ax_lr.plot(
        epochs,
        lrs,
        color=LR_COLOR,
        linewidth=LR_LINEWIDTH,
        label=LR_LABEL,
    )[0]

    train_line = ax_loss.plot(
        epochs,
        train_losses,
        color=TRAIN_LOSS_COLOR,
        linewidth=LOSS_LINEWIDTH,
        label=TRAIN_LOSS_LABEL,
    )[0]

    val_line = ax_loss.plot(
        epochs,
        val_losses,
        color=VAL_LOSS_COLOR,
        linewidth=LOSS_LINEWIDTH,
        label=VAL_LOSS_LABEL,
    )[0]

    # 在 Val MSE 曲线上标出被选中的 epoch，星星放在最上层避免被遮挡
    if HIGHLIGHT_EPOCH in epochs:
        highlight_index = epochs.index(HIGHLIGHT_EPOCH)
        highlight_val_loss = val_losses[highlight_index]
        ax_loss.scatter(
            [HIGHLIGHT_EPOCH],
            [highlight_val_loss],
            s=HIGHLIGHT_MARKER_SIZE,
            marker=HIGHLIGHT_MARKER,
            color=HIGHLIGHT_COLOR,
            edgecolors=HIGHLIGHT_EDGE_COLOR,
            linewidths=HIGHLIGHT_EDGE_WIDTH,
            zorder=100,
            clip_on=False,
        )
    else:
        print(f"[WARN] HIGHLIGHT_EPOCH={HIGHLIGHT_EPOCH} not found in the CSV data.")

    ax_lr.set_title(FIG_TITLE, fontsize=TITLE_FONT_SIZE)
    ax_lr.set_xlabel(X_LABEL, fontsize=LABEL_FONT_SIZE)
    ax_lr.set_ylabel(LEFT_Y_LABEL, fontsize=LABEL_FONT_SIZE, color="black")
    ax_loss.set_ylabel(RIGHT_Y_LABEL, fontsize=LABEL_FONT_SIZE, color="black")

    ax_lr.tick_params(
        axis="both",
        labelsize=TICK_FONT_SIZE,
        width=TICK_WIDTH,
        length=TICK_LENGTH,
        direction="in",
        top=True,
        right=False,
    )
    ax_lr.tick_params(axis="y", colors="black")
    ax_loss.tick_params(
        axis="y",
        labelsize=TICK_FONT_SIZE,
        width=TICK_WIDTH,
        length=TICK_LENGTH,
        direction="in",
        left=False,
        right=True,
        colors="black",
    )

    for spine in ax_lr.spines.values():
        spine.set_linewidth(SPINE_LINE_WIDTH)
    for spine in ax_loss.spines.values():
        spine.set_linewidth(SPINE_LINE_WIDTH)

    if HIGHLIGHT_EPOCH in epochs:
        y_min, y_max = ax_loss.get_ylim()
        y_span = y_max - y_min
        arrow_top_y = min(
            highlight_val_loss + HIGHLIGHT_ARROW_TOP_OFFSET * y_span,
            y_max - 0.04 * y_span,
        )
        text_y = min(
            arrow_top_y + HIGHLIGHT_TEXT_Y_OFFSET * y_span,
            y_max - 0.02 * y_span,
        )
        ax_lr.annotate(
            "",
            xy=(HIGHLIGHT_EPOCH, highlight_val_loss),
            xytext=(HIGHLIGHT_EPOCH, arrow_top_y),
            xycoords=ax_loss.transData,
            textcoords=ax_loss.transData,
            arrowprops=dict(
                arrowstyle="-|>",
                color=HIGHLIGHT_ARROW_COLOR,
                lw=HIGHLIGHT_ARROW_LINE_WIDTH,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=90,
            annotation_clip=False,
        )
        ax_loss.text(
            HIGHLIGHT_EPOCH + HIGHLIGHT_TEXT_X_OFFSET,
            text_y,
            HIGHLIGHT_ARROW_TEXT,
            ha="center",
            va="bottom",
            color=HIGHLIGHT_ARROW_COLOR,
            fontsize=HIGHLIGHT_ARROW_TEXT_SIZE,
            zorder=91,
            clip_on=True,
        )

    lines = [lr_line, train_line, val_line]
    labels = [line.get_label() for line in lines]
    ax_lr.legend(
        lines,
        labels,
        loc="best",
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        facecolor="none",
    )

    fig.tight_layout()
    fig.savefig(output_fig, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, CSV_FILE)
    output_fig_path = os.path.join(script_dir, OUTPUT_FIG)

    print(f"Reading CSV: {csv_path}")
    epochs, lrs, train_losses, val_losses = load_metrics(csv_path)

    print(f"Number of rows used: {len(epochs)}")
    print(f"Saving figure: {output_fig_path}")
    draw_plot(epochs, lrs, train_losses, val_losses, output_fig_path)

    print("Done.")


if __name__ == "__main__":
    main()
