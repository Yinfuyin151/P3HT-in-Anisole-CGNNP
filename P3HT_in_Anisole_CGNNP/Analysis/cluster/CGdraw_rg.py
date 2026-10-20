#!/usr/bin/env python3
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")  # 不打开图形界面，适合服务器 / nohup / 无 Xmanager 环境
import matplotlib.pyplot as plt

import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array


# ============================================================
# User settings
# ============================================================

# Input files
TPR_FILE = "CG.tpr"
XTC_FILE = "molcenterclusterCG.xtc"

# Output folder in the current working directory
OUT_DIR = "10chainsRgCG"

# Number of polymer chains
N_CHAINS = 10

# Number of CG beads per chain
# 例如：10 条链，每条 40 beads，总共 400 beads
BEADS_PER_CHAIN = 40

# 每个 repeat unit 的 bead 数
# 例如：每 5 个 beads 是一个 repeat unit
REPEAT_UNIT_SIZE = 5

# 每个 repeat unit 中构成五元环 / backbone 的 bead 编号，0-based
# 第 1,2,3 个 bead 对应 Python index 0,1,2
RING_BEAD_OFFSETS = [0, 1, 2]

# 如果 tpr/xtc 里只包含 polymer，则保持 "all"
# 如果前 400 个 atoms 是 polymer，可以写 "index 0:400"
POLYMER_SELECTION = "all"

# 图像设置
FONT_FAMILY = "Arial"
TITLE_FONT_SIZE = 38
LABEL_FONT_SIZE = 38
TICK_FONT_SIZE = 34
LEGEND_FONT_SIZE = 34
FIG_DPI = 600
FIG_FORMAT = "svg"  # 可改成 "pdf", "svg", "tif"
# ============================================================
# 用户修改整张图的长宽：这里改
# ------------------------------------------------------------
# FIG_WIDTH：整张图总宽度
# FIG_HEIGHT：整张图总高度
# 单位都是英寸（inch）
# ============================================================
FIG_WIDTH = 10
FIG_HEIGHT = 6

# 线条和边框样式
# TICK_WIDTH：刻度线粗细
# TICK_LENGTH：刻度线长度
# X_TICK_PAD / Y_TICK_PAD：刻度数字离轴线距离
SPINE_LINE_WIDTH = 3
TICK_WIDTH = 4
TICK_LENGTH = 8
X_TICK_PAD = 6
Y_TICK_PAD = 8

# 是否显示标题和图例
SHOW_TITLE = False
SHOW_LEGEND = False
# 图例位置设置
# LEGEND_LOC：图例锚点位置，例如 "upper right"、"upper left"、"best"
# LEGEND_BBOX_TO_ANCHOR：图例位置微调，采用坐标轴相对坐标
# - (0, 0) 是左下角
# - (1, 1) 是右上角
# 如果不想微调，可保持为 None
LEGEND_LOC = "upper right"
LEGEND_BBOX_TO_ANCHOR = None

# 图中字母标签设置
# PANEL_TAG_X / PANEL_TAG_Y 使用坐标轴相对坐标：
# - (0, 0) 是左下角
# - (1, 1) 是右上角
SHOW_PANEL_TAG = False
PANEL_TAG_TEXT = "CG"
PANEL_TAG_X = 0.08
PANEL_TAG_Y = 0.10
PANEL_TAG_FONT_SIZE = 28

# 轴范围设置
# None 表示自动
X_AXIS_LIMITS = (0, 44)
LEFT_Y_AXIS_LIMITS = (0, 5.5)
RIGHT_Y_AXIS_LIMITS = (0, 11)

# 颜色和线宽
LEFT_RAW_COLOR = "0.65"
LEFT_BLOCK_COLOR = "tab:red"
RIGHT_RAW_COLOR = "#8ecae6"
RIGHT_BLOCK_COLOR = "tab:blue"
RAW_LINE_WIDTH = 0.9
BLOCK_LINE_WIDTH = 5
LEFT_SHADE_ALPHA = 0.18
RIGHT_SHADE_ALPHA = 0.16

# 轴标签
X_LABEL = "Time (ns)"
LEFT_Y_LABEL = "Radius of gyration of all chains (nm)"
RIGHT_Y_LABEL = r"Largest cluster size ($N_{\mathrm{chain}}$)"
# 两张图各自使用的左 y 轴标题
LEFT_Y_LABEL_ALL_BEADS = "Rg of all chains (nm)"
LEFT_Y_LABEL_RING_COM = "Rg of ring COMs (nm)"

# 图例文字
RG_LEGEND_LABEL = "Rg"
LC_LEGEND_LABEL = "$N_{\mathrm{chain}}$"

# ------------------------------------------------------------
# 1. 采样间隔设置
# ------------------------------------------------------------
# 每隔多少帧进行一次 Rg 计算。
# 例如 FRAME_STRIDE = 100 表示只处理第 0, 100, 200, ... 帧。
# 如果想恢复逐帧分析，把这里改成 1；如果想更稀疏，把数值改大。
FRAME_STRIDE = 100

# ------------------------------------------------------------
# 2. block average 窗口大小设置
# ------------------------------------------------------------
# 每多少个“已经采样出来的数据点”分成一组做平均。
# 例如 FRAME_STRIDE = 100 且 BLOCK_SIZE_POINTS = 10，
# 就表示每 10 个采样点做一个 block average，相当于每 1000 帧做一个平均。
BLOCK_SIZE_POINTS = 10

# ------------------------------------------------------------
# 3. 作图使用的轨迹时长设置
# ------------------------------------------------------------
# 只使用前多少 ns 的轨迹来做图。
# 例如 PLOT_MAX_TIME_NS = 50.0 表示只画 0-50 ns。
# 如果想使用完整轨迹，把这里改成 None。
PLOT_MAX_TIME_NS = 45

# ------------------------------------------------------------
# 轨迹时间换算设置
# ------------------------------------------------------------
# MDAnalysis 读到的 ts.time 数值有时会继承错误的轨迹时间单位。
# 你的当前轨迹显示为 100 ps，但真实物理时间应为 100 ns，
# 因此这里把 ts.time 直接当作 ns 使用：time_ns = ts.time * 1.0。
#
# 用户修改说明：
# - 当前这条错误标记的轨迹：保持 TRAJECTORY_TIME_TO_NS_FACTOR = 1.0
# - 如果以后轨迹时间是正常 ps，想换算到 ns，则改成 0.001
TRAJECTORY_TIME_TO_NS_FACTOR = 1.0

# 每隔多少帧在终端输出一次进度
PRINT_EVERY = 1000

# ------------------------------------------------------------
# 4. largest cluster nchain 计算设置
# ------------------------------------------------------------
# 判断两条链是否接触：
# 只要两条链之间“任意一对”ring COM 距离 < CLUSTER_CUTOFF_NM，
# 就认为这两条链接触，并基于 chain-chain contact graph 找 largest cluster。
#
# 用户修改说明：
# - 单位是 nm；
# - 例如想用 1.0 nm 判定近接触 cluster，就设为 1.0；
# - 例如想用 1.3 或 1.5 nm 判定中程连通 cluster，就改成 1.3 或 1.5。
CLUSTER_CUTOFF_NM = 1.5

# 是否使用 PBC 最小镜像距离判断 chain-chain contact。
# 如果轨迹已经做过 mol + cluster，通常可以先用 False；
# 如果担心团簇跨盒子边界，可以改成 True。
USE_PBC_FOR_CONTACT = False

# ------------------------------------------------------------
# 5. largest cluster nchain 独立采样和平滑设置
# ------------------------------------------------------------
# 这里的两个参数只控制 largest cluster nchain 曲线，不影响 Rg 曲线。
#
# CLUSTER_FRAME_STRIDE:
# 每隔多少帧计算一次 largest cluster。
# 例如 CLUSTER_FRAME_STRIDE = 50 表示只处理第 0, 50, 100, ... 帧。
CLUSTER_FRAME_STRIDE = 150

# CLUSTER_BLOCK_SIZE_POINTS:
# 每多少个“cluster 采样点”分成一组做 block average 和 mean +/- std。
# 这个参数和上面的 BLOCK_SIZE_POINTS 相互独立。
CLUSTER_BLOCK_SIZE_POINTS = 2


# ============================================================
# Derived settings
# ============================================================

if BEADS_PER_CHAIN % REPEAT_UNIT_SIZE != 0:
    raise ValueError(
        "BEADS_PER_CHAIN must be divisible by REPEAT_UNIT_SIZE. "
        f"Got BEADS_PER_CHAIN={BEADS_PER_CHAIN}, "
        f"REPEAT_UNIT_SIZE={REPEAT_UNIT_SIZE}"
    )

if FRAME_STRIDE < 1:
    raise ValueError(f"FRAME_STRIDE must be >= 1. Got {FRAME_STRIDE}.")

if BLOCK_SIZE_POINTS < 1:
    raise ValueError(
        f"BLOCK_SIZE_POINTS must be >= 1. Got {BLOCK_SIZE_POINTS}."
    )

if CLUSTER_FRAME_STRIDE < 1:
    raise ValueError(
        f"CLUSTER_FRAME_STRIDE must be >= 1. Got {CLUSTER_FRAME_STRIDE}."
    )

if CLUSTER_BLOCK_SIZE_POINTS < 1:
    raise ValueError(
        "CLUSTER_BLOCK_SIZE_POINTS must be >= 1. "
        f"Got {CLUSTER_BLOCK_SIZE_POINTS}."
    )

if PLOT_MAX_TIME_NS is not None and PLOT_MAX_TIME_NS <= 0:
    raise ValueError(
        f"PLOT_MAX_TIME_NS must be positive or None. Got {PLOT_MAX_TIME_NS}."
    )

N_REPEAT_UNITS_PER_CHAIN = BEADS_PER_CHAIN // REPEAT_UNIT_SIZE
CLUSTER_CUTOFF_A = CLUSTER_CUTOFF_NM * 10.0

plt.rcParams["font.family"] = FONT_FAMILY


# ============================================================
# Functions
# ============================================================

def calc_rg_equal_mass(positions):
    """
    Calculate equal-mass radius of gyration.

    Parameters
    ----------
    positions : np.ndarray, shape = (N, 3)
        Coordinates in Angstrom.

    Returns
    -------
    rg_A : float
        Radius of gyration in Angstrom.
    """
    com = positions.mean(axis=0)
    rg2 = np.mean(np.sum((positions - com) ** 2, axis=1))
    return np.sqrt(rg2)


def get_ring_coms(chain_atoms):
    """
    Calculate ring/backbone COMs for one polymer chain.

    这里默认 RING_BEAD_OFFSETS 中的 CG beads 等质量，所以使用几何中心。
    如果希望使用质量加权 COM，可以把：
        ring_com = ring_atoms.positions.mean(axis=0)
    改成：
        ring_com = ring_atoms.center_of_mass()
    """
    ring_coms = []

    for unit_id in range(N_REPEAT_UNITS_PER_CHAIN):
        base = unit_id * REPEAT_UNIT_SIZE
        indices = [base + offset for offset in RING_BEAD_OFFSETS]

        ring_atoms = chain_atoms[indices]
        ring_com = ring_atoms.positions.mean(axis=0)

        ring_coms.append(ring_com)

    return np.array(ring_coms)


def find_clusters(contact_matrix):
    """
    Find connected components from chain-chain contact matrix.

    Parameters
    ----------
    contact_matrix : np.ndarray, shape = (N_CHAINS, N_CHAINS)
        Boolean matrix.
        contact_matrix[i, j] = True means chain i and chain j are connected.

    Returns
    -------
    clusters : list[list[int]]
        Each cluster is a list of chain indices, starting from 0.
    """
    visited = np.zeros(N_CHAINS, dtype=bool)
    clusters = []

    for i in range(N_CHAINS):
        if visited[i]:
            continue

        stack = [i]
        visited[i] = True
        cluster = []

        while stack:
            current = stack.pop()
            cluster.append(current)

            neighbors = np.where(contact_matrix[current])[0]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)

        clusters.append(sorted(cluster))

    return clusters


def calc_largest_cluster_nchain(ring_coms_all_chains, box):
    """
    Calculate largest cluster size from interchain ring-COM contacts.
    """
    contact_matrix = np.zeros((N_CHAINS, N_CHAINS), dtype=bool)
    np.fill_diagonal(contact_matrix, True)

    for i in range(N_CHAINS):
        ring_pos_i = ring_coms_all_chains[i]

        for j in range(i + 1, N_CHAINS):
            ring_pos_j = ring_coms_all_chains[j]

            dists = distance_array(ring_pos_i, ring_pos_j, box=box)
            min_dist_A = dists.min()

            if min_dist_A < CLUSTER_CUTOFF_A:
                contact_matrix[i, j] = True
                contact_matrix[j, i] = True

    clusters = find_clusters(contact_matrix)
    largest_cluster = max(clusters, key=lambda x: (len(x), -min(x)))
    return len(largest_cluster)


def block_average(time_ns, values, block_size_points):
    """
    Calculate block-averaged mean and standard deviation.

    The last incomplete block is kept, because it still contains valid data.
    """
    block_time = []
    block_mean = []
    block_std = []

    for start in range(0, len(values), block_size_points):
        end = min(start + block_size_points, len(values))
        t_block = time_ns[start:end]
        v_block = values[start:end]

        block_time.append(np.mean(t_block))
        block_mean.append(np.mean(v_block))
        block_std.append(np.std(v_block, ddof=0))

    return (
        np.array(block_time),
        np.array(block_mean),
        np.array(block_std),
    )


def plot_raw_blockavg(
    time_ns,
    values,
    ylabel,
    title,
    filename,
    cluster_time_ns=None,
    largest_cluster_nchain=None,
):
    """
    Plot raw Rg(t), block-averaged Rg(t), and mean +/- std shaded band.
    """
    block_time, block_mean, block_std = block_average(
        time_ns,
        values,
        BLOCK_SIZE_POINTS,
    )

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Raw sampled trajectory: light gray, low alpha.
    ax.plot(
        time_ns,
        values,
        color=LEFT_RAW_COLOR,
        alpha=0.35,
        linewidth=RAW_LINE_WIDTH,
        label="_nolegend_",
    )

    # Block average: dark, thick line.
    ax.plot(
        block_time,
        block_mean,
        color=LEFT_BLOCK_COLOR,
        linewidth=BLOCK_LINE_WIDTH,
        label=RG_LEGEND_LABEL,
    )

    # Shaded band: mean +/- std within each block.
    ax.fill_between(
        block_time,
        block_mean - block_std,
        block_mean + block_std,
        color=LEFT_BLOCK_COLOR,
        alpha=LEFT_SHADE_ALPHA,
        linewidth=0,
        label="_nolegend_",
    )

    ax.set_xlabel(X_LABEL, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel if ylabel else LEFT_Y_LABEL, fontsize=LABEL_FONT_SIZE, color="black")
    if SHOW_TITLE:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    if X_AXIS_LIMITS is not None:
        ax.set_xlim(X_AXIS_LIMITS)
    if LEFT_Y_AXIS_LIMITS is not None:
        ax.set_ylim(LEFT_Y_AXIS_LIMITS)

    ax.tick_params(
        axis="both",
        labelsize=TICK_FONT_SIZE,
        width=TICK_WIDTH,
        length=TICK_LENGTH,
        direction="in",
        top=True,
        right=False,
    )
    ax.tick_params(axis="x", pad=X_TICK_PAD)
    ax.tick_params(axis="y", pad=Y_TICK_PAD, colors="black")
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LINE_WIDTH)

    handles, labels = ax.get_legend_handles_labels()

    if largest_cluster_nchain is not None:
        if cluster_time_ns is None:
            raise ValueError(
                "cluster_time_ns must be provided when largest_cluster_nchain "
                "is provided."
            )

        cluster_block_time, cluster_block_mean, cluster_block_std = block_average(
            cluster_time_ns,
            largest_cluster_nchain,
            CLUSTER_BLOCK_SIZE_POINTS,
        )

        ax2 = ax.twinx()

        # Largest cluster raw data: light blue, low alpha.
        ax2.plot(
            cluster_time_ns,
            largest_cluster_nchain,
            color=RIGHT_RAW_COLOR,
            linewidth=RAW_LINE_WIDTH,
            alpha=0.35,
            label="_nolegend_",
        )

        # Largest cluster block average: dark blue, thick line.
        ax2.plot(
            cluster_block_time,
            cluster_block_mean,
            color=RIGHT_BLOCK_COLOR,
            linewidth=BLOCK_LINE_WIDTH,
            alpha=0.95,
            label=LC_LEGEND_LABEL,
        )

        # Largest cluster shaded band: mean +/- std within each block.
        ax2.fill_between(
            cluster_block_time,
            cluster_block_mean - cluster_block_std,
            cluster_block_mean + cluster_block_std,
            color=RIGHT_BLOCK_COLOR,
            alpha=RIGHT_SHADE_ALPHA,
            linewidth=0,
            label="_nolegend_",
        )

        ax2.set_ylabel(RIGHT_Y_LABEL, fontsize=LABEL_FONT_SIZE, color="black")
        ax2.tick_params(
            axis="y",
            labelsize=TICK_FONT_SIZE,
            width=TICK_WIDTH,
            length=TICK_LENGTH,
            direction="in",
            colors="black",
            pad=Y_TICK_PAD,
        )
        for spine in ax2.spines.values():
            spine.set_linewidth(SPINE_LINE_WIDTH)
        if RIGHT_Y_AXIS_LIMITS is not None:
            ax2.set_ylim(RIGHT_Y_AXIS_LIMITS)
        else:
            ax2.set_ylim(0, N_CHAINS + 0.5)

        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2

    if SHOW_LEGEND:
        legend_kwargs = dict(
            frameon=False,
            loc=LEGEND_LOC,
            fontsize=LEGEND_FONT_SIZE,
        )
        if LEGEND_BBOX_TO_ANCHOR is not None:
            legend_kwargs["bbox_to_anchor"] = LEGEND_BBOX_TO_ANCHOR
        ax.legend(handles, labels, **legend_kwargs)

    if SHOW_PANEL_TAG:
        ax.text(
            PANEL_TAG_X,
            PANEL_TAG_Y,
            PANEL_TAG_TEXT,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=PANEL_TAG_FONT_SIZE,
            color="black",
        )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=FIG_DPI)
    plt.close(fig)


def save_block_data(time_ns, values, output_file):
    """
    Save block-averaged data for later plotting or checking.
    """
    block_time, block_mean, block_std = block_average(
        time_ns,
        values,
        BLOCK_SIZE_POINTS,
    )
    out = np.column_stack([block_time, block_mean, block_std])
    np.savetxt(
        output_file,
        out,
        header="time_ns block_mean_Rg_nm block_std_Rg_nm",
        fmt="%.8f",
    )


def save_cluster_block_data(time_ns, values, output_file):
    """
    Save largest-cluster block-averaged data for later plotting or checking.
    """
    block_time, block_mean, block_std = block_average(
        time_ns,
        values,
        CLUSTER_BLOCK_SIZE_POINTS,
    )
    out = np.column_stack([block_time, block_mean, block_std])
    np.savetxt(
        output_file,
        out,
        header="time_ns block_mean_largest_cluster_nchain block_std_largest_cluster_nchain",
        fmt="%.8f",
    )


def main():
    print("Loading trajectory...")
    print(f"TPR: {TPR_FILE}")
    print(f"XTC: {XTC_FILE}")
    print(f"Polymer selection: {POLYMER_SELECTION}")
    print(f"N_CHAINS: {N_CHAINS}")
    print(f"BEADS_PER_CHAIN: {BEADS_PER_CHAIN}")
    print(f"REPEAT_UNIT_SIZE: {REPEAT_UNIT_SIZE}")
    print(f"RING_BEAD_OFFSETS: {RING_BEAD_OFFSETS}")
    print(f"N_REPEAT_UNITS_PER_CHAIN: {N_REPEAT_UNITS_PER_CHAIN}")
    print(f"FRAME_STRIDE: {FRAME_STRIDE}")
    print(f"BLOCK_SIZE_POINTS: {BLOCK_SIZE_POINTS}")
    print(f"PLOT_MAX_TIME_NS: {PLOT_MAX_TIME_NS}")
    print(f"TRAJECTORY_TIME_TO_NS_FACTOR: {TRAJECTORY_TIME_TO_NS_FACTOR}")
    print(f"CLUSTER_CUTOFF_NM: {CLUSTER_CUTOFF_NM}")
    print(f"USE_PBC_FOR_CONTACT: {USE_PBC_FOR_CONTACT}")
    print(f"CLUSTER_FRAME_STRIDE: {CLUSTER_FRAME_STRIDE}")
    print(f"CLUSTER_BLOCK_SIZE_POINTS: {CLUSTER_BLOCK_SIZE_POINTS}")
    print(f"Output folder: {OUT_DIR}")

    os.makedirs(OUT_DIR, exist_ok=True)

    u = mda.Universe(TPR_FILE, XTC_FILE)
    polymer = u.select_atoms(POLYMER_SELECTION)

    expected_n_beads = N_CHAINS * BEADS_PER_CHAIN
    if len(polymer) != expected_n_beads:
        raise ValueError(
            f"Selected polymer atoms = {len(polymer)}, "
            f"but N_CHAINS * BEADS_PER_CHAIN = {expected_n_beads}. "
            f"Please check N_CHAINS, BEADS_PER_CHAIN, or POLYMER_SELECTION."
        )

    chains = []
    for i in range(N_CHAINS):
        start = i * BEADS_PER_CHAIN
        end = (i + 1) * BEADS_PER_CHAIN
        chains.append(polymer[start:end])

    rg_results = []

    for ts in u.trajectory[::FRAME_STRIDE]:
        time_ns = ts.time * TRAJECTORY_TIME_TO_NS_FACTOR
        if PLOT_MAX_TIME_NS is not None and time_ns > PLOT_MAX_TIME_NS:
            break

        iframe = ts.frame

        rg_all_A = calc_rg_equal_mass(polymer.positions)
        rg_all_nm = rg_all_A / 10.0

        ring_coms_all_chains = [get_ring_coms(chain) for chain in chains]
        ring_coms_all = np.vstack(ring_coms_all_chains)
        rg_ring_A = calc_rg_equal_mass(ring_coms_all)
        rg_ring_nm = rg_ring_A / 10.0

        rg_results.append([
            time_ns,
            iframe,
            rg_all_nm,
            rg_ring_nm,
        ])

        if iframe % PRINT_EVERY == 0:
            print(
                f"Frame {iframe:8d}, corrected time = {time_ns:10.3f} ns, "
                f"Rg_all = {rg_all_nm:8.4f} nm, "
                f"Rg_ring = {rg_ring_nm:8.4f} nm"
            )

    if not rg_results:
        raise RuntimeError("No frames were processed. Check trajectory and settings.")

    print("\nCalculating largest cluster nchain with independent sampling...")

    cluster_results = []

    for ts in u.trajectory[::CLUSTER_FRAME_STRIDE]:
        time_ns = ts.time * TRAJECTORY_TIME_TO_NS_FACTOR
        if PLOT_MAX_TIME_NS is not None and time_ns > PLOT_MAX_TIME_NS:
            break

        iframe = ts.frame

        ring_coms_all_chains = [get_ring_coms(chain) for chain in chains]
        box = ts.dimensions if USE_PBC_FOR_CONTACT else None
        largest_cluster_nchain = calc_largest_cluster_nchain(
            ring_coms_all_chains,
            box,
        )

        cluster_results.append([
            time_ns,
            iframe,
            largest_cluster_nchain,
        ])

        if iframe % PRINT_EVERY == 0:
            print(
                f"Cluster frame {iframe:8d}, corrected time = {time_ns:10.3f} ns, "
                f"largest cluster = {largest_cluster_nchain:2d} chains"
            )

    if not cluster_results:
        raise RuntimeError(
            "No cluster frames were processed. "
            "Check CLUSTER_FRAME_STRIDE, PLOT_MAX_TIME_NS, and trajectory."
        )

    rg_results = np.array(rg_results)
    cluster_results = np.array(cluster_results)

    output_raw = os.path.join(OUT_DIR, "10chains_Rg_raw_sampled.dat")
    np.savetxt(
        output_raw,
        rg_results,
        header=(
            "time_ns frame_index Rg_all_beads_nm Rg_ring_COMs_nm"
        ),
        fmt=["%.6f", "%d", "%.6f", "%.6f"],
    )

    output_cluster_raw = os.path.join(
        OUT_DIR,
        "largest_cluster_nchain_raw_sampled.dat",
    )
    np.savetxt(
        output_cluster_raw,
        cluster_results,
        header=(
            "time_ns frame_index largest_cluster_nchain "
            f"CLUSTER_CUTOFF_NM={CLUSTER_CUTOFF_NM} "
            f"USE_PBC_FOR_CONTACT={USE_PBC_FOR_CONTACT} "
            f"TRAJECTORY_TIME_TO_NS_FACTOR={TRAJECTORY_TIME_TO_NS_FACTOR}"
        ),
        fmt=["%.6f", "%d", "%d"],
    )

    time_ns = rg_results[:, 0]
    rg_all_nm = rg_results[:, 2]
    rg_ring_nm = rg_results[:, 3]
    cluster_time_ns = cluster_results[:, 0]
    largest_cluster_nchain = cluster_results[:, 2]

    output_block_all = os.path.join(OUT_DIR, "10chains_Rg_all_beads_blockavg.dat")
    output_block_ring = os.path.join(OUT_DIR, "10chains_Rg_ring_COMs_blockavg.dat")
    output_cluster_block = os.path.join(
        OUT_DIR,
        "largest_cluster_nchain_blockavg.dat",
    )
    save_block_data(time_ns, rg_all_nm, output_block_all)
    save_block_data(time_ns, rg_ring_nm, output_block_ring)
    save_cluster_block_data(
        cluster_time_ns,
        largest_cluster_nchain,
        output_cluster_block,
    )

    print("\nGenerating figures...")
    plot_raw_blockavg(
        time_ns,
        rg_all_nm,
        ylabel=LEFT_Y_LABEL_ALL_BEADS,
        title="10-chain system $R_g$ from all beads",
        filename=f"04_10chains_Rg_all_beads_raw_blockavg_std.{FIG_FORMAT}",
        cluster_time_ns=cluster_time_ns,
        largest_cluster_nchain=largest_cluster_nchain,
    )

    plot_raw_blockavg(
        time_ns,
        rg_ring_nm,
        ylabel=LEFT_Y_LABEL_RING_COM,
        title="10-chain system $R_g$ from ring COMs",
        filename=f"05_10chains_Rg_ring_COMs_raw_blockavg_std.{FIG_FORMAT}",
    )

    print("\nSaved data files:")
    print(f"  {output_raw}")
    print(f"  {output_cluster_raw}")
    print(f"  {output_block_all}")
    print(f"  {output_block_ring}")
    print(f"  {output_cluster_block}")

    print("\nSaved figures:")
    print(
        f"  {os.path.join(OUT_DIR, f'04_10chains_Rg_all_beads_raw_blockavg_std.{FIG_FORMAT}')}"
    )
    print(
        f"  {os.path.join(OUT_DIR, f'05_10chains_Rg_ring_COMs_raw_blockavg_std.{FIG_FORMAT}')}"
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
