#!/usr/bin/env python3
import numpy as np
import math
from pathlib import Path
import MDAnalysis as mda
from MDAnalysis.lib.distances import minimize_vectors
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# ============================================================
# 用户需要修改的参数
# 以后直接运行：
#     python lp2.py
# 即可，不需要再在命令行后面写一长串文件名。
# 需要的输入文件、group 名称、输出前缀等都在这里改。
# ============================================================

# 输入文件
TOP_FILE = "1chian.tpr"
TRAJ_FILE = "CG_centermol.xtc"
NDX_FILE = "bonds.ndx"

# 需要计算持续长度的 group 名称（必须存在于 ndx 中）
GROUP_NAME = "lp"

# 输出文件前缀
OUTPUT_PREFIX = "lp_result"

# 是否生成图片
MAKE_PLOT = True

# 是否使用 PBC 最小镜像来修正相邻键向量
# 如果链可能跨越盒子边界，这里建议保持 True
USE_PBC = True

# 图像设置
FIG_DPI = 300
FONT_FAMILY = "DejaVu Sans"
FIG_SIZE = (6.4, 4.8)

# ---------- model ----------
def exp_decay(n, npers):
    return np.exp(-n / npers)

# ---------- ndx parser ----------
def parse_ndx(fname: str):
    groups = {}
    cur = None
    with open(fname, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(("#", ";")):
                continue
            if line.startswith("["):
                cur = line.strip("[]").strip()
                groups[cur] = []
            else:
                if cur is None:
                    continue
                # ndx is 1-based; convert to 0-based
                groups[cur] += [int(x) - 1 for x in line.split()]
    return groups

# ---------- infer chains from ndx order ----------
def build_chains_from_ndx_order(sel_ids):
    """
    Build chains by using the ORDER in ndx (assumed backbone order),
    and infer the common stride between consecutive beads.

    Works for cases like:
      - R beads only: atom indices often go 0,3,6,9,... (stride ~ 3)
      - any fixed-stride selection
      - multi-chain ndx concatenated: ... last bead of chain1, then big jump to chain2 start
    """
    sel_ids = list(sel_ids)
    if len(sel_ids) < 2:
        raise ValueError("Selected group has fewer than 2 atoms; cannot define bonds.")

    diffs = np.diff(sel_ids)
    pos_diffs = diffs[diffs > 0]
    if len(pos_diffs) == 0:
        raise ValueError(
            "Cannot infer stride from ndx order: no positive diffs found. "
            "Check ndx ordering."
        )

    # robust typical stride
    stride = int(np.median(pos_diffs))
    if stride <= 0:
        raise ValueError(f"Invalid inferred stride={stride}. Check ndx ordering.")

    # threshold for detecting a new chain (big jump)
    break_thresh = max(stride + 1, int(1.5 * stride))

    chains = []
    cur = [sel_ids[0]]
    for i in range(1, len(sel_ids)):
        d = sel_ids[i] - sel_ids[i - 1]

        # same chain if it's roughly the stride (allow +-1)
        if abs(d - stride) <= 1:
            cur.append(sel_ids[i])
        # big jump => new chain
        elif d >= break_thresh:
            chains.append(cur)
            cur = [sel_ids[i]]
        else:
            # weird diff (e.g., small jump but not stride) -> safer to split
            chains.append(cur)
            cur = [sel_ids[i]]

    chains.append(cur)

    # sanity check
    bad = [len(c) for c in chains if len(c) < 2]
    if bad:
        raise ValueError(
            "Detected a chain with <2 atoms. "
            f"Inferred stride={stride}, break_thresh={break_thresh}, "
            f"chain_lengths={[len(c) for c in chains]}. "
            "Check ndx ordering or adjust logic."
        )

    return chains, stride, break_thresh

# ---------- main ----------
def main():
    script_dir = Path(__file__).resolve().parent
    top_path = (script_dir / TOP_FILE).resolve()
    traj_path = (script_dir / TRAJ_FILE).resolve()
    ndx_path = (script_dir / NDX_FILE).resolve()
    out_prefix = str((script_dir / OUTPUT_PREFIX).resolve())

    if not top_path.exists():
        raise FileNotFoundError(f"Topology file not found: {top_path}")
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
    if not ndx_path.exists():
        raise FileNotFoundError(f"Index file not found: {ndx_path}")

    groups = parse_ndx(str(ndx_path))
    if GROUP_NAME not in groups:
        raise ValueError(f"Group '{GROUP_NAME}' not found in {ndx_path}. Available: {list(groups.keys())}")

    sel_ids = groups[GROUP_NAME]
    if len(sel_ids) < 2:
        raise ValueError("Selected group has fewer than 2 atoms; cannot define bonds.")

    # Load trajectory
    u = mda.Universe(str(top_path), str(traj_path))

    # Build chains from ndx ordering (infer stride)
    chains, stride, break_thresh = build_chains_from_ndx_order(sel_ids)
    chain_lens = [len(c) for c in chains]
    print(f"[INFO] inferred stride = {stride}, break_thresh = {break_thresh}")
    print(f"[INFO] detected {len(chains)} chain(s), lengths = {chain_lens}")

    # We can only compute up to min(chain_len-1) across chains
    max_n = min(len(c) - 1 for c in chains)
    if max_n < 1:
        raise ValueError("Chains too short to compute angle correlation.")

    sumc = np.zeros(max_n, dtype=float)
    cnt  = np.zeros(max_n, dtype=int)

    # Pre-make AtomGroup once for speed and consistency
    ag = u.atoms[sel_ids]

    for ts in u.trajectory:
        # MDAnalysis 的长度基准单位是 Angstrom。
        # 对 GROMACS 的 tpr/xtc 来说，读取后这里也是 Angstrom。
        # 本脚本只计算夹角相关函数，长度单位会在归一化时相互抵消，
        # 因此这里不用再手动换算成 nm。
        coords = ag.positions.copy()

        # map atom index -> coordinate according to ndx order
        pos = {aid: coords[i] for i, aid in enumerate(sel_ids)}

        for c in chains:
            r = np.array([pos[i] for i in c], dtype=float)
            b = r[1:] - r[:-1]

            if USE_PBC:
                b = minimize_vectors(b, box=ts.dimensions)

            norm = np.linalg.norm(b, axis=1)
            good = norm > 0
            if np.count_nonzero(good) < 2:
                continue

            b = b[good] / norm[good][:, None]

            nb = len(b)
            for n in range(1, nb):
                dots = np.sum(b[:-n] * b[n:], axis=1)
                sumc[n - 1] += dots.sum()
                cnt[n - 1]  += len(dots)

    if np.any(cnt == 0):
        # truncate to the last n where we have data
        last = np.max(np.where(cnt > 0)[0]) + 1
        sumc = sumc[:last]
        cnt  = cnt[:last]

    cos_avg = sumc / cnt
    n_vals  = np.arange(1, len(cos_avg) + 1)

    # Fit exponential only where cos_avg > 0
    valid = cos_avg > 0
    if np.count_nonzero(valid) < 3:
        raise RuntimeError(
            f"Not enough valid points for exponential fit. "
            f"Valid points={np.count_nonzero(valid)}. "
            f"cos_avg={cos_avg}"
        )

    popt, _ = curve_fit(exp_decay, n_vals[valid], cos_avg[valid], p0=[5.0])
    npers = float(popt[0])

    # Save data
    csv_path = out_prefix + ".csv"
    xvg_path = out_prefix + ".xvg"

    fit_vals = exp_decay(n_vals, npers)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("n,cos_avg,fit_exp_decay,fit_used\n")
        for n, c, fit_v, is_valid in zip(n_vals, cos_avg, fit_vals, valid):
            f.write(f"{n},{c:.9g},{fit_v:.9g},{int(is_valid)}\n")

    with open(xvg_path, "w") as f:
        f.write("# n   <cos(theta_n)>   fit_exp_decay   fit_used\n")
        for n, c, fit_v, is_valid in zip(n_vals, cos_avg, fit_vals, valid):
            f.write(f"{n:6d} {c:12.6f} {fit_v:12.6f} {int(is_valid):3d}\n")

    # Plot
    if MAKE_PLOT:
        plt.rcParams["font.family"] = FONT_FAMILY
        plt.figure(figsize=FIG_SIZE)
        plt.plot(n_vals, cos_avg, "o", label="Simulation")
        plt.plot(n_vals, fit_vals, "--", label="Fit")
        plt.axhline(1 / math.e, color="gray", ls=":")
        plt.axvline(npers, color="k", ls=":")
        plt.xlabel("Monomer index n")
        plt.ylabel(r"$\langle \cos \theta_n \rangle$")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_prefix + ".png", dpi=FIG_DPI)

    print(f"Persistence length n_p = {npers:.2f} monomers")
    print(f"[OUT] wrote {csv_path}")
    print(f"[OUT] wrote {xvg_path}")
    if MAKE_PLOT:
        print(f"[OUT] wrote {out_prefix}.png")

if __name__ == "__main__":
    main()
