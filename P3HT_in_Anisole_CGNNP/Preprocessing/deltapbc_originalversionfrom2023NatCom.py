import os
import numpy as np
from torchmd_cg.utils.make_deltaforces import make_deltaforces

# -------------------------
# Input files in this folder
# -------------------------
coords = "coords_combined_100_50000.npy"
forces = "forces_combined_100_50000.npy"
boxes = "boxes.npy"
out = "P3HT_deltaforces.npy"
priors = "p3ht.yaml"
psf = "p3ht.psf"

# -------------------------
# Box mode (choose one)#######################################
# -------------------------
# If True, load per-frame boxes from boxes.npy.
# If False, use FIXED_BOX for all frames (NVT fixed orthorhombic box).
USE_BOX_FILE = False
FIXED_BOX = np.array([79.1695, 79.2076, 79.2076], dtype=float)

#############################################################
def check_inputs(coords_file, forces_file, box_source):
    coords_arr = np.load(coords_file, mmap_mode="r")
    forces_arr = np.load(forces_file, mmap_mode="r")

    print("coords shape:", coords_arr.shape)
    print("forces shape:", forces_arr.shape)

    if coords_arr.ndim != 3 or coords_arr.shape[2] != 3:
        raise ValueError(
            f"{coords_file} should have shape (nframes, natoms, 3), got {coords_arr.shape}"
        )

    if forces_arr.ndim != 3 or forces_arr.shape[2] != 3:
        raise ValueError(
            f"{forces_file} should have shape (nframes, natoms, 3), got {forces_arr.shape}"
        )

    nframes = coords_arr.shape[0]
    natoms = coords_arr.shape[1]

    if forces_arr.shape[0] != nframes:
        raise ValueError(
            f"Frame count mismatch: coords has {nframes}, forces has {forces_arr.shape[0]}"
        )

    if forces_arr.shape[1] != natoms:
        raise ValueError(
            f"Atom count mismatch: coords has {natoms}, forces has {forces_arr.shape[1]}"
        )

    if isinstance(box_source, str):
        boxes_arr = np.load(box_source, mmap_mode="r")
        print("boxes shape :", boxes_arr.shape)

        if boxes_arr.ndim != 2 or boxes_arr.shape[1] != 3:
            raise ValueError(
                f"{box_source} should have shape (nframes, 3), got {boxes_arr.shape}"
            )
        if boxes_arr.shape[0] != nframes:
            raise ValueError(
                f"Frame count mismatch: coords has {nframes}, boxes has {boxes_arr.shape[0]}"
            )
        if np.any(boxes_arr <= 0):
            raise ValueError(f"{box_source} contains non-positive box lengths")

        mean_box = boxes_arr.mean(axis=0)
        min_box = boxes_arr.min(axis=0)
        max_box = boxes_arr.max(axis=0)
        print("Mean box lengths (angstrom):", mean_box.tolist())
        print("Min  box lengths (angstrom):", min_box.tolist())
        print("Max  box lengths (angstrom):", max_box.tolist())
        print("Box mode: per-frame boxes from file")
    else:
        box_vec = np.asarray(box_source, dtype=float)
        if box_vec.shape != (3,):
            raise ValueError(f"FIXED_BOX should have shape (3,), got {box_vec.shape}")
        if np.any(box_vec <= 0):
            raise ValueError("FIXED_BOX contains non-positive lengths")
        print("Fixed box lengths (angstrom):", box_vec.tolist())
        print("Box mode: fixed box for all frames")

    print("Assuming coords, forces, and box lengths are in consistent units (angstrom-based).")


if USE_BOX_FILE:
    box_input = boxes
    if not os.path.exists(boxes):
        raise FileNotFoundError(f"Cannot find boxes file: {boxes}")
else:
    box_input = FIXED_BOX

check_inputs(coords, forces, box_input)

make_deltaforces(
    coords,
    forces,
    out,
    priors,
    psf,
    boxes_npz=box_input,
    exclusions=("bonds", "angles"),
    device="cpu",
    forceterms=["bonds", "angles", "repulsioncg"],
)
#  boxes_npz   = box_input  or None
print("Wrote", out)

