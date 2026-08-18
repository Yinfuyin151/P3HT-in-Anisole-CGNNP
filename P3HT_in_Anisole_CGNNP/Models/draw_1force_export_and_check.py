# -*- coding: utf-8 -*-
"""
一步完成 TorchMD-Net 力匹配结果导出与检查。

运行方式：
    python force_export_and_check.py

你只需要修改下面“用户需要修改的参数”这一段即可。

输出文件：
    1. CSV：每一行是一个力分量点，可直接后处理
    2. NPZ：一维原始力数组（predicted_force, target_force）
    3. PNG：散点图

散点图：
    横坐标 x = predicted_force
    纵坐标 y = target_force
"""

import os
import csv
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from torchmdnet.data import DataModule
from torchmdnet.module import LNNP
from torch_geometric.loader import DataLoader


# =========================
# 用户需要修改的参数
# =========================

# 训练得到的最佳模型 checkpoint
CHECKPOINT_PATH = "test3_cutoff3_11/epoch=883-val_loss=116.7152-test_loss=8.4444.ckpt"

# 训练时使用的 yaml 文件
CONF_PATH = "test3_cutoff3_11/input.yaml"

# 训练输出目录，里面应该有 splits.npz
LOG_DIR = "test3_cutoff3_11"

# 导出哪个数据集划分：train / val / test
SPLIT = "train"

# 输出文件前缀
OUTPUT_PREFIX = f"test3_force_scatter/{SPLIT}"

# 原始 AA mapped force 文件。
# 这个文件需要和生成 delta force 时使用的原始 force 文件保持一致。
# 有了它，就可以由 prior = aa_mapped - true_delta 反推出 prior force。
AAMAPPED_FORCE_PATH = "coords_combined.npy"

# 推理 batch size；None 表示沿用训练配置
INFERENCE_BATCH_SIZE = None

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 是否打印误差统计
PRINT_METRICS = True

# 散点图最多抽样多少个点来画
MAX_POINTS_TO_PLOT = 1000000

# 点样式
POINT_SIZE = 4
POINT_ALPHA = 0.25
FIG_DPI = 300


# =========================
# 内部函数
# =========================

def ensure_output_dir(output_prefix):
    output_dir = os.path.dirname(os.path.abspath(output_prefix))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def load_hparams(checkpoint_path, conf_path=None, log_dir=None, inference_batch_size=None):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = dict(ckpt["hyper_parameters"])

    if conf_path is not None and os.path.exists(conf_path):
        with open(conf_path, "r", encoding="utf-8") as f:
            conf = yaml.safe_load(f) or {}
        hparams.update(conf)

    if log_dir is not None:
        hparams["log_dir"] = log_dir
    else:
        hparams["log_dir"] = os.path.dirname(os.path.abspath(checkpoint_path))

    if inference_batch_size is not None:
        hparams["inference_batch_size"] = inference_batch_size
    elif hparams.get("inference_batch_size", None) is None:
        hparams["inference_batch_size"] = hparams["batch_size"]

    return hparams


def get_dataloader(data_module, split):
    batch_size = data_module.hparams["inference_batch_size"]
    if split == "train":
        dataset = data_module.train_dataset
        indices = np.asarray(data_module.idx_train)
    if split == "val":
        dataset = data_module.val_dataset
        indices = np.asarray(data_module.idx_val)
    elif split == "test":
        dataset = data_module.test_dataset
        indices = np.asarray(data_module.idx_test)
    elif split != "train":
        raise ValueError(f"Unknown split: {split}")

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=data_module.hparams["num_workers"],
        persistent_workers=False,
        pin_memory=True,
        shuffle=False,
    )
    return loader, indices


def predict_batch_forces(model, batch):
    batch = model.data_transform(batch)

    extra_args = batch.to_dict()
    for key in ("y", "neg_dy", "z", "pos", "batch", "box", "q", "s"):
        if key in extra_args:
            del extra_args[key]

    _, neg_dy = model(
        batch.z,
        batch.pos,
        batch=batch.batch,
        box=batch.box if "box" in batch else None,
        q=batch.q if getattr(model.hparams, "charge", False) else None,
        s=batch.s if getattr(model.hparams, "spin", False) else None,
        extra_args=extra_args,
    )
    return neg_dy


def collect_scatter_rows_and_arrays(dataloader, model, device):
    rows = []
    sample_counter = 0
    pred_all = []
    target_all = []
    component_all = []

    for batch in dataloader:
        batch = batch.to(device)

        with torch.set_grad_enabled(model.hparams.derivative):
            pred = predict_batch_forces(model, batch)

        target = batch.neg_dy

        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        ptr = batch.ptr.detach().cpu().numpy()

        pred_all.append(pred_np.reshape(-1))
        target_all.append(target_np.reshape(-1))
        component_all.append(np.tile(np.array([0, 1, 2], dtype=np.int64), pred_np.shape[0]))

        for local_sample_idx in range(len(ptr) - 1):
            start = int(ptr[local_sample_idx])
            end = int(ptr[local_sample_idx + 1])
            natoms = end - start

            for atom_idx in range(natoms):
                for comp in range(3):
                    rows.append([
                        sample_counter,
                        atom_idx,
                        comp,
                        float(pred_np[start + atom_idx, comp]),
                        float(target_np[start + atom_idx, comp]),
                    ])

            sample_counter += 1

    pred_flat = np.concatenate(pred_all)
    target_flat = np.concatenate(target_all)
    component_flat = np.concatenate(component_all)
    return rows, pred_flat, target_flat, component_flat


def load_aamapped_force_flat(force_path, split_indices):
    all_forces = np.load(force_path, mmap_mode="r")
    split_forces = np.asarray(all_forces[split_indices], dtype=np.float32)
    return split_forces.reshape(-1)


def compute_metrics(pred, target):
    diff = pred - target
    mae = np.mean(np.abs(diff))
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    pred_var = np.var(pred)
    if pred_var > 0:
        slope = np.cov(pred, target, ddof=0)[0, 1] / pred_var
        intercept = np.mean(target) - slope * np.mean(pred)
    else:
        slope = np.nan
        intercept = np.nan

    return {
        "n": len(pred),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "slope": slope,
        "intercept": intercept,
    }


def print_metrics(metrics):
    print("========== Metrics ==========")
    print(f"Number of force components: {metrics['n']}")
    print(f"MAE  = {metrics['mae']:.8f}")
    print(f"MSE  = {metrics['mse']:.8f}")
    print(f"RMSE = {metrics['rmse']:.8f}")
    print(f"R^2  = {metrics['r2']:.8f}")
    print(f"Slope = {metrics['slope']:.8f}")
    print(f"Intercept = {metrics['intercept']:.8f}")
    print("=============================")


def save_csv(rows, output_prefix):
    csv_path = output_prefix + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "atom_id", "component", "predicted_force", "target_force"])
        writer.writerows(rows)
    return csv_path


def save_npz(pred, target, output_prefix):
    npz_path = output_prefix + ".npz"
    np.savez(npz_path, predicted_force=pred, target_force=target)
    return npz_path


def save_combined_npz(output_prefix, pred_delta, true_delta, prior_force, aa_force, total_pred):
    npz_path = output_prefix + "_all_arrays.npz"
    np.savez(
        npz_path,
        predicted_delta_force=pred_delta,
        true_delta_force=true_delta,
        prior_force=prior_force,
        aa_mapped_force=aa_force,
        total_predicted_force=total_pred,
    )
    return npz_path


def sample_for_plot(pred, target, component, max_points):
    n = len(pred)
    if n <= max_points:
        return pred, target, component
    rng = np.random.default_rng(2026)
    idx = rng.choice(n, size=max_points, replace=False)
    return pred[idx], target[idx], component[idx]


def plot_scatter(pred, target, component, metrics, output_prefix, split):
    plot_pred, plot_target, plot_component = sample_for_plot(
        pred, target, component, MAX_POINTS_TO_PLOT
    )

    vmin = min(plot_pred.min(), plot_target.min())
    vmax = max(plot_pred.max(), plot_target.max())

    plt.figure(figsize=(7, 7))
    component_styles = {
        0: ("Fx", "tab:blue"),
        1: ("Fy", "tab:orange"),
        2: ("Fz", "tab:green"),
    }
    for comp_id, (label, color) in component_styles.items():
        mask = plot_component == comp_id
        if np.any(mask):
            plt.scatter(
                plot_pred[mask],
                plot_target[mask],
                s=POINT_SIZE,
                alpha=POINT_ALPHA,
                color=color,
                label=label,
            )
    plt.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=1.2)

    plt.xlabel("Predicted CG force")
    plt.ylabel("Target CG force")
    plt.title(f"Force Scatter ({split})")
    plt.legend(loc="best", framealpha=0.9)

    text = (
        f"N(all) = {metrics['n']}\n"
        f"N(plot) = {len(plot_pred)}\n"
        f"MAE = {metrics['mae']:.4f}\n"
        f"RMSE = {metrics['rmse']:.4f}\n"
        f"R² = {metrics['r2']:.4f}\n"
        f"Slope = {metrics['slope']:.4f}"
    )
    plt.text(
        0.05,
        0.95,
        text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.xlim(vmin, vmax)
    plt.ylim(vmin, vmax)
    plt.tight_layout()

    png_path = output_prefix + ".png"
    plt.savefig(png_path, dpi=FIG_DPI)
    plt.close()
    return png_path


def plot_named_scatter(pred, target, component, name, xlabel, ylabel, output_prefix):
    metrics = compute_metrics(pred, target)
    plot_pred, plot_target, plot_component = sample_for_plot(
        pred, target, component, MAX_POINTS_TO_PLOT
    )

    vmin = min(plot_pred.min(), plot_target.min())
    vmax = max(plot_pred.max(), plot_target.max())

    plt.figure(figsize=(7, 7))
    component_styles = {
        0: ("Fx", "tab:blue"),
        1: ("Fy", "tab:orange"),
        2: ("Fz", "tab:green"),
    }
    for comp_id, (label, color) in component_styles.items():
        mask = plot_component == comp_id
        if np.any(mask):
            plt.scatter(
                plot_pred[mask],
                plot_target[mask],
                s=POINT_SIZE,
                alpha=POINT_ALPHA,
                color=color,
                label=label,
            )
    plt.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=1.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name)
    plt.legend(loc="best", framealpha=0.9)

    text = (
        f"N(all) = {metrics['n']}\n"
        f"N(plot) = {len(plot_pred)}\n"
        f"MAE = {metrics['mae']:.4f}\n"
        f"RMSE = {metrics['rmse']:.4f}\n"
        f"R² = {metrics['r2']:.4f}\n"
        f"Slope = {metrics['slope']:.4f}"
    )
    plt.text(
        0.05,
        0.95,
        text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.xlim(vmin, vmax)
    plt.ylim(vmin, vmax)
    plt.tight_layout()

    safe_name = name.lower().replace(" ", "_").replace("+", "plus").replace("/", "_")
    png_path = f"{output_prefix}_{safe_name}.png"
    plt.savefig(png_path, dpi=FIG_DPI)
    plt.close()
    return png_path, metrics


def main():
    ensure_output_dir(OUTPUT_PREFIX)

    print("Loading hyperparameters...")
    hparams = load_hparams(
        checkpoint_path=CHECKPOINT_PATH,
        conf_path=CONF_PATH,
        log_dir=LOG_DIR,
        inference_batch_size=INFERENCE_BATCH_SIZE,
    )

    print("Building dataset...")
    data = DataModule(hparams)
    data.prepare_data()
    data.setup("fit")
    dataloader, split_indices = get_dataloader(data, SPLIT)

    print("Loading model...")
    model = LNNP.load_from_checkpoint(CHECKPOINT_PATH, map_location=DEVICE)
    model.to(DEVICE)
    model.eval()

    print(f"Running inference on split = {SPLIT} ...")
    rows, pred_flat, target_flat, component_flat = collect_scatter_rows_and_arrays(
        dataloader, model, DEVICE
    )

    print("Loading AA mapped forces...")
    aa_flat = load_aamapped_force_flat(AAMAPPED_FORCE_PATH, split_indices)

    if aa_flat.shape != target_flat.shape:
        raise ValueError(
            f"Shape mismatch: aa_flat {aa_flat.shape} vs target_flat {target_flat.shape}. "
            "Please confirm AAMAPPED_FORCE_PATH matches the same frames/atoms as the training data."
        )

    prior_flat = aa_flat - target_flat
    total_pred_flat = prior_flat + pred_flat

    metrics = compute_metrics(pred_flat, target_flat)

    print("Saving CSV...")
    csv_path = save_csv(rows, OUTPUT_PREFIX)

    print("Saving NPZ...")
    npz_path = save_npz(pred_flat, target_flat, OUTPUT_PREFIX)
    all_npz_path = save_combined_npz(
        OUTPUT_PREFIX,
        pred_delta=pred_flat,
        true_delta=target_flat,
        prior_force=prior_flat,
        aa_force=aa_flat,
        total_pred=total_pred_flat,
    )

    print("Plotting scatter figures...")
    delta_png, delta_metrics = plot_named_scatter(
        pred_flat,
        target_flat,
        component_flat,
        name=f"Predicted delta vs True delta ({SPLIT})",
        xlabel="Predicted delta force",
        ylabel="True delta force",
        output_prefix=OUTPUT_PREFIX,
    )
    prior_png, prior_metrics = plot_named_scatter(
        prior_flat,
        aa_flat,
        component_flat,
        name=f"Prior force vs AA mapped force ({SPLIT})",
        xlabel="Prior force",
        ylabel="AA mapped force",
        output_prefix=OUTPUT_PREFIX,
    )
    total_png, total_metrics = plot_named_scatter(
        total_pred_flat,
        aa_flat,
        component_flat,
        name=f"Prior + NNP vs AA mapped force ({SPLIT})",
        xlabel="Prior + predicted delta force",
        ylabel="AA mapped force",
        output_prefix=OUTPUT_PREFIX,
    )

    print("Done.")
    print(f"CSV saved to: {csv_path}")
    print(f"NPZ saved to: {npz_path}")
    print(f"Combined NPZ saved to: {all_npz_path}")
    print(f"Delta PNG saved to: {delta_png}")
    print(f"Prior PNG saved to: {prior_png}")
    print(f"Total PNG saved to: {total_png}")

    if PRINT_METRICS:
        print("Delta-force metrics:")
        print_metrics(delta_metrics)
        print("Prior-force metrics:")
        print_metrics(prior_metrics)
        print("Total-force metrics:")
        print_metrics(total_metrics)


if __name__ == "__main__":
    main()
