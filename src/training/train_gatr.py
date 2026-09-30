"""GATr multi-task training and evaluation."""

import glob
import json
import logging
import math
import os
import random
import time
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import gatr.utils.einsum


def native_torch_einsum(equation, *operands):
    return torch.einsum(equation, *operands)


gatr.utils.einsum.gatr_einsum = native_torch_einsum
gatr.utils.einsum._gatr_einsum = native_torch_einsum

from configs.paths import FilePaths
from src.evaluation import evaluate_bdt
from src.gatr_metrics import (
    classification_metrics,
    decay_point_metrics,
    format_classification_summary,
    mass_metrics,
)
from src.models.gatr_models import (
    FIRST_LAYER_FRACTION_KEY,
    CalorimeterGATrClassifier,
    ROOTShowerDataset,
)


DEFAULT_GATR_CONFIG = {
    "in_s_channels": 2,
    "hidden_mv_channels": 8,
    "hidden_s_channels": 32,
    "out_s_channels": 16,
    "out_mv_channels": 1,
    "num_blocks": 2,
    "num_heads": 8,
    "mlp_hidden_dim": 32,
    "dropout": 0.1,
    "use_global_features": True,
    "classification_loss_weight": 1.0,
    "mass_loss_weight": 1.0,
    "decay_point_loss_weight": 1.0,
    "point_embedding_reg_weight": 1.0e-3,
    "decay_point_scale_mm": 1000.0,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_amp_dtype(name: str, device: torch.device):
    """Map 'bf16' / 'fp16' / 'none' to an autocast dtype (None disables AMP)."""
    name = (name or "none").lower()
    if device.type != "cuda" or name == "none":
        return None
    if name == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unknown amp dtype '{name}'; use bf16, fp16 or none.")


def _autocast(device, amp_dtype):
    return torch.amp.autocast(
        device_type=device.type,
        dtype=amp_dtype or torch.float32,
        enabled=amp_dtype is not None,
    )


def warmup_cosine_lambda(total_steps, warmup_steps, min_lr_ratio):
    """Linear warmup followed by cosine decay to ``min_lr_ratio``."""

    def factor(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return factor


def _masked_mean(values, mask, reference):
    if bool(mask.any()):
        return values[mask].mean()
    return reference.sum() * 0.0


def compute_multitask_loss(outputs, batch, config):
    """Compute masked classification, mass, and decay-point losses."""
    logits = outputs["classification_logits"].float()
    labels = batch["label"].to(logits.device).float()
    class_valid = batch.get("classification_valid")
    if class_valid is None:
        class_valid = labels >= 0
    else:
        class_valid = class_valid.to(logits.device).bool() & (labels >= 0)

    classification_items = nn.functional.binary_cross_entropy_with_logits(
        logits, labels.clamp(0.0, 1.0), reduction="none"
    )
    classification_loss = _masked_mean(
        classification_items, class_valid, logits
    )

    regression_valid = batch.get("regression_valid")
    if regression_valid is None:
        regression_valid = torch.zeros_like(class_valid)
    else:
        regression_valid = regression_valid.to(logits.device).bool()

    mass_target = batch["llp_mass_GeV"].to(logits.device).float()
    mass_valid = regression_valid & torch.isfinite(mass_target)
    if "mass_valid" in batch:
        mass_valid &= batch["mass_valid"].to(logits.device).bool()
    # Replace missing targets before loss evaluation. Masking only after the loss
    # leaves NaN derivatives in the graph for background clusters.
    safe_mass_target = torch.where(
        mass_valid, mass_target, torch.zeros_like(mass_target)
    )
    mass_items = nn.functional.smooth_l1_loss(
        outputs["mass_log1p"].float(),
        torch.log1p(safe_mass_target.clamp_min(0.0)),
        reduction="none",
    )
    mass_loss = _masked_mean(mass_items, mass_valid, logits)

    decay_target = torch.stack(
        [
            batch["llp_decay_x_mm"],
            batch["llp_decay_y_mm"],
            batch["llp_decay_z_mm"],
        ],
        dim=-1,
    ).to(logits.device).float()
    decay_valid = regression_valid & torch.isfinite(decay_target).all(dim=-1)
    if "decay_vertex_valid" in batch:
        decay_valid &= batch["decay_vertex_valid"].to(logits.device).bool()
    decay_scale = float(config.get("decay_point_scale_mm", 1000.0))
    safe_decay_target = torch.where(
        decay_valid.unsqueeze(-1), decay_target, torch.zeros_like(decay_target)
    )
    decay_items = nn.functional.smooth_l1_loss(
        outputs["decay_point_mm"].float() / decay_scale,
        safe_decay_target / decay_scale,
        reduction="none",
    ).mean(dim=-1)
    decay_loss = _masked_mean(decay_items, decay_valid, logits)

    point_reg_items = outputs["point_embedding_reg"].float().square()
    point_reg_loss = _masked_mean(point_reg_items, decay_valid, logits)

    weighted = {
        "classification": (
            float(config.get("classification_loss_weight", 1.0))
            * classification_loss
        ),
        "mass": float(config.get("mass_loss_weight", 1.0)) * mass_loss,
        "decay_point": (
            float(config.get("decay_point_loss_weight", 1.0))
            * decay_loss
        ),
        "point_regularization": (
            float(config.get("point_embedding_reg_weight", 1.0e-3))
            * point_reg_loss
        ),
    }
    total = sum(weighted.values())
    raw = {
        "classification": classification_loss,
        "mass": mass_loss,
        "decay_point": decay_loss,
        "point_regularization": point_reg_loss,
    }
    masks = {
        "classification": class_valid,
        "mass": mass_valid,
        "decay_point": decay_valid,
    }
    return total, raw, masks, decay_target


def _move_inputs(batch, device):
    inputs = {
        key: batch[key].to(device, non_blocking=True)
        for key in ("x", "y", "z", "energy_raw", "mask")
    }
    first_layer = batch.get(FIRST_LAYER_FRACTION_KEY)
    inputs["first_layer_fraction"] = (
        None if first_layer is None
        else first_layer.to(device, non_blocking=True)
    )
    return inputs


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scaler,
    device,
    config,
    epoch=1,
    log_interval=50,
    logger=None,
    scheduler=None,
    amp_dtype=None,
    grad_clip_norm=None,
):
    model.train()
    totals = {
        "total": 0.0,
        "classification": 0.0,
        "mass": 0.0,
        "decay_point": 0.0,
    }
    total_samples = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader, start=1):
        inputs = _move_inputs(batch, device)
        batch_size = inputs["x"].size(0)
        optimizer.zero_grad(set_to_none=True)

        with _autocast(device, amp_dtype):
            outputs = model(
                inputs["x"],
                inputs["y"],
                inputs["z"],
                inputs["energy_raw"],
                mask=inputs["mask"],
                first_layer_fraction=inputs["first_layer_fraction"],
            )
            loss, raw, _, _ = compute_multitask_loss(
                outputs, batch, config
            )

        scaler.scale(loss).backward()
        if grad_clip_norm:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        totals["total"] += float(loss.detach()) * batch_size
        for key in ("classification", "mass", "decay_point"):
            totals[key] += float(raw[key].detach()) * batch_size
        total_samples += batch_size

        if batch_idx % log_interval == 0:
            message = (
                f"Epoch [{epoch:02d}] Batch [{batch_idx:04d}] | "
                f"Loss {totals['total']/total_samples:.4f} | "
                f"Class {totals['classification']/total_samples:.4f} | "
                f"Mass {totals['mass']/total_samples:.4f} | "
                f"Decay {totals['decay_point']/total_samples:.4f} | "
                f"LR {optimizer.param_groups[0]['lr']:.2e}"
            )
            logger.info(message) if logger else print(message)

    metrics = {
        key: value / max(total_samples, 1) for key, value in totals.items()
    }
    metrics["throughput"] = total_samples / max(
        time.time() - start_time, 1.0e-6
    )
    return metrics


@torch.no_grad()
def evaluate_epoch(
    model,
    dataloader,
    device,
    config,
    logger=None,
    amp_dtype=None,
    working_point=0.5,
):
    """Return (metrics, class probabilities, class labels, cluster energies)."""
    model.eval()
    totals = {
        "total": 0.0,
        "classification": 0.0,
        "mass": 0.0,
        "decay_point": 0.0,
    }
    total_samples = 0
    class_probs, class_targets, class_energies = [], [], []
    mass_predictions, mass_targets = [], []
    decay_predictions, decay_targets = [], []
    start_time = time.time()

    for batch in dataloader:
        inputs = _move_inputs(batch, device)
        batch_size = inputs["x"].size(0)
        with _autocast(device, amp_dtype):
            outputs = model(
                inputs["x"],
                inputs["y"],
                inputs["z"],
                inputs["energy_raw"],
                mask=inputs["mask"],
                first_layer_fraction=inputs["first_layer_fraction"],
            )
            loss, raw, masks, decay_target = compute_multitask_loss(
                outputs, batch, config
            )

        totals["total"] += float(loss) * batch_size
        for key in ("classification", "mass", "decay_point"):
            totals[key] += float(raw[key]) * batch_size
        total_samples += batch_size

        class_mask = masks["classification"]
        if bool(class_mask.any()):
            class_probs.append(
                torch.sigmoid(
                    outputs["classification_logits"].float()[class_mask]
                ).cpu()
            )
            class_targets.append(
                batch["label"].to(device)[class_mask].cpu()
            )
            if "cluster_energy" in batch:
                class_energies.append(
                    batch["cluster_energy"].to(device)[class_mask].cpu()
                )
        mass_mask = masks["mass"]
        if bool(mass_mask.any()):
            mass_predictions.append(
                outputs["mass_GeV"].float()[mass_mask].cpu()
            )
            mass_targets.append(
                batch["llp_mass_GeV"].to(device)[mass_mask].cpu()
            )
        decay_mask = masks["decay_point"]
        if bool(decay_mask.any()):
            decay_predictions.append(
                outputs["decay_point_mm"].float()[decay_mask].cpu()
            )
            decay_targets.append(decay_target[decay_mask].cpu())

    metrics = {
        f"loss_{key}": value / max(total_samples, 1)
        for key, value in totals.items()
    }
    probabilities = (
        torch.cat(class_probs).numpy()
        if class_probs
        else np.asarray([], dtype=np.float32)
    )
    labels = (
        torch.cat(class_targets).numpy()
        if class_targets
        else np.asarray([], dtype=np.float32)
    )
    energies = (
        torch.cat(class_energies).numpy()
        if class_energies and len(class_energies) == len(class_probs)
        else None
    )
    classification = classification_metrics(
        labels, probabilities, energy=energies, working_point=working_point
    )
    metrics["classification"] = classification
    if "auc" in classification:
        metrics["classification_auc"] = classification["auc"]

    if mass_predictions:
        mass = mass_metrics(
            torch.cat(mass_predictions).numpy(),
            torch.cat(mass_targets).numpy(),
        )
        metrics["mass"] = mass
        metrics["mass_mae_GeV"] = mass["mae_GeV"]
        metrics["mass_rmse_GeV"] = mass["rmse_GeV"]
        metrics["n_mass"] = mass["n"]

    if decay_predictions:
        decay = decay_point_metrics(
            torch.cat(decay_predictions).numpy(),
            torch.cat(decay_targets).numpy(),
        )
        metrics["decay_point"] = decay
        metrics["decay_point_mae_mm"] = decay["mae_mm"]
        metrics["decay_point_rmse_mm"] = decay["rmse_mm"]
        metrics["n_decay_point"] = decay["n"]

    message = (
        f"Validation ({time.time()-start_time:.1f}s) | "
        f"Loss {metrics['loss_total']:.4f} | "
        f"Class {metrics['loss_classification']:.4f} | "
        f"Mass {metrics['loss_mass']:.4f} | "
        f"Decay {metrics['loss_decay_point']:.4f} | "
        f"{format_classification_summary(classification)}"
    )
    if "mass_mae_GeV" in metrics:
        message += f" | Mass MAE {metrics['mass_mae_GeV']:.4f} GeV"
    if "decay_point_mae_mm" in metrics:
        message += (
            f" | Decay error {metrics['decay_point_mae_mm']:.1f} mm"
        )
    logger.info(message) if logger else print(message)
    return metrics, probabilities, labels, energies


class GATr:
    """Adapter feeding already-evaluated GATr scores to ``evaluate_bdt``.

    The class name is used by ``evaluate_bdt`` for plot labels.
    """

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, _unused):
        return np.column_stack((1.0 - self.probabilities, self.probabilities))


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _log_baseline_comparison(paths, version_tag, test_metrics, logger):
    """Log GATr test metrics next to the stored simple-feature baselines."""
    baseline_path = os.path.join(
        paths.EVAL_METRICS_DIR, f"gatr_baselines_{version_tag}.json"
    )
    if not logger:
        return
    if not os.path.exists(baseline_path):
        logger.info(
            f"No baseline file {baseline_path}; run "
            "scripts/run_gatr_baselines.py to compare against simple features."
        )
        return
    with open(baseline_path) as handle:
        baselines = json.load(handle)
    hgb = baselines.get("classification", {}).get("hgb_all_features", {})
    gatr = test_metrics.get("classification", {})
    logger.info(
        "Test comparison | GATr: "
        f"{format_classification_summary(gatr)} || "
        f"HGB simple features: {format_classification_summary(hgb)}"
    )
    for key, label, gatr_key in (
        ("mass", "Mass MAE [GeV]", "mass_mae_GeV"),
        ("decay_point", "Decay MAE [mm]", "decay_point_mae_mm"),
    ):
        reference = baselines.get(key, {})
        if gatr_key in test_metrics and reference:
            summary = ", ".join(
                f"{name} {values.get('mae_GeV', values.get('mae_mm')):.4g}"
                for name, values in reference.items()
            )
            logger.info(
                f"Test comparison | {label}: GATr "
                f"{test_metrics[gatr_key]:.4g} || baselines: {summary}"
            )


def run_gatr_training(
    version_tag: str = "30-80GeV_var_angle",
    train_files: Union[str, List[str]] = None,
    val_files: Union[str, List[str]] = None,
    test_files: Union[str, List[str]] = None,
    run_dir: str = None,
    config_dict: dict = None,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 5e-4,
    working_point: float = 0.5,
    prefix: str = "gatr_eval",
    logger: logging.Logger = None,
    patience: int = 15,
    min_delta: float = 1e-4,
    resume: bool = False,
    pretrained_path: str = None,
    reset_optimizer: bool = True,
    classification_loss_weight: float = 1.0,
    mass_loss_weight: float = 1.0,
    decay_point_loss_weight: float = 1.0,
    point_embedding_reg_weight: float = 1.0e-3,
    decay_point_scale_mm: float = 1000.0,
    weight_decay: float = 1.0e-4,
    warmup_epochs: float = 2.0,
    min_lr_ratio: float = 0.02,
    grad_clip_norm: float = 1.0,
    amp: str = "bf16",
    select_metric: str = "auc",
    seed: int = 42,
    use_global_features: bool = True,
):
    """Train and evaluate the joint cluster-level GATr model.

    The LR follows a per-step linear warmup and cosine decay. The checkpoint
    used for testing (``gatr_best_model.pt``) is chosen by ``select_metric``
    ('auc': highest validation classification AUC, 'loss': lowest validation
    total loss); the best checkpoint of the other criterion and the last epoch
    are saved alongside it.
    """
    if select_metric not in ("auc", "loss"):
        raise ValueError("select_metric must be 'auc' or 'loss'.")
    set_seed(seed)
    paths = FilePaths()
    paths.ensure_dirs()

    patterns = paths.get_gatr_file_patterns(version_tag)
    train_files = train_files or patterns["train"]
    val_files = val_files or patterns["val"]
    test_files = test_files or patterns["test"]
    run_dir = run_dir or paths.get_run_dir("gatr", version_tag)

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if isinstance(train_files, str):
        train_files = sorted(glob.glob(train_files))
    if isinstance(val_files, str):
        val_files = sorted(glob.glob(val_files))
    if isinstance(test_files, str):
        test_files = sorted(glob.glob(test_files))

    if not train_files or not val_files:
        raise FileNotFoundError(
            "GATr training or validation ROOT chunk files not found."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(amp, device)
    config = dict(DEFAULT_GATR_CONFIG)
    if config_dict is not None:
        config.update(config_dict)
    config.update(
        {
            "classification_loss_weight": classification_loss_weight,
            "mass_loss_weight": mass_loss_weight,
            "decay_point_loss_weight": decay_point_loss_weight,
            "point_embedding_reg_weight": point_embedding_reg_weight,
            "decay_point_scale_mm": decay_point_scale_mm,
            "use_global_features": use_global_features,
        }
    )

    train_dataset = ROOTShowerDataset(
        train_files, batch_size=batch_size, shuffle=True, seed=seed,
        read_step_size=5000, drop_last=True,
    )
    val_dataset = ROOTShowerDataset(val_files, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_size=None)
    val_loader = DataLoader(val_dataset, batch_size=None)
    steps_per_epoch = max(len(train_dataset), 1)
    config["max_hits"] = train_dataset.max_hits

    model = CalorimeterGATrClassifier(config).to(device)
    n_parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = LambdaLR(
        optimizer,
        warmup_cosine_lambda(
            total_steps=epochs * steps_per_epoch,
            warmup_steps=int(round(warmup_epochs * steps_per_epoch)),
            min_lr_ratio=min_lr_ratio,
        ),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(amp_dtype == torch.float16)
    )
    checkpoint_paths = {
        select_metric: os.path.join(run_dir, "gatr_best_model.pt"),
        ("loss" if select_metric == "auc" else "auc"): os.path.join(
            run_dir,
            "gatr_best_loss_model.pt"
            if select_metric == "auc"
            else "gatr_best_auc_model.pt",
        ),
    }
    last_checkpoint_path = os.path.join(run_dir, "gatr_last_model.pt")
    best_checkpoint_path = checkpoint_paths[select_metric]

    if logger:
        logger.info(
            f"Device {device} | AMP {amp_dtype} | parameters {n_parameters:,} | "
            f"max_hits {train_dataset.max_hits} | train clusters "
            f"{train_dataset.total_events} ({steps_per_epoch} steps/epoch) | "
            f"lr {lr:.1e}, warmup {warmup_epochs} ep, cosine to "
            f"{min_lr_ratio:g}x | weight decay {weight_decay:g} | "
            f"grad clip {grad_clip_norm} | select by {select_metric} | "
            f"seed {seed}"
        )
        logger.info(f"Model config: {json.dumps(config, sort_keys=True)}")

    checkpoint_to_load = (
        pretrained_path
        if pretrained_path
        else (best_checkpoint_path if resume else None)
    )
    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        checkpoint = torch.load(checkpoint_to_load, map_location=device)
        incompatible = model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        if logger:
            logger.info(f"Loaded checkpoint: {checkpoint_to_load}")
            if incompatible.missing_keys or incompatible.unexpected_keys:
                logger.info(
                    "Checkpoint architecture differences: "
                    f"missing={incompatible.missing_keys}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
        if not reset_optimizer and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                for group in optimizer.param_groups:
                    group["lr"] = lr
                    group["initial_lr"] = lr
            except ValueError:
                if logger:
                    logger.info(
                        "Optimizer state was incompatible; using a new optimizer."
                    )

    best = {"auc": -float("inf"), "loss": float("inf")}
    epochs_no_improve = 0
    history = []
    history_path = os.path.join(run_dir, "epoch_metrics.json")

    for epoch in range(1, epochs + 1):
        train_dataset.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            config,
            epoch=epoch,
            logger=logger,
            scheduler=scheduler,
            amp_dtype=amp_dtype,
            grad_clip_norm=grad_clip_norm,
        )
        val_metrics, _, _, _ = evaluate_epoch(
            model, val_loader, device, config, logger=logger,
            amp_dtype=amp_dtype, working_point=working_point,
        )
        val_loss = val_metrics["loss_total"]
        val_auc = val_metrics.get("classification_auc", float("nan"))

        if logger:
            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] | "
                f"Train {train_metrics['total']:.4f} | "
                f"Val {val_loss:.4f} | "
                f"Val AUC {val_auc:.4f} | "
                f"LR {optimizer.param_groups[0]['lr']:.2e}"
            )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_metrics": val_metrics,
            "config": config,
        }
        improved = {
            "auc": np.isfinite(val_auc) and val_auc > best["auc"] + min_delta,
            "loss": val_loss < best["loss"] - min_delta,
        }
        for criterion, value in (("auc", val_auc), ("loss", val_loss)):
            if improved[criterion]:
                best[criterion] = value
                torch.save(state, checkpoint_paths[criterion])
                if logger:
                    logger.info(
                        f"New best {criterion} ({value:.4f}) -> "
                        f"{os.path.basename(checkpoint_paths[criterion])}"
                    )
        torch.save(state, last_checkpoint_path)

        history.append({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        })
        _write_json(history_path, history)

        if improved[select_metric]:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if logger:
                    logger.info(
                        f"Early stopping after {patience} epochs without "
                        f"validation {select_metric} improvement."
                    )
                break

    evaluation = {}
    if test_files:
        test_loader = DataLoader(
            ROOTShowerDataset(test_files, batch_size=64),
            batch_size=None,
        )
        test_summary = {}
        for criterion, path in checkpoint_paths.items():
            if not os.path.exists(path):
                continue
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            if logger:
                logger.info(
                    f"Testing best-{criterion} checkpoint "
                    f"(epoch {checkpoint['epoch']})"
                )
            test_metrics, probabilities, labels, _ = evaluate_epoch(
                model, test_loader, device, config, logger=logger,
                amp_dtype=amp_dtype, working_point=working_point,
            )
            test_metrics["checkpoint_epoch"] = checkpoint["epoch"]
            test_summary[f"best_{criterion}"] = test_metrics
            if criterion != select_metric or not len(labels):
                continue

            evaluation["multitask"] = test_metrics
            _log_baseline_comparison(paths, version_tag, test_metrics, logger)
            scores = GATr(probabilities)
            for log_roc, name in ((False, "linear"), (True, "log")):
                evaluation[name] = evaluate_bdt(
                    model=scores,
                    X_test=None,
                    y_test=labels.astype(int),
                    output_dir=plots_dir,
                    prefix=prefix,
                    log_roc=log_roc,
                    working_point=working_point,
                )
        _write_json(os.path.join(run_dir, "test_metrics.json"), test_summary)

    return {
        "best_checkpoint": best_checkpoint_path,
        "metrics": evaluation,
    }


if __name__ == "__main__":
    run_gatr_training(version_tag="30-80GeV_var_angle")
