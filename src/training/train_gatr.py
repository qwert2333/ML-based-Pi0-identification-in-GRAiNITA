"""GATr Neural Network Training and Evaluation Module."""

import glob
import logging
import os
import time
from typing import List, Union

from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import gatr.utils.einsum


def native_torch_einsum(equation, *operands):
    return torch.einsum(equation, *operands)


gatr.utils.einsum.gatr_einsum = native_torch_einsum
gatr.utils.einsum._gatr_einsum = native_torch_einsum

from configs.paths import FilePaths
from src.evaluation import evaluate_bdt
from models.gatr_models import CalorimeterGATrClassifier, GATrWrapper, ROOTShowerDataset

DEFAULT_GATR_CONFIG = {
    "in_s_channels": 1,
    "hidden_mv_channels": 8,
    "hidden_s_channels": 32,
    "out_s_channels": 16,
    "num_blocks": 2,
}


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch=1, log_interval=50, logger=None):
    model.train()
    total_loss, total_samples = 0.0, 0
    is_cuda = device.type == "cuda"
    start_time = time.time()
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader, start=1):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        z = batch["z"].to(device, non_blocking=True)
        energy_raw = batch["energy_raw"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        batch_size = x.size(0)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=is_cuda):
            logits = model(x, y, z, energy_raw, extra_scalars=None, mask=mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if batch_idx % log_interval == 0 or batch_idx == num_batches:
            avg_loss = total_loss / max(total_samples, 1)
            throughput = total_samples / max(time.time() - start_time, 1e-6)
            msg = f"Epoch [{epoch:02d}] | Batch [{batch_idx:04d}/{num_batches:04d}] | Train Loss: {avg_loss:.4f} | Throughput: {throughput:.1f} events/s"
            logger.info(msg) if logger else print(msg)

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, device, logger=None):
    model.eval()
    total_loss, total_samples = 0.0, 0
    is_cuda = device.type == "cuda"
    start_time = time.time()
    all_logits, all_labels = [], []

    for batch in dataloader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        z = batch["z"].to(device, non_blocking=True)
        energy_raw = batch["energy_raw"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        batch_size = x.size(0)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=is_cuda):
            logits = model(x, y, z, energy_raw, extra_scalars=None, mask=mask)
            loss = criterion(logits, labels)

        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_probs = torch.sigmoid(all_logits).numpy()
    all_labels = all_labels.numpy()
    avg_loss = total_loss / max(total_samples, 1)

    try:
        val_auc = roc_auc_score(all_labels, all_probs)
        msg = f"Validation Complete ({time.time() - start_time:.1f}s) | Val Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}"
    except Exception:
        msg = f"Validation Complete ({time.time() - start_time:.1f}s) | Val Loss: {avg_loss:.4f}"

    logger.info(msg) if logger else print(msg)
    return avg_loss, all_probs, all_labels


def run_gatr_training(
    version_tag: str = "30-80GeV_var_angle",
    train_files: Union[str, List[str]] = None,
    val_files: Union[str, List[str]] = None,
    test_files: Union[str, List[str]] = None,
    run_dir: str = None,
    config_dict: dict = None,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    working_point: float = 0.5,
    prefix: str = "gatr_eval",
    logger: logging.Logger = None,
    patience: int = 5,
    min_delta: float = 1e-4,
    resume: bool = False,
    pretrained_path: str = None,
    reset_optimizer: bool = True,
) -> dict:
    """Executes GATr streaming dataset training, fine-tuning, checkpointing, and evaluation."""
    paths = FilePaths()
    paths.ensure_dirs()
    epochs_no_improve = 0
    best_val_loss = float("inf")

    file_patterns = paths.get_gatr_file_patterns(version_tag)
    train_files = train_files or file_patterns["train"]
    val_files = val_files or file_patterns["val"]
    test_files = test_files or file_patterns["test"]
    run_dir = run_dir or paths.get_run_dir("gatr", version_tag)

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if isinstance(train_files, str):
        train_files = sorted(glob.glob(train_files))
    if isinstance(val_files, str):
        val_files = sorted(glob.glob(val_files))
    if isinstance(test_files, str):
        test_files = sorted(glob.glob(test_files))

    print(f"Resolved GATr file patterns for version '{version_tag}':")
    print(f"  Train files: {len(train_files)}", train_files[:3] if len(train_files) > 3 else train_files)
    print(f"  Val files:   {len(val_files)}", val_files[:3] if len(val_files) > 3 else val_files)
    print(f"  Test files:  {len(test_files)}", test_files[:3] if len(test_files) > 3 else test_files)

    if not train_files or not val_files:
        raise FileNotFoundError("GATr training or validation ROOT chunk files not found.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if logger:
        logger.info(f"Running GATr training on device: {device}")
        logger.info(f"Train files: {len(train_files)} | Val files: {len(val_files)}")

    config = config_dict if config_dict is not None else DEFAULT_GATR_CONFIG
    model = CalorimeterGATrClassifier(config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_checkpoint_path = os.path.join(run_dir, "gatr_best_model.pt")

    # Load weights from specified checkpoint or previous best model
    checkpoint_to_load = pretrained_path if pretrained_path else (best_checkpoint_path if resume else None)
    
    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        if logger:
            logger.info(f"Loading pretrained weights from: {checkpoint_to_load}")
        checkpoint = torch.load(checkpoint_to_load, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer only if explicitly requested; default is fresh optimizer for new samples
        if not reset_optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            if logger:
                logger.info("Restored previous optimizer momentum states.")
        else:
            if logger:
                logger.info(f"Initialized clean AdamW optimizer with target Learning Rate: {lr:.2e}")

    start_epoch = 1
    end_epoch = start_epoch + epochs - 1

    for epoch in range(start_epoch, end_epoch + 1):
        start_time = time.time()
        
        # DataLoader initialization
        train_loader = DataLoader(ROOTShowerDataset(train_files, batch_size=batch_size), batch_size=None)
        val_loader = DataLoader(ROOTShowerDataset(val_files, batch_size=batch_size), batch_size=None)

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch, logger=logger)
        val_loss, _, _ = evaluate_epoch(model, val_loader, criterion, device, logger=logger)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        if logger:
            logger.info(f"Epoch [{epoch:02d}/{end_epoch:02d}] ({time.time() - start_time:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                },
                best_checkpoint_path,
            )
            if logger:
                logger.info(f"  --> Saved new best checkpoint to {best_checkpoint_path}")
        else:
            epochs_no_improve += 1
            if logger:
                logger.info(f"  --> No improvement in validation loss for {epochs_no_improve} epochs.")

            if epochs_no_improve >= patience:
                if logger:
                    logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    # Evaluation phase
    eval_metrics = {}
    if test_files:
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        wrapped_gatr = GATrWrapper(model, device=device)

        test_loader = DataLoader(ROOTShowerDataset(test_files, batch_size=64), batch_size=None)
        all_labels = [batch["label"] for batch in test_loader]
        y_test = torch.cat(all_labels, dim=0).numpy()

        test_loader_eval = DataLoader(ROOTShowerDataset(test_files, batch_size=64), batch_size=None)
        eval_metrics["linear"] = evaluate_bdt(
            model=wrapped_gatr, X_test=test_loader_eval, y_test=y_test, output_dir=plots_dir, prefix=prefix, log_roc=False, working_point=working_point
        )
        eval_metrics["log"] = evaluate_bdt(
            model=wrapped_gatr, X_test=test_loader_eval, y_test=y_test, output_dir=plots_dir, prefix=prefix, log_roc=True, working_point=working_point
        )

    return {"best_checkpoint": best_checkpoint_path, "metrics": eval_metrics}


if __name__ == "__main__":
    run_gatr_training(version_tag="30-80GeV_var_angle")