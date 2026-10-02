#!/usr/bin/env python3
"""Master Execution CLI for Model Training."""

import argparse
import logging
import os
import sys
from pathlib import Path

# Setup Project Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.paths import FilePaths
from configs.BDT_config import LEGACY_14_TRAIN_VARS
from src.training.train_bdt import run_bdt_training
from src.training.train_gatr import run_gatr_training


def setup_logger(log_file_path: str, mode: str = "w") -> logging.Logger:
    """Configures simultaneous console and file logging."""
    logger = logging.getLogger("TrainingMaster")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Log file
    fh = logging.FileHandler(log_file_path, mode=mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console output
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def main():
    parser = argparse.ArgumentParser(description="Master Training Run Orchestrator.")

    parser.add_argument("-m", "--model", type=str, required=True, choices=["bdt", "gatr", "all"], help="Model architecture to train.")
    parser.add_argument("-v", "--version-tag", type=str, default="v1", help="Run dataset version tag identifier.")
    parser.add_argument("-r", "--run-tag", type=str, default=None, help="Optional model descriptor/experiment tag (e.g. 'lr1e-4', 'deep').")
    parser.add_argument("-w", "--working-point", type=float, default=0.5, help="Score threshold working point.")
    parser.add_argument("--epochs", type=int, default=20, help="[GATr] Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="[GATr] Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="[GATr] Learning rate for fine-tuning.")
    parser.add_argument("--patience", type=int, default=5, help="[GATr] Early stopping patience.")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="[GATr] Minimum delta for early stopping.")
    parser.add_argument("--resume", action="store_true", help="[GATr] Load weights from existing gatr_best_model.pt in run_dir.")
    parser.add_argument("--pretrained-path", type=str, default=None, help="[GATr] Explicit file path to pretrained .pt checkpoint weights.")
    parser.add_argument("--keep_optimizer", action="store_true", help="[GATr] Restore old optimizer state instead of starting fresh AdamW.")

    parser.add_argument(
        "--bdt-features",
        choices=["all", "legacy14"],
        default="all",
        help="[BDT] 'all': every non-target column (default); "
             "'legacy14': the 14 features of model_bdt_preselection.",
    )
    args = parser.parse_args()
    paths = FilePaths()

    # Determine unique folder name for model output
    folder_name = (
        f"model_{args.version_tag}_{args.run_tag}"
        if args.run_tag
        else f"model_{args.version_tag}"
    )

    # --- Train BDT Pipeline ---
    if args.model in ["bdt", "all"]:
        bdt_run_dir = os.path.join(paths.BDT_MODELS_DIR, folder_name)
        os.makedirs(bdt_run_dir, exist_ok=True)

        logger = setup_logger(os.path.join(bdt_run_dir, "training.log"))
        logger.info(f"=== BDT Run Directory: {bdt_run_dir} ===")

        run_bdt_training(
            run_dir=bdt_run_dir,
            version_tag=args.version_tag,
            logger=logger,
            working_point=args.working_point,
            feature_cols=(
                LEGACY_14_TRAIN_VARS if args.bdt_features == "legacy14" else None
            ),
        )

    # --- Train GATr Pipeline ---
    if args.model in ["gatr", "all"]:
        gatr_run_dir = os.path.join(paths.GATR_MODELS_DIR, folder_name)
        os.makedirs(gatr_run_dir, exist_ok=True)

        log_mode = "a" if (args.resume or args.pretrained_path) else "w"
        logger = setup_logger(os.path.join(gatr_run_dir, "training.log"), mode=log_mode)
        logger.info(f"=== GATr Run Directory: {gatr_run_dir} ===")

        run_gatr_training(
            run_dir=gatr_run_dir,
            version_tag=args.version_tag,
            logger=logger,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            working_point=args.working_point,
            patience=args.patience,
            min_delta=args.min_delta,
            resume=args.resume,
            pretrained_path=args.pretrained_path,
            reset_optimizer=not args.keep_optimizer,
        )


if __name__ == "__main__":
    main()