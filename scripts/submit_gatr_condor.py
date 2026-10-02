#!/usr/bin/env python3
"""Submit Pi0ID 1:3-readout GATr training through CERN EosSubmit."""

import argparse
from datetime import datetime
from pathlib import Path
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOB_SCRIPT = PROJECT_ROOT / "scripts" / "run_gatr_condor.sh"
PYTHON = PROJECT_ROOT.parent / "ML-based-Pi0-identification-in-GRAiNITA" / "gatr" / "bin" / "python"
FLAVOURS = (
    "espresso", "microcentury", "longlunch", "workday",
    "tomorrow", "testmatch", "nextweek",
)


def safe_tag(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise argparse.ArgumentTypeError(
            "Use only letters, digits, dots, underscores and hyphens"
        )
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-tag", type=safe_tag, default="0.5-80GeV_1to3")
    parser.add_argument("--run-tag", type=safe_tag, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--memory-gb", type=int, default=16)
    parser.add_argument("--disk-gb", type=int, default=8)
    parser.add_argument("--job-flavour", choices=FLAVOURS, default="nextweek")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if min(
        args.epochs, args.batch_size, args.patience,
        args.cpus, args.memory_gb, args.disk_gb,
    ) <= 0 or args.lr <= 0:
        parser.error("Training and resource parameters must be positive")
    if not PYTHON.is_file() or not JOB_SCRIPT.is_file():
        parser.error("Missing gatr Python environment or Condor worker script")

    data_root = PROJECT_ROOT / "data" / "processed_1to3" / "GATr"
    dataset = data_root / f"dataset_{args.version_tag}" / "split"
    for split in ("Training", "Validation", "Testing"):
        if not any((dataset / split).glob("*.root")):
            parser.error(f"No preprocessed ROOT chunks found in {dataset / split}")

    cuda_check = subprocess.run(
        [
            str(PYTHON), "-c",
            "import torch; raise SystemExit(0 if torch.version.cuda else 'CPU-only PyTorch')",
        ],
        capture_output=True,
        text=True,
    )
    if cuda_check.returncode:
        parser.error(f"CUDA environment check failed: {cuda_check.stderr.strip()}")

    run_tag = args.run_tag or f"gpu_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = (
        PROJECT_ROOT / "trained_models" / "GATr"
        / f"model_{args.version_tag}_{run_tag}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    submit_file = run_dir / "train.sub"
    arguments = (
        f"--version-tag {args.version_tag} --run-tag {run_tag} "
        f"--epochs {args.epochs} --batch-size {args.batch_size} "
        f"--lr {args.lr} --patience {args.patience} "
        f"--min-delta {args.min_delta}"
    )
    submit_file.write_text(
        "\n".join((
            "universe = vanilla",
            f"executable = {JOB_SCRIPT}",
            f"arguments = {arguments}",
            f"initialdir = {PROJECT_ROOT}",
            f'environment = "PI0ID_PROJECT_DIR={PROJECT_ROOT} PI0ID_GATR_DATA_DIR={data_root}"',
            f"output = {run_dir}/condor.$(ClusterId).$(ProcId).out",
            f"error = {run_dir}/condor.$(ClusterId).$(ProcId).err",
            f"log = {run_dir}/condor.$(ClusterId).log",
            "should_transfer_files = YES",
            f"request_cpus = {args.cpus}",
            f"request_memory = {args.memory_gb}GB",
            f"request_disk = {args.disk_gb}GB",
            "request_gpus = 1",
            f'+JobFlavour = "{args.job_flavour}"',
            "queue 1",
            "",
        ))
    )
    print(f"Run directory: {run_dir}")
    print(f"Submit file: {submit_file}")
    if args.dry_run:
        print(submit_file.read_text(), end="")
        return 0

    result = subprocess.run(
        [
            "bash", "-lc",
            'module load lxbatch/eossubmit && exec condor_submit "$1"',
            "bash", str(submit_file),
        ],
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
