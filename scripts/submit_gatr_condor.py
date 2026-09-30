#!/usr/bin/env python3
"""Submit GATr training on the preprocessed Ayy/gamma dataset to HTCondor.

Default ('eossubmit'): the submit file, job script and Condor logs live in the
EOS run directory and are submitted through the EosSubmit pool
(``module load lxbatch/eossubmit``). '--schedd standard' instead puts the
submit file, a small worker wrapper and the Condor logs in a real AFS
directory (``--afs-submit-root``) and uses the standard CERN schedd, with the
job reaching the EOS project through an AFS->EOS link such as
``~/eos -> /eos/user/<initial>/<user>``.
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOB_SCRIPT = PROJECT_ROOT / "scripts" / "run_gatr_condor.sh"
PYTHON = PROJECT_ROOT / "gatr" / "bin" / "python"
DEFAULT_AFS_SUBMIT_ROOT = Path.home() / "private" / "grainita_condor"
DEFAULT_EOS_LINK = Path.home() / "eos"
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


def linked_project_dir(eos_link: Path) -> Path:
    """Path of this EOS project seen through an AFS->EOS symlink."""
    relative = PROJECT_ROOT.relative_to(eos_link.resolve())
    linked = eos_link / relative
    if linked.resolve() != PROJECT_ROOT:
        raise ValueError(f"{linked} does not resolve to {PROJECT_ROOT}")
    return linked


def write_worker(path: Path, project_dir: Path):
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "# Reach the EOS project through the AFS->EOS link; only this wrapper,\n"
        "# the submit file and the Condor logs live on AFS.\n"
        f"project_dir={shlex.quote(str(project_dir))}\n"
        'export GRAINITA_PROJECT_DIR="$project_dir"\n'
        'cd "$project_dir"\n\n'
        'exec "$project_dir/scripts/run_gatr_condor.sh" "$@"\n'
    )
    path.chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version-tag", type=safe_tag, default="Ayy_gamma_70k_h512")
    parser.add_argument("--run-tag", type=safe_tag, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--select-metric", choices=("auc", "loss"), default="auc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mass-loss-weight", type=float, default=1.0)
    parser.add_argument("--decay-point-loss-weight", type=float, default=1.0)
    parser.add_argument("--no-global-features", action="store_true")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--memory-gb", type=int, default=16)
    parser.add_argument("--disk-gb", type=int, default=8)
    parser.add_argument("--job-flavour", choices=FLAVOURS, default="nextweek")
    parser.add_argument(
        "--schedd", choices=("standard", "eossubmit"), default="eossubmit",
        help="eossubmit: everything on EOS via the EosSubmit pool (default); "
             "standard: AFS submit dir + standard schedd",
    )
    parser.add_argument(
        "--afs-submit-root", type=Path, default=DEFAULT_AFS_SUBMIT_ROOT,
        help="Real AFS directory for submit files and Condor logs",
    )
    parser.add_argument(
        "--eos-link", type=Path, default=DEFAULT_EOS_LINK,
        help="AFS symlink pointing to the EOS home containing this project",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Write the submit file without queueing the job",
    )
    args = parser.parse_args()

    if min(
        args.epochs, args.batch_size, args.patience,
        args.cpus, args.memory_gb, args.disk_gb,
    ) <= 0 or args.lr <= 0:
        parser.error("Epochs, batch size, patience, resources and learning rate must be positive")
    if not PYTHON.is_file() or not JOB_SCRIPT.is_file():
        parser.error("Missing gatr/bin/python or scripts/run_gatr_condor.sh")

    dataset = PROJECT_ROOT / "data" / "GATr" / f"dataset_{args.version_tag}" / "split"
    for split in ("Training", "Validation", "Testing"):
        if not any((dataset / split).glob("*.root")):
            parser.error(f"No preprocessed ROOT chunks found in {dataset / split}")

    run_tag = args.run_tag or f"gpu_{datetime.now():%Y%m%d_%H%M%S}"
    run_name = f"model_{args.version_tag}_{run_tag}"
    run_dir = PROJECT_ROOT / "trained_models" / "GATr" / run_name
    if run_dir.exists() and any(run_dir.glob("*.pt")):
        parser.error(f"{run_dir} already holds checkpoints; choose a new --run-tag")

    if args.schedd == "standard":
        submit_root = args.afs_submit_root.expanduser().absolute()
        if str(submit_root.resolve()).startswith("/eos/"):
            parser.error(f"--afs-submit-root {submit_root} resolves to EOS")
        try:
            project_dir = linked_project_dir(args.eos_link.expanduser())
        except ValueError as error:
            parser.error(f"Cannot reach the project through --eos-link: {error}")
        submit_dir = submit_root / run_name
        executable = submit_dir / "train_worker.sh"
    else:
        project_dir = PROJECT_ROOT
        submit_dir = run_dir
        executable = JOB_SCRIPT

    if not args.dry_run:
        cuda_check = subprocess.run(
            [str(PYTHON), "-c", "import torch; raise SystemExit(0 if torch.version.cuda else 'gatr/bin has CPU-only PyTorch')"],
            capture_output=True, text=True,
        )
        if cuda_check.returncode:
            parser.error(f"CUDA environment check failed: {cuda_check.stderr.strip()}")

    run_dir.mkdir(parents=True, exist_ok=True)
    submit_dir.mkdir(parents=True, exist_ok=True)
    if args.schedd == "standard":
        write_worker(executable, project_dir)
        (run_dir / "condor_submit_dir.txt").write_text(f"{submit_dir}\n")

    submit_file = submit_dir / "train.sub"
    arguments = (
        f"--version-tag {args.version_tag} --run-tag {run_tag} "
        f"--epochs {args.epochs} --batch-size {args.batch_size} "
        f"--lr {args.lr} --patience {args.patience} "
        f"--warmup-epochs {args.warmup_epochs} "
        f"--min-lr-ratio {args.min_lr_ratio} "
        f"--weight-decay {args.weight_decay} --grad-clip {args.grad_clip} "
        f"--amp {args.amp} --select-metric {args.select_metric} "
        f"--seed {args.seed} "
        f"--mass-loss-weight {args.mass_loss_weight} "
        f"--decay-point-loss-weight {args.decay_point_loss_weight}"
        + (" --no-global-features" if args.no_global_features else "")
    )
    submit_file.write_text(
        "\n".join((
            "universe = vanilla",
            f"executable = {executable}",
            f"arguments = {arguments}",
            f"initialdir = {submit_dir}",
            f'environment = "GRAINITA_PROJECT_DIR={project_dir}"',
            f"output = {submit_dir}/condor.$(ClusterId).$(ProcId).out",
            f"error = {submit_dir}/condor.$(ClusterId).$(ProcId).err",
            f"log = {submit_dir}/condor.$(ClusterId).log",
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
    print(f"Submit file: {submit_file}")
    print(f"Model outputs: {run_dir}")
    if args.dry_run:
        print(submit_file.read_text(), end="")
        return 0
    if args.schedd == "standard":
        command = ["condor_submit", str(submit_file)]
    else:
        # CERN's standard schedds reject /eos paths; select EosSubmit first.
        command = [
            "bash", "-lc",
            'module load lxbatch/eossubmit && exec condor_submit "$1"',
            "bash", str(submit_file),
        ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
