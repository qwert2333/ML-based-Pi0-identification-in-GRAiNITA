#!/usr/bin/env bash
set -euo pipefail

project_dir="${GRAINITA_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export GRAINITA_PROJECT_DIR="$project_dir"
cd "$project_dir"

python_bin="$project_dir/gatr/bin/python"
if [[ ! -x "$python_bin" ]]; then
    echo "Missing Python environment: $python_bin" >&2
    exit 1
fi

"$python_bin" - <<'PY'
import os
import torch

print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}", flush=True)
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
if not torch.cuda.is_available():
    raise SystemExit("GPU requested, but this Python environment cannot use CUDA")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
PY

exec "$python_bin" -u scripts/run_training.py --model gatr "$@"
