# ML-based $\pi^0$ Identification in GRAiNITA

[![GitHub Repository](https://img.shields.io/badge/GitHub-mdgr03ninger-blue?logo=github)](https://github.com/mdgr03ninger/ML-based-Pi0-identification-in-GRAiNITA)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

This repository contains the machine learning framework and analysis pipeline for **neutral pion ($\pi^0$) identification and energy reconstruction within the GRAiNITA calorimeter concept**.

The project uses geometric machine learning architectures (GATr) to improve photon pair separation and $\pi^0$ reconstruction performance in granular electromagnetic calorimeters. It uses boosted decision trees as a baseline.

---

## Project Report

The complete written report accompanying this work is available on [CDS]()

---

## Environment & Setup (lxplus)

To run this project on CERN's **lxplus** cluster, follow the environment configuration steps below.

### 1. Load CVMFS LCG Stack & Create Virtual Environment

Log into `lxplus` and source a modern LCG release via CVMFS to load system dependencies (Python, ROOT, CMake, GCC). In developement the following stack has been used:

```bash
# Source LCG Software View 
source source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

# Create and activate a local virtual environment
python -m venv .venv
source .venv/bin/activate

# Install GaTR (Geometric Algebra Transformer) and core ML stack
pip install gatr

# Install this repository in editable mode (enables `import src.x`)
pip install -e .
```
## Repository structure
```

├── configs/
├── GRAiNITA_performance/
│   ├── 68_method/
│   └── fit_method/
├── plotting_scripts/
├── scripts/
│   ├── run_preprocessing.py
│   └── run_training.py
├── src/
│   ├── evaluation.py
│   ├── split_utils.py
│   ├── models/
│   ├── preprocessing/
│   └── training/
├── pyproject.toml
└── README.md
```


## HTCondor GPU training

Preprocess the 70k sample with up to 512 hits per cluster (written to `data/GATr/dataset_Ayy_gamma_70k_h512/`), compute the simple-feature baselines, then submit training:

```bash
gatr/bin/python scripts/preprocess_Ayy_gamma_70k.py
gatr/bin/python scripts/run_gatr_baselines.py --version-tag Ayy_gamma_70k_h512
./scripts/submit_gatr_condor.py --version-tag Ayy_gamma_70k_h512
```

Training defaults: 60 epochs, AdamW with peak LR 5e-4, 2-epoch linear warmup and cosine decay, gradient clipping at 1.0, bf16 autocast, per-epoch reshuffling, seed 42. The classification and mass heads also receive batch-normalised cluster-level absolute-scale features and the front-layer energy fraction, which preprocessing computes from all hits before truncation (`--no-global-features` disables them; datasets preprocessed before this branch existed need `--no-global-features` or re-preprocessing). `gatr_best_model.pt` is selected by validation AUC (`--select-metric loss` for total loss); `gatr_best_loss_model.pt` and `gatr_last_model.pt` are saved alongside. Per-epoch metrics go to `epoch_metrics.json`, test metrics (AUC, signal efficiency at fixed background efficiency, per-energy-bin AUC, per-mass regression) to `test_metrics.json`, and the log compares them with `results/metrics/gatr_baselines_<version-tag>.json`.

By default the helper writes the submit description and Condor logs to the EOS run directory `trained_models/GATr/model_<version-tag>_<run-tag>/` and submits through CERN EosSubmit (`module load lxbatch/eossubmit`); it refuses a run tag whose directory already holds checkpoints. `--schedd standard` instead keeps the submit file, a worker wrapper and the Condor logs in the real AFS directory `~/private/grainita_condor/` and submits through the standard schedd, reaching the EOS project through the AFS->EOS link `~/eos` (`--afs-submit-root`, `--eos-link`). Jobs request one GPU, four CPUs, 16 GB of RAM, and the CERN `nextweek` job flavour by default. Use `--dry-run` to inspect the submit description without queueing a job. The worker runs `gatr/bin/python` and writes `gatr_best_model.pt` and `training.log` to the same model directory.

Before submission, the helper checks that `gatr/bin` contains a CUDA-enabled PyTorch build. A CPU-only PyTorch installation cannot use an allocated GPU; replace it with a CUDA-enabled build compatible with the batch nodes before submitting.
