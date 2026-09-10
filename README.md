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

