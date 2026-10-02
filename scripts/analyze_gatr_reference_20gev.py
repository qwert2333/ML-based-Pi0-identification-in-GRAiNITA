#!/usr/bin/env python3
"""Compare 18--22 GeV reference GATr performance events with local 20 GeV samples."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import uproot
from scipy.stats import ks_2samp
from sklearn.metrics import confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

from models.gatr_models import CalorimeterGATrClassifier

REFERENCE = ROOT / "data/reference_performance_18_22GeV/gatr_reference_18_22GeV.root"
LOCAL_DIR = ROOT / "data/inference_20GeV/GATr"
LOCAL_SCORES = ROOT / "results/inference_20GeV/gatr_event_scores.csv"
MODEL = ROOT / "trained_models/GATr/model_dataset_preselection_0.5-80GeV_v1/gatr_best_model.pt"
OUT = ROOT / "results/gatr_reference_18_22GeV"


def load_events(paths):
    main_parts, extra_parts = [], []
    for path in paths:
        with uproot.open(path) as root_file:
            main_parts.append(root_file["CLUEShowers"].arrays(library="np"))
            extra_parts.append(root_file["CLUEExtra"].arrays(library="np"))
    main = {key: np.concatenate([part[key] for part in main_parts]) for key in main_parts[0]}
    extra = {key: np.concatenate([part[key] for part in extra_parts]) for key in extra_parts[0]}
    return main, extra


def shower_angles(main, extra):
    mask = main["hit_mask"].astype(bool) & (main["hit_E_raw"] > 0)
    weights = main["hit_E_raw"] * mask
    denominator = np.maximum(weights.sum(axis=1), 1.0e-12)
    centroid = []
    for axis in "xyz":
        centroid.append((extra[f"hit_{axis}"] * weights).sum(axis=1) / denominator)
    x, y, z = centroid
    theta = np.degrees(np.arctan2(np.hypot(x, y), z))
    phi = np.degrees(np.arctan2(y, x))
    return theta, phi


@torch.no_grad()
def infer(main, batch_size=32):
    checkpoint = torch.load(MODEL, map_location="cpu")
    model = CalorimeterGATrClassifier(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    scores = []
    for start in range(0, len(main["label"]), batch_size):
        stop = min(start + batch_size, len(main["label"]))
        tensors = {
            key: torch.from_numpy(main[key][start:stop]).float()
            for key in ("hit_x_norm", "hit_y_norm", "hit_z_norm", "hit_E_raw")
        }
        mask = torch.from_numpy(main["hit_mask"][start:stop]).bool()
        logits = model(
            tensors["hit_x_norm"], tensors["hit_y_norm"], tensors["hit_z_norm"],
            tensors["hit_E_raw"], extra_scalars=None, mask=mask,
        )
        scores.append(torch.sigmoid(logits).cpu().numpy())
        print(f"inference {stop}/{len(main['label'])}", flush=True)
    return np.concatenate(scores)


def describe(frame, prefix):
    result = {}
    for label, sample in ((0, "gamma"), (1, "pi0")):
        selected = frame[frame.label == label]
        result[sample] = {"n": int(len(selected))}
        for name in ("theta_deg", "phi_deg", "pi0_score", "total_energy", "n_hits"):
            values = selected[name].to_numpy()
            result[sample][name] = {
                "mean": float(values.mean()), "std": float(values.std()),
                "min": float(values.min()), "max": float(values.max()),
            }
    return {prefix: result}


def plot_reference(frame, matrix):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for label, sample, color in ((0, "gamma", "crimson"), (1, "pi0", "royalblue")):
        selected = frame[frame.label == label]
        axes[0, 0].hist(selected.theta_deg, bins=35, density=True, histtype="step", lw=2,
                        color=color, label=sample)
        axes[0, 1].hist(selected.phi_deg, bins=35, density=True, histtype="step", lw=2,
                        color=color, label=sample)
        axes[1, 0].hist(selected.pi0_score, bins=40, range=(0, 1), density=True,
                        histtype="step", lw=2, color=color, label=sample)
    for axis, xlabel in ((axes[0, 0], "shower-axis theta [deg]"),
                          (axes[0, 1], "shower-axis phi [deg]"),
                          (axes[1, 0], "GATr pi0 score")):
        axis.set_xlabel(xlabel); axis.set_ylabel("normalized density"); axis.grid(alpha=0.25); axis.legend()
    axes[1, 0].axvline(0.5, color="black", ls="--", lw=1)
    image = axes[1, 1].imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1))
    for row in range(2):
        for col in range(2):
            axes[1, 1].text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=14)
    axes[1, 1].set_xticks([0, 1], ["pred gamma", "pred pi0"])
    axes[1, 1].set_yticks([0, 1], ["true gamma", "true pi0"])
    axes[1, 1].set_title("threshold = 0.5")
    fig.colorbar(image, ax=axes[1, 1]); fig.tight_layout()
    fig.savefig(OUT / "reference_18_22GeV_distributions.png", dpi=180)
    fig.savefig(OUT / "reference_18_22GeV_distributions.pdf"); plt.close(fig)


def plot_comparison(reference, local):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    variables = (("theta_deg", "shower-axis theta [deg]", None),
                 ("phi_deg", "shower-axis phi [deg]", None),
                 ("pi0_score", "GATr pi0 score", (0, 1)))
    for axis, (name, xlabel, value_range) in zip(axes, variables):
        for label, sample, color in ((0, "gamma", "crimson"), (1, "pi0", "royalblue")):
            ref = reference.loc[reference.label == label, name]
            new = local.loc[local.label == label, name]
            bins = np.linspace(*value_range, 41) if value_range else 35
            axis.hist(ref, bins=bins, density=True, histtype="step", lw=2, color=color,
                      label=f"reference {sample}")
            axis.hist(new, bins=bins, density=True, histtype="step", lw=2, ls="--", color=color,
                      label=f"local {sample}")
        axis.set_xlabel(xlabel); axis.set_ylabel("normalized density"); axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "reference_vs_local_20GeV.png", dpi=180)
    fig.savefig(OUT / "reference_vs_local_20GeV.pdf"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reference_main, reference_extra = load_events([REFERENCE])
    reference_theta, reference_phi = shower_angles(reference_main, reference_extra)
    reference_scores = infer(reference_main)
    reference = pd.DataFrame({
        "true_energy": reference_main["true_mc_energy"], "label": reference_main["label"],
        "theta_deg": reference_theta, "phi_deg": reference_phi, "pi0_score": reference_scores,
        "total_energy": (reference_main["hit_E_raw"] * reference_main["hit_mask"]).sum(axis=1),
        "n_hits": reference_main["n_hits"],
    })

    local_main, local_extra = load_events([
        LOCAL_DIR / "gamma_preprocessed.root", LOCAL_DIR / "pi0_preprocessed.root"
    ])
    local_theta, local_phi = shower_angles(local_main, local_extra)
    local_scores = pd.read_csv(LOCAL_SCORES).pi0_score.to_numpy()
    local = pd.DataFrame({
        "true_energy": local_main["true_mc_energy"], "label": local_main["label"],
        "theta_deg": local_theta, "phi_deg": local_phi, "pi0_score": local_scores,
        "total_energy": (local_main["hit_E_raw"] * local_main["hit_mask"]).sum(axis=1),
        "n_hits": local_main["n_hits"],
    })

    predicted = (reference.pi0_score.to_numpy() >= 0.5).astype(int)
    matrix = confusion_matrix(reference.label, predicted, labels=[0, 1])
    metrics = {
        "reference_auc": float(roc_auc_score(reference.label, reference.pi0_score)),
        "reference_confusion_matrix_at_0p5": matrix.tolist(),
        "reference_gamma_efficiency_at_0p5": float(matrix[0, 0] / matrix[0].sum()),
        "reference_pi0_efficiency_at_0p5": float(matrix[1, 1] / matrix[1].sum()),
        **describe(reference, "reference"), **describe(local, "local"), "ks_reference_vs_local": {},
    }
    for label, sample in ((0, "gamma"), (1, "pi0")):
        metrics["ks_reference_vs_local"][sample] = {}
        for name in ("theta_deg", "phi_deg", "pi0_score", "total_energy", "n_hits"):
            test = ks_2samp(reference.loc[reference.label == label, name], local.loc[local.label == label, name])
            metrics["ks_reference_vs_local"][sample][name] = {
                "statistic": float(test.statistic), "pvalue": float(test.pvalue)
            }
    reference.to_csv(OUT / "reference_event_scores_angles.csv", index=False)
    local.to_csv(OUT / "local_event_scores_angles.csv", index=False)
    with (OUT / "metrics_and_shift.json").open("w") as output:
        json.dump(metrics, output, indent=2)
    plot_reference(reference, matrix); plot_comparison(reference, local)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
