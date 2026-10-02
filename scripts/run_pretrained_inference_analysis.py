#!/usr/bin/env python3
"""Run BDT/GATr inference in separate low-memory stages and combine results."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

DATA = ROOT / "data/inference_20GeV"
OUT = ROOT / "results/inference_20GeV"
GATR_MODEL = ROOT / "trained_models/GATr/model_dataset_preselection_0.5-80GeV_v1/gatr_best_model.pt"
BDT_MODEL = ROOT / "trained_models/BDT/model_bdt_preselection/clue_gamma_pi0_bdt_model.joblib"


def bdt_inference():
    import joblib

    saved = joblib.load(BDT_MODEL)
    model = saved.model if hasattr(saved, "model") else saved
    parts = []
    for sample, true_pi0 in (("gamma", 0), ("pi0", 1)):
        with uproot.open(DATA / "BDT" / f"{sample}_features.root") as root_file:
            arrays = root_file["events"].arrays(library="np")
        frame = pd.DataFrame({name: values for name, values in arrays.items()})
        frame["sample"], frame["true_pi0"] = sample, true_pi0
        parts.append(frame)
    data = pd.concat(parts, ignore_index=True)
    # Training preprocessing labels gamma=1 and pi0=0.
    scores = 1.0 - model.predict_proba(data[list(model.feature_names_in_)])[:, 1]
    return data.true_pi0.to_numpy(), scores, data["sample"].to_numpy()


def gatr_inference(batch_size=16):
    import torch
    from torch.utils.data import DataLoader
    from models.gatr_models import CalorimeterGATrClassifier, GATrWrapper, ROOTShowerDataset

    checkpoint = torch.load(GATR_MODEL, map_location="cpu")
    model = CalorimeterGATrClassifier(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    wrapper = GATrWrapper(model, device=torch.device("cpu"))
    labels, scores, samples = [], [], []
    for sample, true_pi0 in (("gamma", 0), ("pi0", 1)):
        source = DATA / "GATr" / f"{sample}_preprocessed.root"
        loader = DataLoader(ROOTShowerDataset([str(source)], batch_size=batch_size), batch_size=None)
        prediction = wrapper.predict_proba(loader)[:, 1]
        scores.append(prediction)
        labels.append(np.full(len(prediction), true_pi0, dtype=np.int64))
        samples.append(np.full(len(prediction), sample, dtype=object))
    return np.concatenate(labels), np.concatenate(scores), np.concatenate(samples)


def bootstrap_auc(labels, scores, repeats=2000):
    rng = np.random.default_rng(42)
    gamma, pi0 = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    values = []
    for _ in range(repeats):
        indices = np.r_[rng.choice(gamma, len(gamma), replace=True), rng.choice(pi0, len(pi0), replace=True)]
        values.append(roc_auc_score(labels[indices], scores[indices]))
    return np.quantile(values, (0.025, 0.975)).tolist()


def summarize(labels, scores):
    predicted = (scores >= 0.5).astype(np.int64)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    best = int(np.argmax(tpr - fpr))
    metrics = {
        "n_gamma": int(np.sum(labels == 0)), "n_pi0": int(np.sum(labels == 1)),
        "auc": float(roc_auc_score(labels, scores)),
        "auc_bootstrap_95pct": bootstrap_auc(labels, scores),
        "accuracy_at_0p5": float(accuracy_score(labels, predicted)),
        "confusion_matrix_at_0p5": matrix.tolist(),
        "gamma_efficiency_at_0p5": float(matrix[0, 0] / matrix[0].sum()),
        "pi0_efficiency_at_0p5": float(matrix[1, 1] / matrix[1].sum()),
        "best_youden_threshold": float(thresholds[best]),
        "gamma_rejection_at_best_youden": float(1.0 - fpr[best]),
        "pi0_efficiency_at_best_youden": float(tpr[best]),
        "score_summary": {},
    }
    for label, name in ((0, "gamma"), (1, "pi0")):
        values = scores[labels == label]
        metrics["score_summary"][name] = {
            "mean": float(values.mean()), "std": float(values.std()),
            "median": float(np.median(values)),
            "q05": float(np.quantile(values, 0.05)), "q95": float(np.quantile(values, 0.95)),
        }
    return metrics, fpr, tpr


def save_result(name, labels, scores, samples):
    metrics, _, _ = summarize(labels, scores)
    pd.DataFrame({"sample": samples, "true_pi0": labels, "pi0_score": scores,
                  "predicted_pi0_at_0p5": scores >= 0.5}).to_csv(
                      OUT / f"{name.lower()}_event_scores.csv", index=False)
    with (OUT / f"{name.lower()}_metrics.json").open("w") as output:
        json.dump(metrics, output, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)


def load_result(name):
    frame = pd.read_csv(OUT / f"{name.lower()}_event_scores.csv")
    labels = frame.true_pi0.to_numpy(dtype=np.int64)
    scores = frame.pi0_score.to_numpy(dtype=float)
    metrics, fpr, tpr = summarize(labels, scores)
    return {"labels": labels, "scores": scores, "metrics": metrics, "fpr": fpr, "tpr": tpr}


def combine_results():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {name: load_result(name) for name in ("BDT", "GATr")}
    with (OUT / "metrics.json").open("w") as output:
        json.dump({name: result["metrics"] for name, result in results.items()}, output, indent=2)
    colors = {"GATr": "#3366cc", "BDT": "#dd4477"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, result in results.items():
        labels, scores = result["labels"], result["scores"]
        for label, sample, style in ((0, "gamma", "--"), (1, "pi0", "-")):
            axes[0].hist(scores[labels == label], bins=20, range=(0, 1), density=True,
                         histtype="step", linewidth=2, linestyle=style,
                         color=colors[name], label=f"{name} {sample}")
        axes[1].plot(result["fpr"], result["tpr"], linewidth=2, color=colors[name],
                     label=f"{name} (AUC={result['metrics']['auc']:.4f})")
    axes[0].axvline(0.5, color="black", linewidth=1, alpha=0.6)
    axes[0].set(xlabel="pi0 score", ylabel="normalized density", xlim=(0, 1))
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.25)
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[1].set(xlabel="gamma misidentification rate", ylabel="pi0 efficiency",
                xlim=(0, 1), ylim=(0, 1.02))
    axes[1].legend(); axes[1].grid(alpha=0.25)
    fig.suptitle("20 GeV gamma vs pi0 discrimination"); fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=180)
    fig.savefig(OUT / "model_comparison.pdf"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("bdt", "gatr", "combine"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "bdt":
        save_result("BDT", *bdt_inference())
    elif args.stage == "gatr":
        save_result("GATr", *gatr_inference())
    else:
        combine_results()
        print(f"Results written to {OUT}", flush=True)


if __name__ == "__main__":
    main()
