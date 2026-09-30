"""Shared GATr / baseline evaluation metrics.

Classification metrics are conditional on a valid single CLUE cluster: they
describe merged-diphoton vs single-photon cluster separation only.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

BACKGROUND_EFFICIENCIES = (1.0e-3, 1.0e-2, 5.0e-2, 1.0e-1)
ENERGY_BINS_GEV = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 100.0)
NOMINAL_MASSES_GEV = (0.05, 0.10, 0.135, 0.20, 0.50, 1.0, 2.0)


def _eff_key(eff):
    return f"sig_eff_at_bkg_eff_{eff:g}"


def signal_efficiency_at_background_efficiency(labels, scores, background_eff):
    """Signal efficiency for the score threshold passing ``background_eff``."""
    background = np.sort(scores[labels == 0])
    signal = scores[labels == 1]
    if not len(background) or not len(signal):
        return float("nan"), float("nan")
    index = int(np.ceil((1.0 - background_eff) * len(background))) - 1
    threshold = background[min(max(index, 0), len(background) - 1)]
    return float(np.mean(signal > threshold)), float(threshold)


def classification_metrics(
    labels,
    scores,
    energy=None,
    working_point=0.5,
    background_efficiencies=BACKGROUND_EFFICIENCIES,
    energy_bins=ENERGY_BINS_GEV,
):
    """ROC AUC, fixed-background-efficiency points, working point, energy bins."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    metrics = {
        "n_signal": int(labels.sum()),
        "n_background": int((labels == 0).sum()),
    }
    if len(np.unique(labels)) < 2:
        return metrics
    metrics["auc"] = float(roc_auc_score(labels, scores))
    for eff in background_efficiencies:
        sig_eff, threshold = signal_efficiency_at_background_efficiency(
            labels, scores, eff
        )
        metrics[_eff_key(eff)] = sig_eff
        metrics[f"threshold_at_bkg_eff_{eff:g}"] = threshold

    predicted = (scores >= working_point).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    metrics["working_point"] = {
        "threshold": float(working_point),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "signal_efficiency": float(tp / max(tp + fn, 1)),
        "background_efficiency": float(fp / max(fp + tn, 1)),
        "precision": float(tp / max(tp + fp, 1)),
    }

    if energy is not None:
        energy = np.asarray(energy, dtype=np.float64)
        per_bin = []
        for low, high in zip(energy_bins[:-1], energy_bins[1:]):
            select = (energy >= low) & (energy < high)
            entry = {
                "energy_low_GeV": float(low),
                "energy_high_GeV": float(high),
                "n_signal": int(labels[select].sum()),
                "n_background": int((labels[select] == 0).sum()),
            }
            if len(np.unique(labels[select])) > 1:
                entry["auc"] = float(roc_auc_score(labels[select], scores[select]))
                entry["background_efficiency_at_wp"] = float(
                    np.mean(predicted[select][labels[select] == 0])
                )
                entry["signal_efficiency_at_wp"] = float(
                    np.mean(predicted[select][labels[select] == 1])
                )
            per_bin.append(entry)
        metrics["energy_bins"] = per_bin
    return metrics


def nearest_nominal_mass(mass, nominal=NOMINAL_MASSES_GEV):
    nominal = np.asarray(nominal, dtype=np.float64)
    log_mass = np.log(np.clip(np.asarray(mass, dtype=np.float64), 1.0e-6, None))
    index = np.argmin(np.abs(log_mass[:, None] - np.log(nominal)[None, :]), axis=1)
    return nominal[index]


def mass_metrics(predicted, target, nominal=NOMINAL_MASSES_GEV):
    """Aggregate and per-nominal-mass regression metrics in GeV."""
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if not len(target):
        return {}
    residual = predicted - target
    metrics = {
        "n": int(len(target)),
        "mae_GeV": float(np.mean(np.abs(residual))),
        "rmse_GeV": float(np.sqrt(np.mean(residual**2))),
        "per_mass": [],
    }
    groups = nearest_nominal_mass(target, nominal)
    for mass in np.unique(groups):
        select = groups == mass
        metrics["per_mass"].append({
            "nominal_mass_GeV": float(mass),
            "n": int(select.sum()),
            "mean_prediction_GeV": float(predicted[select].mean()),
            "bias_GeV": float(residual[select].mean()),
            "mae_GeV": float(np.mean(np.abs(residual[select]))),
            "relative_resolution": float(
                np.std(residual[select]) / max(float(mass), 1.0e-6)
            ),
        })
    return metrics


def decay_point_metrics(predicted, target):
    """3D-distance metrics in mm."""
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if not len(target):
        return {}
    distance = np.linalg.norm(predicted - target, axis=1)
    return {
        "n": int(len(target)),
        "mae_mm": float(distance.mean()),
        "rmse_mm": float(np.sqrt(np.mean(distance**2))),
        "median_mm": float(np.median(distance)),
    }


def format_classification_summary(metrics):
    """One-line summary for logs."""
    if "auc" not in metrics:
        return "AUC n/a (single class)"
    parts = [f"AUC {metrics['auc']:.4f}"]
    for eff in BACKGROUND_EFFICIENCIES:
        key = _eff_key(eff)
        if key in metrics:
            parts.append(f"eS@eB={eff:g} {metrics[key]:.3f}")
    return " | ".join(parts)
