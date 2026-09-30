#!/usr/bin/env python3
"""Simple-feature and constant baselines for the GATr cluster tasks.

Uses the same preprocessed splits as GATr training. The classifier is a
HistGradientBoosting model on cluster-level features (energy, hit count and
the absolute-scale and front-layer-fraction features that the GATr heads
get);
regression baselines are constants, a HistGradientBoosting regressor, and a
decay point placed along the origin->centroid direction. Results are written
to results/metrics/gatr_baselines_<version-tag>.json, which the GATr training
reads to log a direct test-set comparison.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-tag", default="Ayy_gamma_70k_h512")
    parser.add_argument("--threads", type=int, default=4,
                        help="OpenMP threads for the boosted trees")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--working-point", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


ARGS = parse_args()
# Must be set before sklearn/torch start their thread pools; many threads on
# a shared node make the boosted trees dramatically slower.
os.environ["OMP_NUM_THREADS"] = str(ARGS.threads)
os.environ.setdefault("GRAINITA_PROJECT_DIR", str(PROJECT_ROOT))

import numpy as np
import torch
import uproot
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from configs.paths import FilePaths
from src.gatr_metrics import (
    classification_metrics,
    decay_point_metrics,
    format_classification_summary,
    mass_metrics,
)
from src.models.gatr_models import GLOBAL_FEATURE_NAMES, cluster_global_features

MAIN_BRANCHES = [
    "hit_E_raw", "hit_mask", "label", "classification_valid",
    "regression_valid", "cluster_energy", "n_hits_original",
    "cluster_energy_fraction_kept", "cluster_first_layer_energy_fraction",
    "cluster_x", "cluster_y", "cluster_z",
    "llp_mass_GeV", "llp_decay_x_mm", "llp_decay_y_mm", "llp_decay_z_mm",
    "llp_decay_length_mm",
]
EXTRA_BRANCHES = ["hit_x", "hit_y", "hit_z"]


def load_split(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    parts = {}
    for path in files:
        with uproot.open(path) as root_file:
            main = root_file["CLUEShowers"].arrays(MAIN_BRANCHES, library="np")
            extra = root_file["CLUEExtra"].arrays(EXTRA_BRANCHES, library="np")
        for source in (main, extra):
            for key, value in source.items():
                parts.setdefault(key, []).append(value)
    return {key: np.concatenate(value) for key, value in parts.items()}


def cluster_features(data, chunk=4096):
    """Feature matrix and names: energy/hit count plus GATr global features."""
    energy = data["hit_E_raw"] * data["hit_mask"]
    fraction = energy / np.clip(energy.sum(axis=1, keepdims=True), 1e-12, None)
    globals_ = []
    for start in range(0, len(energy), chunk):
        stop = start + chunk
        coordinates = torch.from_numpy(np.stack(
            [data[key][start:stop] for key in EXTRA_BRANCHES], axis=-1
        ))
        globals_.append(cluster_global_features(
            coordinates,
            torch.from_numpy(fraction[start:stop]),
            torch.from_numpy(data["hit_mask"][start:stop].astype(bool)),
            torch.from_numpy(
                data["cluster_first_layer_energy_fraction"][start:stop]
            ),
        ).numpy())
    names = [
        "log_cluster_energy", "log_n_hits_original",
        "energy_fraction_kept", "max_hit_energy_fraction",
    ] + list(GLOBAL_FEATURE_NAMES)
    matrix = np.column_stack([
        np.log(np.clip(data["cluster_energy"], 1e-6, None)),
        np.log(np.clip(data["n_hits_original"], 1, None)),
        data["cluster_energy_fraction_kept"],
        fraction.max(axis=1),
        np.concatenate(globals_),
    ])
    return matrix, names


def decay_targets(data, select):
    return np.column_stack([
        data[f"llp_decay_{axis}_mm"][select] for axis in "xyz"
    ])


def centroid_direction(data, select):
    centre = np.column_stack([data[f"cluster_{axis}"][select] for axis in "xyz"])
    return centre / np.clip(np.linalg.norm(centre, axis=1, keepdims=True), 1e-6, None)


def main():
    paths = FilePaths()
    paths.ensure_dirs()
    patterns = paths.get_gatr_file_patterns(ARGS.version_tag)
    train, test = load_split(patterns["train"]), load_split(patterns["test"])
    x_train, names = cluster_features(train)
    x_test, _ = cluster_features(test)
    column = {name: index for index, name in enumerate(names)}

    ctr = train["classification_valid"].astype(bool)
    cte = test["classification_valid"].astype(bool)
    y_train, y_test = train["label"][ctr], test["label"][cte]
    energy_test = test["cluster_energy"][cte]

    feature_sets = {
        "hgb_all_features": names,
        "hgb_energy_only": ["log_cluster_energy"],
        "hgb_energy_and_hits": ["log_cluster_energy", "log_n_hits_original"],
        "hgb_scale_and_first_layer_only": list(GLOBAL_FEATURE_NAMES),
    }
    results = {"version_tag": ARGS.version_tag, "classification": {},
               "mass": {}, "decay_point": {}, "feature_names": names}
    for label, selected in feature_sets.items():
        index = [column[name] for name in selected]
        model = HistGradientBoostingClassifier(
            max_iter=300, random_state=ARGS.seed
        ).fit(x_train[ctr][:, index], y_train)
        scores = model.predict_proba(x_test[cte][:, index])[:, 1]
        metrics = classification_metrics(
            y_test, scores, energy=energy_test, working_point=ARGS.working_point
        )
        results["classification"][label] = metrics
        print(f"{label:28s} {format_classification_summary(metrics)}", flush=True)

    rtr = train["regression_valid"].astype(bool)
    rte = test["regression_valid"].astype(bool)
    mass_train, mass_test = train["llp_mass_GeV"][rtr], test["llp_mass_GeV"][rte]
    results["mass"]["constant_median"] = mass_metrics(
        np.full_like(mass_test, np.median(mass_train)), mass_test
    )
    regressor = HistGradientBoostingRegressor(
        max_iter=300, random_state=ARGS.seed
    ).fit(x_train[rtr], np.log1p(mass_train))
    results["mass"]["hgb_all_features"] = mass_metrics(
        np.expm1(regressor.predict(x_test[rte])), mass_test
    )

    decay_train, decay_test = decay_targets(train, rtr), decay_targets(test, rte)
    results["decay_point"]["origin"] = decay_point_metrics(
        np.zeros_like(decay_test), decay_test
    )
    results["decay_point"]["constant_mean"] = decay_point_metrics(
        np.broadcast_to(decay_train.mean(axis=0), decay_test.shape), decay_test
    )
    length_regressor = HistGradientBoostingRegressor(
        max_iter=300, random_state=ARGS.seed
    ).fit(x_train[rtr], train["llp_decay_length_mm"][rtr])
    results["decay_point"]["centroid_direction_hgb_length"] = decay_point_metrics(
        centroid_direction(test, rte)
        * np.clip(length_regressor.predict(x_test[rte]), 0.0, None)[:, None],
        decay_test,
    )
    for task, unit in (("mass", "mae_GeV"), ("decay_point", "mae_mm")):
        for label, metrics in results[task].items():
            print(f"{task:12s} {label:32s} MAE {metrics[unit]:.4g}", flush=True)

    output = ARGS.output or os.path.join(
        paths.EVAL_METRICS_DIR, f"gatr_baselines_{ARGS.version_tag}.json"
    )
    with open(output, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved baselines to {output}")


if __name__ == "__main__":
    main()
