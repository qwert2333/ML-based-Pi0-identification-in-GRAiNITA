#!/usr/bin/env python3
"""Compare BDT input-feature distributions across hit-layer readouts.

For every feature, draws one panel per readout (4 layers, 1:3 merge, single
layer) with the signal (gamma, label 1) and background (pi0, label 0)
distributions normalised to unit area, using identical bins in all panels.
Each panel reports the separation <S^2> = 1/2 sum (s - b)^2 / (s + b) of the
binned, normalised distributions (0: identical, 1: disjoint). Values outside
the common range are folded into the first/last bin.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from configs.BDT_config import LEGACY_14_TRAIN_VARS  # noqa: E402

READOUTS = (
    ("4layer", "4 layers"),
    ("1to3", "1:3 merge"),
    ("1layer", "Single layer"),
)
SIGNAL_COLOR = "#2a78d6"      # gamma, label 1
BACKGROUND_COLOR = "#eb6834"  # pi0, label 0
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
INTEGER_FEATURES = {"n_hits", "n_clusters"}


def load_dataset(dataset_dir: str, splits) -> pd.DataFrame:
    frames = []
    for split in splits:
        for path in sorted(glob.glob(os.path.join(dataset_dir, split, "*.root"))):
            with uproot.open(path) as root_file:
                frames.append(
                    root_file["events"].arrays(
                        LEGACY_14_TRAIN_VARS + ["label"], library="pd"
                    )
                )
    if not frames:
        raise FileNotFoundError(f"No ROOT files under {dataset_dir} for {splits}")
    return pd.concat(frames, ignore_index=True)


def common_bins(values, feature, n_bins, lo_q, hi_q):
    pooled = np.concatenate([v[np.isfinite(v)] for v in values])
    low, high = np.percentile(pooled, [lo_q, hi_q])
    if high <= low:
        return np.array([low - 0.5, low + 0.5])
    if feature in INTEGER_FEATURES:
        low, high = np.floor(low), np.ceil(high) + 1
        step = max(1.0, np.ceil((high - low) / n_bins))
        return np.arange(low, high + step, step) - 0.5
    return np.linspace(low, high, n_bins + 1)


def normalised_hist(values, bins):
    # Under/overflow is folded into the first/last bin.
    clipped = np.clip(values[np.isfinite(values)], bins[0], bins[-1])
    counts, _ = np.histogram(clipped, bins=bins)
    total = counts.sum()
    return counts / total if total else counts.astype(float)


def separation(signal, background):
    denom = signal + background
    mask = denom > 0
    return 0.5 * np.sum((signal[mask] - background[mask]) ** 2 / denom[mask])


def style_axis(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def is_constant(values):
    finite = values[np.isfinite(values)]
    return len(finite) == 0 or np.all(finite == finite[0])


def plot_feature(feature, datasets, n_bins, lo_q, hi_q):
    # Constant readouts (e.g. layer features with a single layer) carry no
    # information and are excluded from the shared x range and y scale.
    values = [df[feature].to_numpy(dtype=float) for df in datasets.values()]
    varying = [v for v in values if not is_constant(v)] or values
    bins = common_bins(varying, feature, n_bins, lo_q, hi_q)
    fig, axes = plt.subplots(1, len(datasets), figsize=(13, 3.6), sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    ymax = 0.0
    for ax, (tag, label) in zip(axes, READOUTS):
        df = datasets[tag]
        column = df[feature].to_numpy(dtype=float)
        ax.set_title(label, fontsize=11, color=TEXT_PRIMARY, loc="left")
        ax.set_xlabel(feature, fontsize=10, color=TEXT_PRIMARY)
        style_axis(ax)
        if is_constant(column):
            ax.text(0.5, 0.5, f"constant = {column[0]:g}\nfor all events\n(no information)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color=TEXT_SECONDARY)
            ax.set_xlim(bins[0], bins[-1])
            continue
        sig = normalised_hist(column[df["label"].to_numpy() == 1], bins)
        bkg = normalised_hist(column[df["label"].to_numpy() == 0], bins)
        ymax = max(ymax, sig.max(), bkg.max())
        ax.stairs(sig, bins, fill=True, color=SIGNAL_COLOR, alpha=0.15, linewidth=0)
        ax.stairs(sig, bins, color=SIGNAL_COLOR, linewidth=2, label=r"Signal ($\gamma$)")
        ax.stairs(bkg, bins, color=BACKGROUND_COLOR, linewidth=2, linestyle="--",
                  label=r"Background ($\pi^0$)")
        ax.text(0.98, 0.95, rf"$\langle S^2\rangle$ = {separation(sig, bkg):.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color=TEXT_SECONDARY)
    axes[0].set_ylabel("Fraction of events", fontsize=10, color=TEXT_PRIMARY)
    axes[0].set_ylim(0, ymax * 1.18 if ymax > 0 else 1)
    handles, labels = next(
        (ax.get_legend_handles_labels() for ax in axes if ax.get_legend_handles_labels()[0]),
        ([], []),
    )
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(0.995, 1.0))
    fig.suptitle(feature, x=0.01, ha="left", fontsize=13, color=TEXT_PRIMARY,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "BDT"))
    parser.add_argument("--dataset-prefix", default="dataset_0.5-80GeV_")
    parser.add_argument("--splits", nargs="+",
                        default=["Training", "Validation", "Testing"])
    parser.add_argument("--output-dir", default=str(
        PROJECT_ROOT / "results" / "plots" / "bdt_feature_readouts"))
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--range-quantiles", type=float, nargs=2, default=(1.0, 99.0),
                        help="Pooled percentiles defining the common x range")
    args = parser.parse_args()

    datasets = {}
    for tag, _ in READOUTS:
        path = os.path.join(args.data_dir, f"{args.dataset_prefix}{tag}")
        datasets[tag] = load_dataset(path, args.splits)
        print(f"{tag}: {len(datasets[tag])} events from {path}")

    os.makedirs(args.output_dir, exist_ok=True)
    combined = os.path.join(args.output_dir, "all_features_readouts.pdf")
    with PdfPages(combined) as pdf:
        for feature in LEGACY_14_TRAIN_VARS:
            fig = plot_feature(feature, datasets, args.bins, *args.range_quantiles)
            fig.savefig(os.path.join(args.output_dir, f"{feature}.png"), dpi=130)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Saved {len(LEGACY_14_TRAIN_VARS)} feature plots to {args.output_dir}")
    print(f"Combined PDF: {combined}")


if __name__ == "__main__":
    main()
