#!/usr/bin/env python3
import argparse
import glob
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot


def load_dataset_from_directory(
    data_dir: str, tree_name: str = "events"
) -> pd.DataFrame:
    """Loads all chunked ROOT files from a directory into a single Pandas DataFrame."""
    root_files = sorted(glob.glob(os.path.join(data_dir, "*.root")))
    if not root_files:
        raise FileNotFoundError(f"No ROOT files found in: {data_dir}")

    print(f"--> Loading {len(root_files)} ROOT file(s) from {data_dir}...")
    dfs = []
    for fpath in root_files:
        with uproot.open(fpath) as f:
            if tree_name in f:
                df = f[tree_name].arrays(library="pd")
                dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def plot_feature_distributions(
    df: pd.DataFrame,
    output_filename: str = "feature_distributions.png",
    bins: int = 50,
    feature_name: str = None,
    e_min: float = None,
    e_max: float = None,
):
    """Generates distribution plots overlaying Signal (Gamma) vs Background (Pi0).

    Can plot either a grid of all features or a single specified feature.
    """
    ignore_cols = ["target_pdg", "target_energy", "label", "weight"]

    if feature_name:
        if feature_name not in df.columns:
            raise ValueError(f"Feature '{feature_name}' not found in DataFrame.")
        feature_cols = [feature_name]
    else:
        feature_cols = [c for c in df.columns if c not in ignore_cols]

    n_features = len(feature_cols)
    if n_features == 0:
        print("Error: No feature columns found in DataFrame.")
        return

    # Set up single subplot or multi-plot grid
    if n_features == 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        axes_flat = [ax]
    else:
        n_cols = 3
        n_rows = math.ceil(n_features / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(15, 3.5 * n_rows), squeeze=False
        )
        axes_flat = axes.flatten()

    sig = df[df["label"] == 1]
    bkg = df[df["label"] == 0]

    # Energy range string for titles
    e_str = ""
    if e_min is not None or e_max is not None:
        e_low = f"{e_min}" if e_min is not None else "0"
        e_high = f"{e_max}" if e_max is not None else "inf"
        e_str = f" [{e_low}–{e_high} GeV]"

    print(
        f"--> Plotting distributions for {n_features} feature(s){e_str} "
        f"({len(sig)} Signal vs {len(bkg)} Background)..."
    )

    for i, col in enumerate(feature_cols):
        ax = axes_flat[i]
        print(f"  [{i+1}/{n_features}] Processing feature: {col}...")

        # 1. Force numeric conversion & clean non-finite values
        sig_series = pd.to_numeric(sig[col], errors="coerce")
        bkg_series = pd.to_numeric(bkg[col], errors="coerce")

        sig_data = sig_series.replace([np.inf, -np.inf], np.nan).dropna().values
        bkg_data = bkg_series.replace([np.inf, -np.inf], np.nan).dropna().values

        combined = np.concatenate([sig_data, bkg_data])

        # 2. Compute histogram boundaries safely
        low_val, high_val = 0.0, 1.0

        if len(combined) > 0:
            c_min = np.min(combined)
            c_max = np.max(combined)

            if np.isclose(c_min, c_max, atol=1e-6) or c_min >= c_max:
                low_val = c_min - 0.5
                high_val = c_max + 0.5
            else:
                p_low, p_high = np.percentile(combined, [0.1, 99.9])
                if (
                    np.isfinite(p_low)
                    and np.isfinite(p_high)
                    and p_high > p_low
                    and not np.isclose(p_low, p_high, atol=1e-6)
                ):
                    low_val, high_val = p_low, p_high
                else:
                    low_val, high_val = c_min, c_max

        if not (np.isfinite(low_val) and np.isfinite(high_val)) or low_val >= high_val:
            low_val, high_val = 0.0, 1.0

        hist_range = (float(low_val), float(high_val))

        # 3. Plot Signal (Gamma = 1)
        if len(sig_data) > 0:
            ax.hist(
                sig_data,
                bins=bins,
                range=hist_range,
                density=True,
                histtype="stepfilled",
                alpha=0.35,
                color="#1f77b4",
                label=r"Signal ($\gamma$)",
            )
            ax.hist(
                sig_data,
                bins=bins,
                range=hist_range,
                density=True,
                histtype="step",
                linewidth=1.5,
                color="#1f77b4",
            )

        # 4. Plot Background (Pi0 = 0)
        if len(bkg_data) > 0:
            ax.hist(
                bkg_data,
                bins=bins,
                range=hist_range,
                density=True,
                histtype="stepfilled",
                alpha=0.3,
                color="#ff7f0e",
                label=r"Background ($\pi^0$)",
            )
            ax.hist(
                bkg_data,
                bins=bins,
                range=hist_range,
                density=True,
                histtype="step",
                linewidth=1.5,
                color="#d62728",
            )

        ax.set_title(f"{col}{e_str}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Normalized Density", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize=9)

    # Hide unused subplots if grid layout was created
    if n_features > 1:
        for j in range(n_features, len(axes_flat)):
            fig.delaxes(axes_flat[j])

    out_dir = os.path.dirname(output_filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"--> Saved feature distribution plot to: {output_filename}")


def plot_correlation_matrices(
    df: pd.DataFrame,
    output_filename: str = "feature_correlations.png",
):
    """Generates side-by-side Pearson correlation heatmaps for Signal, Background, and Difference."""
    ignore_cols = ["target_pdg", "target_energy", "label", "weight"]
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    if len(feature_cols) == 0:
        print("Error: No feature columns found for correlation plot.")
        return

    sig_df = df[df["label"] == 1][feature_cols]
    bkg_df = df[df["label"] == 0][feature_cols]

    corr_sig = sig_df.corr(method="pearson")
    corr_bkg = bkg_df.corr(method="pearson")
    corr_diff = (corr_sig - corr_bkg).abs()

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    def draw_heatmap(ax, corr_data, title, cmap, vmin, vmax):
        im = ax.imshow(
            corr_data.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto"
        )
        ax.set_xticks(np.arange(len(feature_cols)))
        ax.set_yticks(np.arange(len(feature_cols)))
        ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(feature_cols, fontsize=8)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

        if len(feature_cols) <= 20:
            for i in range(len(feature_cols)):
                for j in range(len(feature_cols)):
                    val = corr_data.iloc[i, j]
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color="black" if abs(val) < 0.75 else "white",
                        fontsize=6,
                    )
        return im

    im1 = draw_heatmap(
        axes[0],
        corr_sig,
        r"Signal ($\gamma$) Correlation Matrix",
        "coolwarm",
        -1.0,
        1.0,
    )
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = draw_heatmap(
        axes[1],
        corr_bkg,
        r"Background ($\pi^0$) Correlation Matrix",
        "coolwarm",
        -1.0,
        1.0,
    )
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    im3 = draw_heatmap(
        axes[2],
        corr_diff,
        r"Correlation Difference ($|R_{\gamma} - R_{\pi^0}|$)",
        "YlOrRd",
        0.0,
        1.0,
    )
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"--> Saved correlation matrices to: {output_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot feature distributions for signal and background."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing ROOT files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="feature_distributions.png",
        help="Output image file path or output directory",
    )
    parser.add_argument(
        "-f",
        "--feature",
        type=str,
        default=None,
        help="Specific feature name to plot (e.g., 'leading_cluster_ratio'). If omitted, plots all features.",
    )
    parser.add_argument(
        "--e-min",
        type=float,
        default=None,
        help="Minimum target energy cut in GeV",
    )
    parser.add_argument(
        "--e-max",
        type=float,
        default=None,
        help="Maximum target energy cut in GeV",
    )
    parser.add_argument(
        "-c",
        "--correlation",
        action="store_true",
        help="Also compute correlation heatmaps",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of bins (default: 50)",
    )
    args = parser.parse_args()

    # Load dataset
    df = load_dataset_from_directory(args.input_dir)

    # Apply energy range cuts if requested
    if "target_energy" in df.columns:
        initial_len = len(df)
        if args.e_min is not None:
            df = df[df["target_energy"] >= args.e_min]
        if args.e_max is not None:
            df = df[df["target_energy"] <= args.e_max]
        if args.e_min is not None or args.e_max is not None:
            print(
                f"--> Applied energy cut [{args.e_min}, {args.e_max}] GeV: "
                f"Kept {len(df)} / {initial_len} events."
            )
    else:
        if args.e_min is not None or args.e_max is not None:
            print(
                "Warning: 'target_energy' column not found in DataFrame. Skipping energy cuts."
            )

    # Automatically handle output path vs output directory
    out_dist = args.output
    if os.path.isdir(out_dist) or out_dist.endswith("/"):
        os.makedirs(out_dist, exist_ok=True)
        fname = (
            f"{args.feature}_dist.png"
            if args.feature
            else "feature_distributions.png"
        )
        out_dist = os.path.join(out_dist, fname)

    plot_feature_distributions(
        df,
        output_filename=out_dist,
        bins=args.bins,
        feature_name=args.feature,
        e_min=args.e_min,
        e_max=args.e_max,
    )

    if args.correlation:
        out_corr = os.path.join(
            os.path.dirname(out_dist) or ".", "feature_correlations.png"
        )
        plot_correlation_matrices(df, output_filename=out_corr)


if __name__ == "__main__":
    main()