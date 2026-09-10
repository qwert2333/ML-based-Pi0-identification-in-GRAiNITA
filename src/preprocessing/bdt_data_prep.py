import glob
import os

import numpy as np
import pandas as pd
import awkward as ak
import uproot

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from typing import Optional


from configs.BDT_config import TREE_NAME
from split_utils import safe_to_numpy




def load_and_extract_multiple_files(file_paths, extension=True):
    """Utility to process multiple raw ROOT files and combine their feature DataFrames."""

    # 1. Handle single string input if passed
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    # 2. Expand directories and glob patterns into actual .root file paths
    expanded_paths = []
    for p in file_paths:
        if not p:
            continue
        if os.path.isdir(p):
            # Recursively find all .root files inside the directory
            matched = sorted(
                glob.glob(os.path.join(p, "**", "*.root"), recursive=True)
            )
            if not matched:
                matched = sorted(glob.glob(os.path.join(p, "*.root")))
            expanded_paths.extend(matched)
        elif "*" in p or "?" in p:
            expanded_paths.extend(sorted(glob.glob(p)))
        else:
            expanded_paths.append(p)

    # Remove duplicates while preserving order
    expanded_paths = list(dict.fromkeys(expanded_paths))

    if not expanded_paths:
        print(f"Warning: No valid file paths found matching: {file_paths}")
        return pd.DataFrame()

    dfs = []
    skipped_count = 0

    print(f"--> Found {len(expanded_paths)} file(s) to process.")

    # 3. Process each .root file
    for fpath in expanded_paths:
        if not os.path.exists(fpath) or not os.path.isfile(fpath):
            print(f"Warning: File {fpath} does not exist or is not a file. Skipping.")
            skipped_count += 1
            continue

        try:
            if extension:
                df_file = extended_extract_clue_features(fpath)
            else:
                df_file = extract_clue_features(fpath)

            # Check if feature extraction succeeded
            if df_file is not None and not df_file.empty:
                dfs.append(df_file)
            else:
                print(
                    f"Warning: Extraction returned empty/None for {os.path.basename(fpath)}. Skipping."
                )
                skipped_count += 1

        except Exception as e:
            print(
                f"Warning: Failed to extract features from {os.path.basename(fpath)}. Error: {e}"
            )
            skipped_count += 1

    if not dfs:
        print("\n[ERROR] No valid data extracted from provided files.")
        return pd.DataFrame()

    total_events = sum(len(d) for d in dfs)
    print(
        f"--> Successfully processed {len(dfs)} file(s) ({total_events} total events). "
        f"(Skipped {skipped_count} missing/corrupted/empty files)"
    )

    return pd.concat(dfs, ignore_index=True)

def extended_extract_clue_features(
    input_root_path: str, side_radius_mm: float = 15.0
) -> Optional[pd.DataFrame]:
    """Reads raw CLUE TTrees (CLUEClustersHits, CLUEClusters, MCParticles) from a ROOT file

    and calculates advanced shower shape, layer profile, and substructure
    features for photon vs. pi0 BDT classification.
    """
    print(f"--> Opening ROOT file: {input_root_path}")

    try:
        with uproot.open(input_root_path) as f:
            required_trees = ["CLUEClustersHits", "CLUEClusters", "MCParticles"]
            for tree_name in required_trees:
                if tree_name not in f:
                    print(
                        f"Warning: Tree '{tree_name}' missing in {input_root_path}. Skipping."
                    )
                    return None

            if f["CLUEClustersHits"].num_entries == 0:
                print(
                    f"Warning: File {input_root_path} contains 0 events. Skipping."
                )
                return None

            cluster_hits = f["CLUEClustersHits"].arrays(
                ["x", "y", "z", "layer", "energy"]
            )
            clusters = f["CLUEClusters"].arrays(
                ["totEnergy", "totSize", "maxLayer", "clusters"]
            )
            mc = f["MCParticles"].arrays(
                ["pdg", "energy", "primary", ]
            )

        print("--> Computing fine-grained calorimeter features...")

        # -------------------------------------------------------------------
        # A. KINEMATICS & MULTIPLICITY
        # -------------------------------------------------------------------
        tot_hit_e = ak.sum(cluster_hits["energy"], axis=1)
        n_hits = ak.num(cluster_hits["energy"], axis=1)
        n_clusters = clusters["clusters"].array()
        print(n_clusters)
        safe_tot_e = ak.where(tot_hit_e > 0, tot_hit_e, 1.0)

        sorted_cluster_e = ak.sort(
            clusters["totEnergy"], axis=1, ascending=False
        )
        leading_cluster_e = ak.fill_none(ak.firsts(sorted_cluster_e), 0.0)
        leading_cluster_ratio = leading_cluster_e / safe_tot_e

        # -------------------------------------------------------------------
        # B. HIT-LEVEL SUBSTRUCTURE & PEAK RATIOS
        # -------------------------------------------------------------------
        sorted_hit_e = ak.sort(
            cluster_hits["energy"], axis=1, ascending=False
        )
        e_max1 = ak.fill_none(ak.firsts(sorted_hit_e), 0.0)
        e_max2 = ak.fill_none(ak.firsts(sorted_hit_e[:, 1:]), 0.0)
        e_min = ak.fill_none(ak.min(cluster_hits["energy"], axis=1), 0.0)

        ratio_e_max_2ndmax = ak.where(e_max2 > 0, e_max1 / e_max2, e_max1)
        delta_e_2ndmax_min = ak.where(e_max2 - e_min > 0, e_max2 - e_min, 0.0)
        max_hit_energy_ratio = e_max1 / safe_tot_e

        # -------------------------------------------------------------------
        # C. TRANSVERSE SHOWER SHAPE & SIDE FRACTION
        # -------------------------------------------------------------------
        x_mean = (
            ak.sum(cluster_hits["x"] * cluster_hits["energy"], axis=1)
            / safe_tot_e
        )
        y_mean = (
            ak.sum(cluster_hits["y"] * cluster_hits["energy"], axis=1)
            / safe_tot_e
        )

        r_sq = (cluster_hits["x"] - x_mean) ** 2 + (
            cluster_hits["y"] - y_mean
        ) ** 2
        radius_var = (
            ak.sum(r_sq * cluster_hits["energy"], axis=1) / safe_tot_e
        )
        sigma_r = np.sqrt(ak.where(radius_var > 0, radius_var, 0.0))

        x_var = (
            ak.sum(
                ((cluster_hits["x"] - x_mean) ** 2) * cluster_hits["energy"],
                axis=1,
            )
            / safe_tot_e
        )
        y_var = (
            ak.sum(
                ((cluster_hits["y"] - y_mean) ** 2) * cluster_hits["energy"],
                axis=1,
            )
            / safe_tot_e
        )
        width_x = np.sqrt(ak.where(x_var > 0, x_var, 0.0))
        width_y = np.sqrt(ak.where(y_var > 0, y_var, 0.0))

        r_hits_unflat = np.sqrt(r_sq)
        side_hits_energy = ak.where(
            r_hits_unflat > side_radius_mm, cluster_hits["energy"], 0.0
        )
        e_fr_side = ak.sum(side_hits_energy, axis=1) / safe_tot_e

        # -------------------------------------------------------------------
        # D. LONGITUDINAL PROFILE
        # -------------------------------------------------------------------
        weighted_layer = (
            ak.sum(cluster_hits["layer"] * cluster_hits["energy"], axis=1)
            / safe_tot_e
        )
        layer_var = (
            ak.sum(
                ((cluster_hits["layer"] - weighted_layer) ** 2)
                * cluster_hits["energy"],
                axis=1,
            )
            / safe_tot_e
        )
        sigma_layer = np.sqrt(ak.where(layer_var > 0, layer_var, 0.0))

        # -------------------------------------------------------------------
        # E. TARGET LABELS & MC TRUTH
        # -------------------------------------------------------------------
        primary_pdgs = ak.firsts(mc["pdg"][mc["primary"]])
        primary_energies = ak.firsts(mc["energy"][mc["primary"]])

        primary_pdgs = ak.fill_none(
            primary_pdgs, ak.fill_none(ak.firsts(mc["pdg"]), 0)
        )
        primary_energies = ak.fill_none(
            primary_energies, ak.fill_none(ak.firsts(mc["energy"]), 0.0)
        )

        true_pdg = safe_to_numpy(primary_pdgs, dtype=np.int64, default_val=0)
        true_e = safe_to_numpy(
            primary_energies, dtype=np.float64, default_val=0.0
        )
        binary_label = np.where(true_pdg == 22, 1, 0)

        # -------------------------------------------------------------------
        # F. DATAFRAME CONSTRUCTION
        # -------------------------------------------------------------------
        df = pd.DataFrame(
            {
                "n_hits": safe_to_numpy(
                    n_hits, dtype=np.int64, default_val=0
                ),
                "n_clusters": safe_to_numpy(
                    n_clusters, dtype=np.int64, default_val=0
                ),
                "total_reco_energy": safe_to_numpy(tot_hit_e),
                "leading_cluster_energy": safe_to_numpy(leading_cluster_e),
                "leading_cluster_ratio": safe_to_numpy(leading_cluster_ratio),
                "max_hit_energy_ratio": safe_to_numpy(max_hit_energy_ratio),
                "ratio_e_max_2ndmax": safe_to_numpy(ratio_e_max_2ndmax),
                "delta_e_2ndmax_min": safe_to_numpy(delta_e_2ndmax_min),
                "shower_radius": safe_to_numpy(sigma_r),
                "width_x": safe_to_numpy(width_x),
                "width_y": safe_to_numpy(width_y),
                "e_fr_side": safe_to_numpy(e_fr_side),
                "mean_layer": safe_to_numpy(weighted_layer),
                "sigma_layer": safe_to_numpy(sigma_layer),
                "target_pdg": true_pdg,
                "target_energy": true_e,
                "label": binary_label,
            }
        )

        # Final sanitization
        feature_cols = [
            c
            for c in df.columns
            if c not in ["target_pdg", "target_energy", "label"]
        ]
        df[feature_cols] = (
            df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

        return df

    except Exception as e:
        print(
            f"Warning: Could not process {input_root_path}. Error: {e}. Skipping."
        )
        return None

def extract_clue_features(input_root_path: str) -> Optional[pd.DataFrame]:
    """Reads raw CLUE TTrees (CLUEClustersHits, CLUEClusters, MCParticles) from a ROOT file

    and calculates tabular shower shape features for BDT training.

    Returns None if file is corrupted, empty, or missing required trees.

    Legacy version of feature extraction for backward compatibility with older datasets. Not recommended for new datasets.
    """
    print(f"--> Opening ROOT file: {input_root_path}")

    try:
        with uproot.open(input_root_path) as f:
            # 1. Verify required TTrees exist (using CLUEClustersHits instead of CLUEHits)
            required_trees = ["CLUEClustersHits", "CLUEClusters", "MCParticles"]
            for tree_name in required_trees:
                if tree_name not in f:
                    print(
                        f"Warning: Tree '{tree_name}' missing in {input_root_path}. Skipping."
                    )
                    return None

            # 2. Check for empty tree
            if f["CLUEClustersHits"].num_entries == 0:
                print(
                    f"Warning: File {input_root_path} contains 0 events. Skipping."
                )
                return None

            # Load hits (excluding rho, phi, delta, eta), clusters, and MC truth
            hits = f["CLUEClustersHits"].arrays(
                ["x", "y", "z", "layer", "energy"]
            )
            clusters = f["CLUEClusters"].arrays(
                ["totEnergy", "totSize", "maxLayer"]
            )
            mc = f["MCParticles"].arrays(["pdg", "energy", "primary"])

        print("--> Computing shower shape features...")

        # 1. Total Energy & Multiplicity
        tot_hit_e = ak.sum(hits["energy"], axis=1)
        n_hits = ak.num(hits["energy"])
        n_clusters = ak.num(clusters["totEnergy"])

        # Safe division array to avoid divide-by-zero errors
        safe_tot_e = ak.where(tot_hit_e > 0, tot_hit_e, 1.0)

        # 2. Longitudinal Profile (Layer-wise)
        weighted_layer = (
            ak.sum(hits["layer"] * hits["energy"], axis=1) / safe_tot_e
        )
        layer_var = (
            ak.sum(
                ((hits["layer"] - weighted_layer) ** 2) * hits["energy"], axis=1
            )
            / safe_tot_e
        )
        # Protect against tiny negative values due to float precision before sqrt
        sigma_layer = np.sqrt(np.maximum(0.0, ak.to_numpy(layer_var)))

        # 3. Transverse Profile (Shower Radius)
        x_mean = ak.sum(hits["x"] * hits["energy"], axis=1) / safe_tot_e
        y_mean = ak.sum(hits["y"] * hits["energy"], axis=1) / safe_tot_e

        r_sq = (hits["x"] - x_mean) ** 2 + (hits["y"] - y_mean) ** 2
        radius_var = ak.sum(r_sq * hits["energy"], axis=1) / safe_tot_e
        sigma_r = np.sqrt(np.maximum(0.0, ak.to_numpy(radius_var)))

        # 4. Cluster Summaries (Fill 0 for events without clusters)
        max_cluster_e = ak.max(
            clusters["totEnergy"], axis=1, mask_identity=False
        )
        leading_cluster_ratio = ak.fill_none(max_cluster_e / safe_tot_e, 0.0)

        # 5. Target Labels & Class Assignments from MC Truth
        primary_mask = mc["primary"]
        true_pdg = ak.flatten(mc["pdg"][primary_mask]).to_numpy()
        true_e = ak.flatten(mc["energy"][primary_mask]).to_numpy()

        # Assign binary label: 1 for Gamma (22), 0 for Pi0 (111)
        binary_label = np.where(true_pdg == 22, 1, 0)

        # Construct DataFrame
        df = pd.DataFrame(
            {
                "n_hits": ak.to_numpy(n_hits),
                "n_clusters": ak.to_numpy(n_clusters),
                "total_reco_energy": ak.to_numpy(tot_hit_e),
                "avg_hit_energy": ak.to_numpy(
                    tot_hit_e / ak.where(n_hits > 0, n_hits, 1)
                ),
                "mean_layer": ak.to_numpy(weighted_layer),
                "sigma_layer": sigma_layer,
                "shower_radius": sigma_r,
                "leading_cluster_ratio": ak.to_numpy(leading_cluster_ratio),
                # Target Metadata & Classification Label
                "target_pdg": true_pdg,
                "target_energy": true_e,
                "label": binary_label,
            }
        )

        # 6. Final pass: sanitize any residual Inf / NaN values
        feature_cols = [
            c
            for c in df.columns
            if c not in ["target_pdg", "target_energy", "label"]
        ]
        df[feature_cols] = (
            df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

        return df

    except Exception as e:
        print(
            f"Warning: Could not process {input_root_path}. Error: {e}. Skipping."
        )
        return None


def process_mix_and_split(
    signal_files,
    background_files,
    base_output_dir,
    chunk_size=50000,
    train_size=0.6,
    val_size=0.2,
    test_size=0.2,
    random_seed=42,
):
    """Extracts features from signal & background, enforces 50/50 balance,

    performs 3-way stratified splitting, and writes chunked ROOT datasets.
    """
    print("\n--- Extracting Features from Signal Files ---")
    df_sig = load_and_extract_multiple_files(signal_files)

    print("\n--- Extracting Features from Background Files ---")
    df_bkg = load_and_extract_multiple_files(background_files)

    # 1. 50/50 Subsampling Balance
    n_sig, n_bkg = len(df_sig), len(df_bkg)
    n_samples = min(n_sig, n_bkg)

    print(
        f"\nRaw Event Counts -> Signal (Gamma): {n_sig} | Background (Pi0): {n_bkg}\n"
        f"Enforcing 50/50 balance with {n_samples} events per class (Total: {2 * n_samples})"
    )

    df_sig_sampled = df_sig.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)
    df_bkg_sampled = df_bkg.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)

    df_mixed = pd.concat([df_sig_sampled, df_bkg_sampled], ignore_index=True)
    df_mixed = shuffle(df_mixed, random_state=random_seed).reset_index(drop=True)

    # 2. Stratified 3-way Split
    total_ratio = train_size + val_size + test_size
    p_train = train_size / total_ratio
    p_val = val_size / total_ratio
    p_test = test_size / total_ratio

    df_train, df_rem = train_test_split(
        df_mixed, train_size=p_train, stratify=df_mixed["label"], random_state=random_seed
    )

    val_rel_ratio = p_val / (p_val + p_test)
    df_val, df_test = train_test_split(
        df_rem, train_size=val_rel_ratio, stratify=df_rem["label"], random_state=random_seed
    )

    print(
        f"Splits Created -> Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}"
    )

    # 3. Export to Chunked ROOT Files
    splits = [
        ("Training_100GeV", df_train, os.path.join(base_output_dir, "Training"), "clue_train"),
        ("Validation_100GeV", df_val, os.path.join(base_output_dir, "Validation"), "clue_val"),
        ("Testing_100GeV", df_test, os.path.join(base_output_dir, "Testing"), "clue_test"),
    ]

    for name, df_split, out_dir, prefix in splits:
        os.makedirs(out_dir, exist_ok=True)
        num_chunks = int(np.ceil(len(df_split) / chunk_size))

        print(f"Writing {len(df_split)} events to '{out_dir}' in {num_chunks} chunk(s)...")
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(df_split))
            chunk_df = df_split.iloc[start:end]

            out_path = os.path.join(out_dir, f"{prefix}_chunk_{i}.root")
            with uproot.recreate(out_path) as out_f:
                out_f[TREE_NAME] = chunk_df

    print("\nData preparation, mixing, and chunked export complete!")



def run_bdt_preprocessing(signal_files: list, background_files: list, output_dir: str, seed: int = 42, chunk_size: int = 50000):
    print(f"\n[BDT Preprocessing] Processing {len(signal_files)} signal & {len(background_files)} bkg files...")

    process_mix_and_split(
            signal_files=signal_files,
            background_files=background_files,
            base_output_dir=output_dir,
            chunk_size=chunk_size,
            random_seed=seed
        )


    print("[BDT Preprocessing] Done!")