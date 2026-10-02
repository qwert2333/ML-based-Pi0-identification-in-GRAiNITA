from collections import Counter
import glob
import os
from typing import Optional

import awkward as ak
import numpy as np
import pandas as pd
import uproot
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from configs.BDT_config import TREE_NAME
from src.split_utils import safe_to_numpy
from src.preprocessing.hit_readout import (
    merge_all_layer_hits_to_one,
    merge_four_layer_hits_to_one_plus_three,
)


def print_cutflow_table(cutflow: Counter, label: str):
    """Prints a clean cutflow summary table."""
    print(f"\n================ Cutflow Statistics: {label} ================")
    print(f"{'Cut Step':<38} | {'Events Saved':<12} | {'Efficiency':<10}")
    print("-" * 68)

    total_events = cutflow.get("0_Total_Events", 1)
    for cut_name in sorted(cutflow.keys()):
        count = cutflow[cut_name]
        eff = (count / total_events) * 100 if total_events > 0 else 0.0
        print(f"{cut_name:<38} | {count:<12} | {eff:.2f}%")
    print("=" * 68)


def clue_physics_preselection(
    tree_dict: dict,
    cutflow: Optional[Counter] = None,
    r_min: float = 2200.0,
    r_max: float = 2645.0,
    max_energy_ratio: float = 0.95,
) -> ak.Array:
    """Applies vertex fiducial volume cuts and pi0 asymmetry vetoes safely."""
    mc = tree_dict["MCParticles"]
    n_events = len(mc["pdg"])

    if cutflow is not None:
        cutflow["0_Total_Events"] += n_events

    if n_events == 0:
        return ak.Array([])

    pdg = mc["pdg"]
    energy = mc["energy"]

    mc_keys = mc.fields
    x_key = next((k for k in ["x", "vertex_x", "endpoint_x"] if k in mc_keys), "x")
    y_key = next((k for k in ["y", "vertex_y", "endpoint_y"] if k in mc_keys), "y")

    mc_x = mc[x_key]
    mc_y = mc[y_key]

    pdg_pad = ak.pad_none(pdg, 3, axis=1)
    energy_pad = ak.pad_none(energy, 3, axis=1)
    mc_x_pad = ak.pad_none(mc_x, 5, axis=1)
    mc_y_pad = ak.pad_none(mc_y, 5, axis=1)

    primary_pdg = ak.fill_none(pdg_pad[:, 0], 0)
    is_photon_event = primary_pdg == 22
    is_pion_event = primary_pdg == 111

    final_mask = ak.zeros_like(primary_pdg, dtype=bool)

    # 1. Photon Selection
    if ak.any(is_photon_event):
        x0 = ak.fill_none(mc_x_pad[:, 1], 0.0)
        y0 = ak.fill_none(mc_y_pad[:, 1], 0.0)
        r_photon = np.sqrt(x0**2 + y0**2)
        pass_photon_r = is_photon_event & (r_photon >= r_min) & (r_photon < r_max)

        final_mask = ak.where(is_photon_event, pass_photon_r, final_mask)

        if cutflow is not None:
            cutflow["1_Photon_Pass_R_Fiducial"] += int(ak.sum(pass_photon_r))

    # 2. Pion Selection
    if ak.any(is_pion_event):
        x1 = ak.fill_none(mc_x_pad[:, 3], 0.0)
        y1 = ak.fill_none(mc_y_pad[:, 3], 0.0)
        x2 = ak.fill_none(mc_x_pad[:, 4], 0.0)
        y2 = ak.fill_none(mc_y_pad[:, 4], 0.0)

        r_g1 = np.sqrt(x1**2 + y1**2)
        r_g2 = np.sqrt(x2**2 + y2**2)
        pass_pion_r = (
            is_pion_event
            & (r_g1 >= r_min)
            & (r_g1 < r_max)
            & (r_g2 >= r_min)
            & (r_g2 < r_max)
        )

        e_pi0 = ak.fill_none(energy_pad[:, 0], 1.0)
        e_g1 = ak.fill_none(energy_pad[:, 1], 0.0)
        e_g2 = ak.fill_none(energy_pad[:, 2], 0.0)

        pass_pion_asymmetry = (
            is_pion_event
            & (e_g1 < max_energy_ratio * e_pi0)
            & (e_g2 < max_energy_ratio * e_pi0)
        )

        pass_pion_all = pass_pion_r & pass_pion_asymmetry
        final_mask = ak.where(is_pion_event, pass_pion_all, final_mask)

        if cutflow is not None:
            cutflow["1_Pion_Pass_Both_Gamma_R_Fiducial"] += int(ak.sum(pass_pion_r))
            cutflow["2_Pion_Pass_Energy_Asymmetry_Veto"] += int(ak.sum(pass_pion_asymmetry))
            cutflow["3_Pion_Pass_Combined_Cuts"] += int(ak.sum(pass_pion_all))

    if cutflow is not None:
        cutflow["4_Final_Selected_Events"] += int(ak.sum(final_mask))

    return final_mask


def compute_diphoton_invariant_mass(
    cluster_hits: ak.Array,
    clusters: ak.Array,
    vertex_pos: tuple = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Computes diphoton invariant mass (m_inv) for each event in awkward arrays."""
    n_events = len(cluster_hits)
    m_inv_arr = np.zeros(n_events, dtype=np.float32)

    has_cid = "cluster_id" in cluster_hits.fields or "clusterId" in cluster_hits.fields
    cid_key = "cluster_id" if "cluster_id" in cluster_hits.fields else "clusterId"

    for i in range(n_events):
        x = ak.to_numpy(cluster_hits["x"][i])
        y = ak.to_numpy(cluster_hits["y"][i])
        z = ak.to_numpy(cluster_hits["z"][i])
        e = ak.to_numpy(cluster_hits["energy"][i])

        if has_cid:
            cids = ak.to_numpy(cluster_hits[cid_key][i])
            valid = (e > 0) & (cids >= 0)
        else:
            valid = e > 0
            cids = np.zeros_like(e, dtype=int)

        if not np.any(valid):
            continue

        valid_cids = cids[valid]
        unique_cids = np.unique(valid_cids)
        if len(unique_cids) < 2:
            continue

        c_energies = []
        c_positions = []
        x_v, y_v, z_v, e_v = x[valid], y[valid], z[valid], e[valid]

        for cid in unique_cids:
            c_mask = valid_cids == cid
            e_tot = np.sum(e_v[c_mask])
            if e_tot <= 0:
                continue
            pos = np.average(
                np.column_stack([x_v[c_mask], y_v[c_mask], z_v[c_mask]]),
                axis=0,
                weights=e_v[c_mask],
            )
            c_energies.append(e_tot)
            c_positions.append(pos)

        if len(c_energies) < 2:
            continue

        sort_idx = np.argsort(c_energies)[::-1]
        e1, r1 = c_energies[sort_idx[0]], c_positions[sort_idx[0]]
        e2, r2 = c_energies[sort_idx[1]], c_positions[sort_idx[1]]

        v1 = r1 - np.array(vertex_pos)
        v2 = r2 - np.array(vertex_pos)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            continue

        cos_theta = np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)
        m_inv_arr[i] = np.sqrt(max(0.0, 2.0 * e1 * e2 * (1.0 - cos_theta)))

    return m_inv_arr


def extended_extract_clue_features(
    input_root_path: str,
    side_radius_mm: float = 15.0,
    apply_preselection: bool = True,
    cutflow: Optional[Counter] = None,
    merge_back_layers: bool = False,
    merge_all_layers: bool = False,
) -> Optional[pd.DataFrame]:
    """Reads raw CLUE TTrees, applies preselection, and computes features + m_inv."""
    print(f"--> Opening ROOT file: {input_root_path}")

    try:
        with uproot.open(input_root_path) as f:
            required_trees = ["CLUEClustersHits", "CLUEClusters", "MCParticles"]
            for tree_name in required_trees:
                if tree_name not in f:
                    print(f"Warning: Tree '{tree_name}' missing in {input_root_path}. Skipping.")
                    return None

            if f["CLUEClustersHits"].num_entries == 0:
                print(f"Warning: File {input_root_path} contains 0 events. Skipping.")
                return None

            hit_keys = f["CLUEClustersHits"].keys()
            hit_branches = ["x", "y", "z", "layer", "energy"]
            for cid_name in ["cluster_id", "clusterId", "clusterIndex"]:
                if cid_name in hit_keys:
                    hit_branches.append(cid_name)

            cluster_hits = f["CLUEClustersHits"].arrays(hit_branches)
            clusters = f["CLUEClusters"].arrays(["totEnergy", "totSize", "maxLayer", "clusters"])

            mc_keys = f["MCParticles"].keys()
            mc_branches = ["pdg", "energy", "primary"]
            for coord in ["x", "y", "vertex_x", "vertex_y", "endpoint_x", "endpoint_y"]:
                if coord in mc_keys:
                    mc_branches.append(coord)

            mc = f["MCParticles"].arrays(mc_branches)

        # Apply Physics Preselection
        if apply_preselection:
            tree_dict = {
                "CLUEClustersHits": cluster_hits,
                "CLUEClusters": clusters,
                "MCParticles": mc,
            }
            event_mask = clue_physics_preselection(tree_dict, cutflow=cutflow)
            if len(event_mask) == 0 or ak.sum(event_mask) == 0:
                print(f"Warning: 0 events passed preselection in {input_root_path}. Skipping.")
                return None

            cluster_hits = cluster_hits[event_mask]
            clusters = clusters[event_mask]
            mc = mc[event_mask]

        if merge_back_layers and merge_all_layers:
            raise ValueError("merge_back_layers and merge_all_layers are exclusive")
        if merge_back_layers:
            cluster_hits = merge_four_layer_hits_to_one_plus_three(cluster_hits)
        elif merge_all_layers:
            cluster_hits = merge_all_layer_hits_to_one(cluster_hits)

        print("--> Computing fine-grained calorimeter features & diphoton m_inv...")

        # Calculate Diphoton Invariant Mass (m_inv)
        m_inv = compute_diphoton_invariant_mass(cluster_hits, clusters)

        # A. KINEMATICS & MULTIPLICITY
        tot_hit_e = ak.sum(cluster_hits["energy"], axis=1)
        n_hits = ak.num(cluster_hits["energy"], axis=1)

         
        n_clusters = ak.flatten(clusters["clusters"])

        safe_tot_e = ak.where(tot_hit_e > 0, tot_hit_e, 1.0)

        sorted_cluster_e = ak.sort(clusters["totEnergy"], axis=1, ascending=False)
        leading_cluster_e = ak.fill_none(ak.firsts(sorted_cluster_e), 0.0)
        leading_cluster_ratio = leading_cluster_e / safe_tot_e

        # B. HIT-LEVEL SUBSTRUCTURE & PEAK RATIOS
        sorted_hit_e = ak.sort(cluster_hits["energy"], axis=1, ascending=False)
        e_max1 = ak.fill_none(ak.firsts(sorted_hit_e), 0.0)
        e_max2 = ak.fill_none(ak.firsts(sorted_hit_e[:, 1:]), 0.0)
        e_min = ak.fill_none(ak.min(cluster_hits["energy"], axis=1), 0.0)

        ratio_e_max_2ndmax = ak.where(e_max2 > 0, e_max1 / e_max2, e_max1)
        delta_e_2ndmax_min = ak.where(e_max2 - e_min > 0, e_max2 - e_min, 0.0)
        max_hit_energy_ratio = e_max1 / safe_tot_e

        # C. TRANSVERSE SHOWER SHAPE & SIDE FRACTION
        x_mean = ak.sum(cluster_hits["x"] * cluster_hits["energy"], axis=1) / safe_tot_e
        y_mean = ak.sum(cluster_hits["y"] * cluster_hits["energy"], axis=1) / safe_tot_e

        r_sq = (cluster_hits["x"] - x_mean) ** 2 + (cluster_hits["y"] - y_mean) ** 2
        radius_var = ak.sum(r_sq * cluster_hits["energy"], axis=1) / safe_tot_e
        sigma_r = np.sqrt(ak.where(radius_var > 0, radius_var, 0.0))

        x_var = ak.sum(((cluster_hits["x"] - x_mean) ** 2) * cluster_hits["energy"], axis=1) / safe_tot_e
        y_var = ak.sum(((cluster_hits["y"] - y_mean) ** 2) * cluster_hits["energy"], axis=1) / safe_tot_e
        width_x = np.sqrt(ak.where(x_var > 0, x_var, 0.0))
        width_y = np.sqrt(ak.where(y_var > 0, y_var, 0.0))

        r_hits_unflat = np.sqrt(r_sq)
        side_hits_energy = ak.where(r_hits_unflat > side_radius_mm, cluster_hits["energy"], 0.0)
        e_fr_side = ak.sum(side_hits_energy, axis=1) / safe_tot_e

        # D. LONGITUDINAL PROFILE
        weighted_layer = ak.sum(cluster_hits["layer"] * cluster_hits["energy"], axis=1) / safe_tot_e
        layer_var = ak.sum(((cluster_hits["layer"] - weighted_layer) ** 2) * cluster_hits["energy"], axis=1) / safe_tot_e
        sigma_layer = np.sqrt(ak.where(layer_var > 0, layer_var, 0.0))

        # E. TARGET LABELS & MC TRUTH
        primary_pdgs = ak.firsts(mc["pdg"][mc["primary"]])
        primary_energies = ak.firsts(mc["energy"][mc["primary"]])

        primary_pdgs = ak.fill_none(primary_pdgs, ak.fill_none(ak.firsts(mc["pdg"]), 0))
        primary_energies = ak.fill_none(primary_energies, ak.fill_none(ak.firsts(mc["energy"]), 0.0))

        true_pdg = safe_to_numpy(primary_pdgs, dtype=np.int64, default_val=0)
        true_e = safe_to_numpy(primary_energies, dtype=np.float64, default_val=0.0)
        binary_label = np.where(true_pdg == 22, 1, 0)

        # F. DATAFRAME CONSTRUCTION (Including m_inv and true_mc_energy)
        df = pd.DataFrame(
            {
                "n_hits": safe_to_numpy(n_hits, dtype=np.int64, default_val=0),
                "n_clusters": safe_to_numpy(n_clusters, dtype=np.int64, default_val=0),
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
                "m_inv": m_inv,
                "true_mc_energy": true_e,
                "target_pdg": true_pdg,
                "target_energy": true_e,
                "label": binary_label,
            }
        )

        non_feature_cols = ["target_pdg", "target_energy", "true_mc_energy", "m_inv", "label"]
        feature_cols = [c for c in df.columns if c not in non_feature_cols]
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return df

    except Exception as e:
        print(f"Warning: Could not process {input_root_path}. Error: {e}. Skipping.")
        return None


def extract_clue_features(
    input_root_path: str,
    apply_preselection: bool = True,
    cutflow: Optional[Counter] = None,
    merge_back_layers: bool = False,
    merge_all_layers: bool = False,
) -> Optional[pd.DataFrame]:
    """Base tabular feature extraction including fiducial preselection filtering."""
    print(f"--> Opening ROOT file: {input_root_path}")

    try:
        with uproot.open(input_root_path) as f:
            required_trees = ["CLUEClustersHits", "CLUEClusters", "MCParticles"]
            for tree_name in required_trees:
                if tree_name not in f:
                    print(f"Warning: Tree '{tree_name}' missing in {input_root_path}. Skipping.")
                    return None

            if f["CLUEClustersHits"].num_entries == 0:
                print(f"Warning: File {input_root_path} contains 0 events. Skipping.")
                return None

            hits = f["CLUEClustersHits"].arrays(["x", "y", "z", "layer", "energy"])
            clusters = f["CLUEClusters"].arrays(["totEnergy", "totSize", "maxLayer", "clusters"])

            mc_keys = f["MCParticles"].keys()
            mc_branches = ["pdg", "energy", "primary"]
            for coord in ["x", "y", "vertex_x", "vertex_y", "endpoint_x", "endpoint_y"]:
                if coord in mc_keys:
                    mc_branches.append(coord)

            mc = f["MCParticles"].arrays(mc_branches)

        if apply_preselection:
            tree_dict = {
                "CLUEClustersHits": hits,
                "CLUEClusters": clusters,
                "MCParticles": mc,
            }
            event_mask = clue_physics_preselection(tree_dict, cutflow=cutflow)
            if len(event_mask) == 0 or ak.sum(event_mask) == 0:
                print(f"Warning: 0 events passed preselection in {input_root_path}. Skipping.")
                return None

            hits = hits[event_mask]
            clusters = clusters[event_mask]
            mc = mc[event_mask]

        if merge_back_layers and merge_all_layers:
            raise ValueError("merge_back_layers and merge_all_layers are exclusive")
        if merge_back_layers:
            hits = merge_four_layer_hits_to_one_plus_three(hits)
        elif merge_all_layers:
            hits = merge_all_layer_hits_to_one(hits)

        print("--> Computing shower shape features...")

        tot_hit_e = ak.sum(hits["energy"], axis=1)
        n_hits = ak.num(hits["energy"])
        n_clusters = clusters["clusters"].array()
        print(n_clusters)
        safe_tot_e = ak.where(tot_hit_e > 0, tot_hit_e, 1.0)

        weighted_layer = ak.sum(hits["layer"] * hits["energy"], axis=1) / safe_tot_e
        layer_var = ak.sum(((hits["layer"] - weighted_layer) ** 2) * hits["energy"], axis=1) / safe_tot_e
        sigma_layer = np.sqrt(np.maximum(0.0, ak.to_numpy(layer_var)))

        x_mean = ak.sum(hits["x"] * hits["energy"], axis=1) / safe_tot_e
        y_mean = ak.sum(hits["y"] * hits["energy"], axis=1) / safe_tot_e

        r_sq = (hits["x"] - x_mean) ** 2 + (hits["y"] - y_mean) ** 2
        radius_var = ak.sum(r_sq * hits["energy"], axis=1) / safe_tot_e
        sigma_r = np.sqrt(np.maximum(0.0, ak.to_numpy(radius_var)))

        max_cluster_e = ak.max(clusters["totEnergy"], axis=1, mask_identity=False)
        leading_cluster_ratio = ak.fill_none(max_cluster_e / safe_tot_e, 0.0)

        primary_mask = mc["primary"]
        true_pdg = ak.flatten(mc["pdg"][primary_mask]).to_numpy()
        true_e = ak.flatten(mc["energy"][primary_mask]).to_numpy()

        binary_label = np.where(true_pdg == 22, 1, 0)

        df = pd.DataFrame(
            {
                "n_hits": ak.to_numpy(n_hits),
                "n_clusters": ak.to_numpy(n_clusters),
                "total_reco_energy": ak.to_numpy(tot_hit_e),
                "avg_hit_energy": ak.to_numpy(tot_hit_e / ak.where(n_hits > 0, n_hits, 1)),
                "mean_layer": ak.to_numpy(weighted_layer),
                "sigma_layer": sigma_layer,
                "shower_radius": sigma_r,
                "leading_cluster_ratio": ak.to_numpy(leading_cluster_ratio),
                "target_pdg": true_pdg,
                "target_energy": true_e,
                "label": binary_label,
            }
        )

        feature_cols = [c for c in df.columns if c not in ["target_pdg", "target_energy", "label"]]
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return df

    except Exception as e:
        print(f"Warning: Could not process {input_root_path}. Error: {e}. Skipping.")
        return None


def load_and_extract_multiple_files(
    file_paths,
    extension: bool = True,
    apply_preselection: bool = True,
    cutflow: Optional[Counter] = None,
    merge_back_layers: bool = False,
    merge_all_layers: bool = False,
) -> pd.DataFrame:
    """Utility to process multiple raw ROOT files with preselection and feature extraction."""
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    expanded_paths = []
    for p in file_paths:
        if not p:
            continue
        if os.path.isdir(p):
            matched = sorted(glob.glob(os.path.join(p, "**", "*.root"), recursive=True))
            if not matched:
                matched = sorted(glob.glob(os.path.join(p, "*.root")))
            expanded_paths.extend(matched)
        elif "*" in p or "?" in p:
            expanded_paths.extend(sorted(glob.glob(p)))
        else:
            expanded_paths.append(p)

    expanded_paths = list(dict.fromkeys(expanded_paths))

    if not expanded_paths:
        print(f"Warning: No valid file paths found matching: {file_paths}")
        return pd.DataFrame()

    dfs = []
    skipped_count = 0

    print(f"--> Found {len(expanded_paths)} file(s) to process.")

    for fpath in expanded_paths:
        if not os.path.exists(fpath) or not os.path.isfile(fpath):
            print(f"Warning: File {fpath} does not exist or is not a file. Skipping.")
            skipped_count += 1
            continue

        try:
            if extension:
                df_file = extended_extract_clue_features(
                    fpath,
                    apply_preselection=apply_preselection,
                    cutflow=cutflow,
                    merge_back_layers=merge_back_layers,
                    merge_all_layers=merge_all_layers,
                )
            else:
                df_file = extract_clue_features(
                    fpath,
                    apply_preselection=apply_preselection,
                    cutflow=cutflow,
                    merge_back_layers=merge_back_layers,
                    merge_all_layers=merge_all_layers,
                )

            if df_file is not None and not df_file.empty:
                dfs.append(df_file)
            else:
                print(f"Warning: Extraction returned empty/None for {os.path.basename(fpath)}. Skipping.")
                skipped_count += 1

        except Exception as e:
            print(f"Warning: Failed to extract features from {os.path.basename(fpath)}. Error: {e}")
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


def process_mix_and_split(
    signal_files: list,
    background_files: list,
    base_output_dir: str,
    chunk_size: int = 50000,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_seed: int = 42,
    apply_preselection: bool = True,
    merge_back_layers: bool = False,
    merge_all_layers: bool = False,
):
    """Applies preselection, enforces class balance, splits data, and exports ROOT chunks."""
    sig_cutflow = Counter()
    bkg_cutflow = Counter()

    print("\n--- Extracting Features from Signal Files (Gamma) ---")
    df_sig = load_and_extract_multiple_files(
        signal_files,
        apply_preselection=apply_preselection,
        cutflow=sig_cutflow,
        merge_back_layers=merge_back_layers,
        merge_all_layers=merge_all_layers,
    )

    print("\n--- Extracting Features from Background Files (Pi0) ---")
    df_bkg = load_and_extract_multiple_files(
        background_files,
        apply_preselection=apply_preselection,
        cutflow=bkg_cutflow,
        merge_back_layers=merge_back_layers,
        merge_all_layers=merge_all_layers,
    )

    if apply_preselection:
        print_cutflow_table(sig_cutflow, "Signal (Gamma)")
        print_cutflow_table(bkg_cutflow, "Background (Pi0)")

    n_sig, n_bkg = len(df_sig), len(df_bkg)
    n_samples = min(n_sig, n_bkg)

    if n_samples == 0:
        print("\n[ERROR] One or both dataset classes contain 0 events after filtering. Aborting split.")
        return

    print(
        f"\nFiltered Event Counts -> Signal (Gamma): {n_sig} | Background (Pi0): {n_bkg}\n"
        f"Enforcing 50/50 balance with {n_samples} events per class (Total: {2 * n_samples})"
    )

    df_sig_sampled = df_sig.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)
    df_bkg_sampled = df_bkg.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)

    df_mixed = pd.concat([df_sig_sampled, df_bkg_sampled], ignore_index=True)
    df_mixed = shuffle(df_mixed, random_state=random_seed).reset_index(drop=True)

    total_ratio = train_size + val_size + test_size
    p_train = train_size / total_ratio
    p_val = val_size / total_ratio
    p_test = test_size / total_ratio

    df_train, df_rem = train_test_split(
        df_mixed, train_size=p_train, stratify=df_mixed["label"], random_state=random_seed
    )
    df_train = df_train.reset_index(drop=True)

    val_rel_ratio = p_val / (p_val + p_test)
    df_val, df_test = train_test_split(
        df_rem, train_size=val_rel_ratio, stratify=df_rem["label"], random_state=random_seed
    )
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print(f"Splits Created -> Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

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

            # Filter out index columns and convert to clean NumPy dict for Uproot write
            cols_to_write = [
                c for c in chunk_df.columns if c not in ["index", "Unnamed: 0", "__index_level_0__"]
            ]
            chunk_dict = {col: safe_to_numpy(chunk_df[col]) for col in cols_to_write}

            out_path = os.path.join(out_dir, f"{prefix}_chunk_{i}.root")
            with uproot.recreate(out_path) as out_f:
                out_f[TREE_NAME] = chunk_dict

    print("\nData preparation, preselection, mixing, and chunked export complete!")


def run_bdt_preprocessing(
    signal_files: list,
    background_files: list,
    output_dir: str,
    seed: int = 42,
    chunk_size: int = 50000,
    apply_preselection: bool = True,
    merge_back_layers: bool = False,
    merge_all_layers: bool = False,
):
    print(f"\n[BDT Preprocessing] Processing {len(signal_files)} signal & {len(background_files)} bkg files...")

    process_mix_and_split(
        signal_files=signal_files,
        background_files=background_files,
        base_output_dir=output_dir,
        chunk_size=chunk_size,
        random_seed=seed,
        apply_preselection=apply_preselection,
        merge_back_layers=merge_back_layers,
        merge_all_layers=merge_all_layers,
    )

    print("[BDT Preprocessing] Done!")