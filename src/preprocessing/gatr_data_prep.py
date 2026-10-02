"""Data preparation pipeline for GATr calorimeter shower classification

and CLUE diphoton mass reconstruction evaluation.
"""

import glob
import os
import shutil
import tempfile

import awkward as ak
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from collections import Counter
import uproot

from src.preprocessing.hit_readout import merge_four_layer_hits_to_one_plus_three

MAIN_KEYS = [
    "n_hits",
    "hit_x_norm",
    "hit_y_norm",
    "hit_z_norm",
    "hit_E_raw",
    "hit_E_norm",
    "hit_layer",
    "cluster_totEnergy",
    "cluster_totSize",
    "hit_mask",
    "label",
    "true_mc_energy",
]

EXTRA_KEYS = [
    "hit_x",
    "hit_y",
    "hit_z",
    "hit_cluster_id",
]



def print_cutflow_table(cutflow: Counter, label: str):
    """Prints a clean cutflow summary table."""
    print(f"\n================ Cutflow Statistics: {label} ================")
    print(f"{'Cut Step':<35} | {'Events Saved':<12} | {'Efficiency':<10}")
    print("-" * 65)

    total_events = cutflow.get("0_Total_Events", 1)
    for cut_name in sorted(cutflow.keys()):
        count = cutflow[cut_name]
        eff = (count / total_events) * 100 if total_events > 0 else 0.0
        print(f"{cut_name:<35} | {count:<12} | {eff:.2f}%")
    print("=" * 65)




    


def clue_physics_preselection(
    tree_dict: dict,
    cutflow: Counter,
    r_min: float = 2200.0,
    r_max: float = 2645.0,
    max_energy_ratio: float = 0.95,
) -> ak.Array:
    """Applies vertex fiducial volume cuts and pi0 asymmetry vetoes safely."""
    mc = tree_dict["MCParticles"]
    n_events = len(mc["pdg"])
    cutflow["0_Total_Events"] += n_events

    if n_events == 0:
        return ak.Array([])

    pdg = mc["pdg"]
    
    energy = mc["energy"]

    mc_keys = mc.fields
    x_key = next((k for k in ["x", "vertex_x", "endpoint_x"] if k in mc_keys), "x")
    y_key = next((k for k in ["y", "vertex_y", "endpoint_y"] if k in mc_keys), "y")

    ##--Debugging: Print the keys found in the MCParticles tree
    print(f"MCParticles keys found: {mc_keys}")

    mc_x = mc[x_key]
    mc_y = mc[y_key]

    # ---------------------------------------------------------
    # FIX: Pad arrays to length 3 so [:, 1] and [:, 2] never crash
    # ---------------------------------------------------------
    pdg_pad = ak.pad_none(pdg, 3, axis=1)
    #print(f"PDG Codes Sample (padded): {pdg_pad[:5]}")  # Debugging output
    energy_pad = ak.pad_none(energy, 3, axis=1)
    mc_x_pad = ak.pad_none(mc_x, 5, axis=1)
    mc_y_pad = ak.pad_none(mc_y, 5, axis=1)
    #print(f"x coordinate arrays{mc_x_pad[:5]}")
    #print(f"y coordinate arrays {mc_y_pad[:5]}")
    # Primary particle identification (fill None with 0 to avoid boolean errors)
    primary_pdg = ak.fill_none(pdg_pad[:, 0], 0)
    #print(f"Primary PDG Codes Sample: {primary_pdg[:5]}")  # Debugging output
    is_photon_event = primary_pdg == 22
    is_pion_event = primary_pdg == 111
    if primary_pdg[0] == 22:
        print("Event Type: Photon (gamma)")
        # --- 1. Photon Events Selection ---
        x0 = ak.fill_none(mc_x_pad[:, 1], 0.0)
        y0 = ak.fill_none(mc_y_pad[:, 1], 0.0)
        #x1 = ak.fill_none(mc_x_pad[:, 1], 0.0)
        #y1 = ak.fill_none(mc_y_pad[:, 1], 0.0)
        #x2 = ak.fill_none(mc_x_pad[:, 2], 0.0)
        #y2 = ak.fill_none(mc_y_pad[:, 2], 0.0)
        #print(f"Photon Vertex Coordinates (x0, y0) Sample: {x0[:5]}, {y0[:5]}")
        #print(f"Photon Secondary Coordinates (x1, y1, x2, y2) Sample: {x1[:5]}, {y1[:5]}, {x2[:5]}, {y2[:5]}")


        r_photon = np.sqrt(x0**2 + y0**2)
        #if len(r_photon) > 0:
        #    print(f"Photon Radius Range: {ak.min(r_photon):.2f} - {ak.max(r_photon):.2f} (r_min={r_min}, r_max={r_max})")
        #else:
        #    print("Photon Radius Range: Empty Array")
        pass_photon_r = is_photon_event & (r_photon >= r_min) & (r_photon < r_max)

        final_mask = pass_photon_r

    elif primary_pdg[0] == 111:
        print("Event Type: Neutral Pion (pi0)")
        # --- 1. Pion Events Selection ---
        x0 = ak.fill_none(mc_x_pad[:, 0], 0.0)
        y0 = ak.fill_none(mc_y_pad[:, 0], 0.0)
        #x1 = ak.fill_none(mc_x_pad[:, 1], 0.0)
        #y1 = ak.fill_none(mc_y_pad[:, 1], 0.0)
        #x2 = ak.fill_none(mc_x_pad[:, 2], 0.0)
        #y2 = ak.fill_none(mc_y_pad[:, 2], 0.0)
        x1 = ak.fill_none(mc_x_pad[:, 3], 0.0)
        y1 = ak.fill_none(mc_y_pad[:, 3], 0.0)
        x2 = ak.fill_none(mc_x_pad[:, 4], 0.0)
        y2 = ak.fill_none(mc_y_pad[:, 4], 0.0)
        
        #print(f"Pion Gamma Vertex Coordinates (x1, y1, x2, y2) Sample: {x1[:5]}, {y1[:5]}, {x2[:5]}, {y2[:5]}")

        r_g1 = np.sqrt(x1**2 + y1**2)
        r_g2 = np.sqrt(x2**2 + y2**2)
        #if len(r_g1) > 0 and len(r_g2) > 0:
        #    print(f"Pion Gamma Radii Ranges: g1({ak.min(r_g1):.2f} - {ak.max(r_g1):.2f}), g2({ak.min(r_g2):.2f} - {ak.max(r_g2):.2f}) (r_min={r_min}, r_max={r_max})")
        #else:
        #    print("Pion Gamma Radii Ranges: Empty Arrays")
        pass_pion_r = (
            is_pion_event
            & (r_g1 >= r_min)
            & (r_g1 < r_max)
            & (r_g2 >= r_min)
            & (r_g2 < r_max)
        )



        # Energy Asymmetry Veto
        e_pi0 = ak.fill_none(energy_pad[:, 0], 1.0)  # Default to 1.0 to avoid div by zero
        e_g1 = ak.fill_none(energy_pad[:, 1], 0.0)
        e_g2 = ak.fill_none(energy_pad[:, 2], 0.0)

        pass_pion_asymmetry = (
            is_pion_event
            & (e_g1 < max_energy_ratio * e_pi0)
            & (e_g2 < max_energy_ratio * e_pi0)
        )

        pass_pion_all = pass_pion_r & pass_pion_asymmetry
        final_mask = pass_pion_all



    # --- Log Cutflow Details ---
    pion_events_count = int(ak.sum(is_pion_event))
    photon_events_count = int(ak.sum(is_photon_event))

    if photon_events_count > 0:
        cutflow["1_Photon_Pass_R_Fiducial"] += int(ak.sum(pass_photon_r))

    if pion_events_count > 0:
        cutflow["1_Pion_Pass_Both_Gamma_R_Fiducial"] += int(ak.sum(pass_pion_r))
        cutflow["2_Pion_Pass_Energy_Asymmetry_Veto"] += int(ak.sum(pass_pion_asymmetry))
        cutflow["3_Pion_Pass_Combined_Cuts"] += int(ak.sum(pass_pion_all))

    cutflow["4_Final_Selected_Events"] += int(ak.sum(final_mask))

    return final_mask


def preprocess_clue_root_file(
    input_root_paths: list[str],
    output_root_path: str,
    stats: dict = None,
    max_hits: int = 256,
    extra_trees_to_load: dict[str, list[str]] = None,
    preselection_fn=None,
    cutflow_logger: Counter = None,
    merge_back_layers: bool = False,
    ):
    """Reads raw ROOT files, applies multi-tree preselections, and extracts features."""
    if cutflow_logger is None:
        cutflow_logger = Counter()

    all_hits_x, all_hits_y, all_hits_z, all_hits_E, all_hits_layer, all_hits_cid = (
        [], [], [], [], [], []
    )
    all_clusters_totE, all_clusters_totSize = [], []
    all_mc_pdg, all_mc_primary, all_mc_energy = [], [], []

    valid_files = 0
    for path in input_root_paths:
        try:
            with uproot.open(path) as f:
                required_trees = ["CLUEClustersHits", "CLUEClusters", "MCParticles"]
                if extra_trees_to_load:
                    required_trees.extend(extra_trees_to_load.keys())

                if not all(tree in f for tree in set(required_trees)):
                    print(f" Missing required trees in {path}. Skipping.")
                    continue

                hits_tree = f["CLUEClustersHits"]
                hit_keys = hits_tree.keys()
                cid_name = next(
                    (k for k in ["hit_cluster_id", "cluster_id", "clusterId"] if k in hit_keys),
                    "cluster",
                )

                # Load standard trees
                hits = hits_tree.arrays(["x", "y", "z", "layer", "energy", cid_name])
                clusters = f["CLUEClusters"].arrays(["totEnergy", "totSize"])
                mc_keys = f["MCParticles"].keys()
                mc_branches = ["pdg", "energy", "primary"]
                for coord in ["x", "y", "vertex_x", "vertex_y", "endpoint_x", "endpoint_y"]:
                    if coord in mc_keys:
                        mc_branches.append(coord)

                mc = f["MCParticles"].arrays(mc_branches)

                # Assemble tree dict for preselection logic
                tree_dict = {
                    "CLUEClustersHits": hits,
                    "CLUEClusters": clusters,
                    "MCParticles": mc,
                }

                # Load any additional requested trees/branches
                if extra_trees_to_load:
                    for t_name, b_names in extra_trees_to_load.items():
                        tree_dict[t_name] = f[t_name].arrays(b_names)

                # Apply Preselection Cuts
                if preselection_fn is not None:
                    event_mask = preselection_fn(tree_dict, cutflow_logger)
                    hits = hits[event_mask]
                    clusters = clusters[event_mask]
                    mc = mc[event_mask]

                if len(mc["pdg"]) == 0:
                    continue

                if merge_back_layers:
                    hits = merge_four_layer_hits_to_one_plus_three(hits)

                all_hits_x.append(hits["x"])
                all_hits_y.append(hits["y"])
                all_hits_z.append(hits["z"])
                all_hits_E.append(hits["energy"])
                all_hits_layer.append(hits["layer"])
                all_hits_cid.append(hits[cid_name])

                all_clusters_totE.append(clusters["totEnergy"])
                all_clusters_totSize.append(clusters["totSize"])

                all_mc_pdg.append(mc["pdg"])
                all_mc_primary.append(mc["primary"])
                all_mc_energy.append(mc["energy"])
                valid_files += 1

        except Exception as e:
            print(f" Skipping unreadable input file {path}: {e}")
            continue

    if valid_files == 0 or len(all_mc_pdg) == 0:
        print(f" No valid events remaining for destination: {output_root_path}")
        return

    # Concatenate Awkward arrays
    hits_x = ak.concatenate(all_hits_x)
    hits_y = ak.concatenate(all_hits_y)
    hits_z = ak.concatenate(all_hits_z)
    hits_E = ak.concatenate(all_hits_E)
    hits_layer = ak.concatenate(all_hits_layer)
    hits_cid = ak.concatenate(all_hits_cid)

    clusters_totE = ak.concatenate(all_clusters_totE)
    clusters_totSize = ak.concatenate(all_clusters_totSize)

    mc_pdg = ak.concatenate(all_mc_pdg)
    mc_primary = ak.concatenate(all_mc_primary)
    mc_energy = ak.concatenate(all_mc_energy)

    # Calculate Normalization Statistics if not provided
    if stats is None:
        flat_x = ak.to_numpy(ak.flatten(hits_x)).astype(np.float32)
        flat_y = ak.to_numpy(ak.flatten(hits_y)).astype(np.float32)
        flat_z = ak.to_numpy(ak.flatten(hits_z)).astype(np.float32)
        flat_E = np.log1p(ak.to_numpy(ak.flatten(hits_E)).astype(np.float32))

        def calc_stat(arr):
            arr_64 = arr.astype(np.float64)
            m = float(np.mean(arr_64))
            s = float(np.std(arr_64))
            return [m, s if s > 1e-8 else 1.0]

        stats = {
            "x": calc_stat(flat_x),
            "y": calc_stat(flat_y),
            "z": calc_stat(flat_z),
            "log_E": calc_stat(flat_E),
        }

    def safe_norm(val, mean, std):
        val_clean = np.nan_to_num(val, nan=0.0, posinf=1e5, neginf=-1.0)
        return (val_clean - mean) / (std if std > 1e-8 else 1.0)

    processed_main = {k: [] for k in MAIN_KEYS}
    processed_extra = {k: [] for k in EXTRA_KEYS}

    num_events = len(mc_pdg)

    for i in range(num_events):
        ev_mc_pdg = ak.to_numpy(mc_pdg[i])
        ev_mc_primary = ak.to_numpy(mc_primary[i])
        ev_mc_energy = ak.to_numpy(mc_energy[i])

        primary_mask = (ev_mc_primary == 1) | (ev_mc_primary == True)
        primary_pdgs = ev_mc_pdg[primary_mask]
        primary_energies = ev_mc_energy[primary_mask]

        if len(primary_pdgs) == 0:
            continue

        primary_pdg = primary_pdgs[0]
        true_energy = float(primary_energies[0])

        if primary_pdg == 22:
            label = 0  # Photon (gamma)
        elif primary_pdg == 111:
            label = 1  # Neutral Pion (pi0)
        else:
            continue

        hx = ak.to_numpy(hits_x[i]).astype(np.float32)
        hy = ak.to_numpy(hits_y[i]).astype(np.float32)
        hz = ak.to_numpy(hits_z[i]).astype(np.float32)
        hE = ak.to_numpy(hits_E[i]).astype(np.float32)
        hlayer = ak.to_numpy(hits_layer[i]).astype(np.int32)
        hcid = ak.to_numpy(hits_cid[i]).astype(np.int32)

        num_hits = len(hx)
        if num_hits == 0:
            continue

        c_totE = (
            float(np.sum(clusters_totE[i]))
            if len(clusters_totE[i]) > 0
            else float(np.sum(hE))
        )
        c_totSize = (
            int(np.sum(clusters_totSize[i]))
            if len(clusters_totSize[i]) > 0
            else num_hits
        )

        x_norm = safe_norm(hx, stats["x"][0], stats["x"][1])
        y_norm = safe_norm(hy, stats["y"][0], stats["y"][1])
        z_norm = safe_norm(hz, stats["z"][0], stats["z"][1])
        E_norm = safe_norm(np.log1p(hE), stats["log_E"][0], stats["log_E"][1])

        if num_hits > max_hits:
            top_idx = np.argsort(hE)[::-1][:max_hits]
            x_norm, y_norm, z_norm = (
                x_norm[top_idx],
                y_norm[top_idx],
                z_norm[top_idx],
            )
            hE, E_norm = hE[top_idx], E_norm[top_idx]
            hlayer = hlayer[top_idx]
            hx, hy, hz = hx[top_idx], hy[top_idx], hz[top_idx]
            hcid = hcid[top_idx]
            actual_hits = max_hits
        else:
            actual_hits = num_hits

        # Buffers for ML Training Tree (CLUEShowers)
        buf_x_norm = np.zeros(max_hits, dtype=np.float32)
        buf_y_norm = np.zeros(max_hits, dtype=np.float32)
        buf_z_norm = np.zeros(max_hits, dtype=np.float32)
        buf_E_raw = np.zeros(max_hits, dtype=np.float32)
        buf_E_norm = np.zeros(max_hits, dtype=np.float32)
        buf_layer = np.zeros(max_hits, dtype=np.int32)
        buf_mask = np.zeros(max_hits, dtype=np.bool_)

        buf_x_norm[:actual_hits] = x_norm
        buf_y_norm[:actual_hits] = y_norm
        buf_z_norm[:actual_hits] = z_norm
        buf_E_raw[:actual_hits] = hE
        buf_E_norm[:actual_hits] = E_norm
        buf_layer[:actual_hits] = hlayer
        buf_mask[:actual_hits] = True

        processed_main["n_hits"].append(actual_hits)
        processed_main["hit_x_norm"].append(buf_x_norm)
        processed_main["hit_y_norm"].append(buf_y_norm)
        processed_main["hit_z_norm"].append(buf_z_norm)
        processed_main["hit_E_raw"].append(buf_E_raw)
        processed_main["hit_E_norm"].append(buf_E_norm)
        processed_main["hit_layer"].append(buf_layer)
        processed_main["cluster_totEnergy"].append(c_totE)
        processed_main["cluster_totSize"].append(c_totSize)
        processed_main["hit_mask"].append(buf_mask)
        processed_main["label"].append(label)
        processed_main["true_mc_energy"].append(true_energy)

        # Buffers for Evaluation Tree (CLUEExtra)
        buf_x_phys = np.zeros(max_hits, dtype=np.float32)
        buf_y_phys = np.zeros(max_hits, dtype=np.float32)
        buf_z_phys = np.zeros(max_hits, dtype=np.float32)
        buf_cluster_id = np.full(max_hits, -1, dtype=np.int32)  # -1 for padded hits

        buf_x_phys[:actual_hits] = hx
        buf_y_phys[:actual_hits] = hy
        buf_z_phys[:actual_hits] = hz
        buf_cluster_id[:actual_hits] = hcid

        processed_extra["hit_x"].append(buf_x_phys)
        processed_extra["hit_y"].append(buf_y_phys)
        processed_extra["hit_z"].append(buf_z_phys)
        processed_extra["hit_cluster_id"].append(buf_cluster_id)

    if len(processed_main["label"]) == 0:
        print(f"Zero valid events processed for {output_root_path}. Skipping write.")
        return

    # Write to local temp disk before moving to EOS
    tmp_dir = os.environ.get("TMPDIR", "/tmp")
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".root", dir=tmp_dir)
    os.close(tmp_fd)

    try:
        with uproot.recreate(tmp_path) as out_f:
            out_f["CLUEShowers"] = {
                "n_hits": np.array(processed_main["n_hits"], dtype=np.int32),
                "hit_x_norm": np.array(processed_main["hit_x_norm"], dtype=np.float32),
                "hit_y_norm": np.array(processed_main["hit_y_norm"], dtype=np.float32),
                "hit_z_norm": np.array(processed_main["hit_z_norm"], dtype=np.float32),
                "hit_E_raw": np.array(processed_main["hit_E_raw"], dtype=np.float32),
                "hit_E_norm": np.array(processed_main["hit_E_norm"], dtype=np.float32),
                "hit_layer": np.array(processed_main["hit_layer"], dtype=np.int32),
                "cluster_totEnergy": np.array(processed_main["cluster_totEnergy"], dtype=np.float32),
                "cluster_totSize": np.array(processed_main["cluster_totSize"], dtype=np.int32),
                "hit_mask": np.array(processed_main["hit_mask"], dtype=np.bool_),
                "label": np.array(processed_main["label"], dtype=np.int32),
                "true_mc_energy": np.array(processed_main["true_mc_energy"], dtype=np.float32),
            }

            out_f["CLUEExtra"] = {
                "hit_x": np.array(processed_extra["hit_x"], dtype=np.float32),
                "hit_y": np.array(processed_extra["hit_y"], dtype=np.float32),
                "hit_z": np.array(processed_extra["hit_z"], dtype=np.float32),
                "hit_cluster_id": np.array(processed_extra["hit_cluster_id"], dtype=np.int32),
            }

        with uproot.open(tmp_path) as test_f:
            _ = test_f["CLUEShowers"]["hit_x_norm"].array(entry_stop=1)
            _ = test_f["CLUEExtra"]["hit_x"].array(entry_stop=1)

        os.makedirs(os.path.dirname(output_root_path), exist_ok=True)
        shutil.move(tmp_path, output_root_path)
        print(
            f"Saved {len(processed_main['label'])} events across {valid_files} file(s) -> {output_root_path}"
        )

    except Exception as e:
        print(f" Failed write/transfer to {output_root_path}: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_clue_data_from_files(file_list: list[str]) -> dict:
    """Loads feature arrays from 'CLUEShowers' and 'CLUEExtra' trees into a single dict."""
    all_keys = MAIN_KEYS + EXTRA_KEYS
    combined_dict = {k: [] for k in all_keys}

    for fpath in file_list:
        try:
            with uproot.open(fpath) as f:
                if "CLUEShowers" not in f or "CLUEExtra" not in f:
                    continue

                tree_main = f["CLUEShowers"]
                tree_extra = f["CLUEExtra"]

                file_data = {k: tree_main[k].array(library="np") for k in MAIN_KEYS}
                file_data.update({k: tree_extra[k].array(library="np") for k in EXTRA_KEYS})

            for k in all_keys:
                combined_dict[k].append(file_data[k])

        except Exception as e:
            print(f"Skipping corrupted file {fpath}: {e}")

    for k in all_keys:
        if combined_dict[k]:
            combined_dict[k] = np.concatenate(combined_dict[k], axis=0)

    return combined_dict


def process_mix_and_split(
    signal_files: list[str],
    background_files: list[str],
    base_output_dir: str,
    tree_name: str = "CLUEShowers",
    extra_tree_name: str = "CLUEExtra",
    chunk_size: int = 50000,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_seed: int = 42,
):
    """Enforces 50/50 class balance, performs 3-way stratified splitting,

    and exports aligned 'CLUEShowers' and 'CLUEExtra' trees to ROOT chunks.
    """
    print("\n--- Loading Signal Files (Pi0) ---")
    data_sig = load_clue_data_from_files(signal_files)

    print("\n--- Loading Background Files (Gamma) ---")
    data_bkg = load_clue_data_from_files(background_files)

    n_sig = len(data_sig["label"])
    n_bkg = len(data_bkg["label"])
    n_samples = min(n_sig, n_bkg)

    print(
        f"\nRaw Event Counts -> Signal (Pi0): {n_sig} | Background (Gamma): {n_bkg}\n"
        f"Enforcing 50/50 balance with {n_samples} events per class (Total: {2 * n_samples})"
    )

    np.random.seed(random_seed)
    idx_sig_sampled = np.random.choice(n_sig, size=n_samples, replace=False)
    idx_bkg_sampled = np.random.choice(n_bkg, size=n_samples, replace=False)

    all_keys = MAIN_KEYS + EXTRA_KEYS

    data_mixed = {}
    for k in all_keys:
        sig_part = data_sig[k][idx_sig_sampled]
        bkg_part = data_bkg[k][idx_bkg_sampled]
        data_mixed[k] = np.concatenate([sig_part, bkg_part], axis=0)

    n_total = len(data_mixed["label"])
    shuffle_indices = np.arange(n_total)
    np.random.seed(random_seed)
    np.random.shuffle(shuffle_indices)

    for k in all_keys:
        data_mixed[k] = data_mixed[k][shuffle_indices]

    total_ratio = train_size + val_size + test_size
    p_train = train_size / total_ratio
    p_val = val_size / total_ratio
    p_test = test_size / total_ratio

    indices = np.arange(n_total)
    labels = data_mixed["label"]

    train_idx, rem_idx = train_test_split(
        indices,
        train_size=p_train,
        stratify=labels,
        random_state=random_seed,
    )

    val_rel_ratio = p_val / (p_val + p_test)
    val_idx, test_idx = train_test_split(
        rem_idx,
        train_size=val_rel_ratio,
        stratify=labels[rem_idx],
        random_state=random_seed,
    )

    print(
        f"Splits Created -> Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}"
    )

    def slice_dict(dict_data, idxs):
        return {k: dict_data[k][idxs] for k in dict_data.keys()}

    splits = [
        (
            "Training",
            slice_dict(data_mixed, train_idx),
            os.path.join(base_output_dir, "Training"),
            "clue_train_gatr",
        ),
        (
            "Validation",
            slice_dict(data_mixed, val_idx),
            os.path.join(base_output_dir, "Validation"),
            "clue_val_gatr",
        ),
        (
            "Testing",
            slice_dict(data_mixed, test_idx),
            os.path.join(base_output_dir, "Testing"),
            "clue_test_gatr",
        ),
    ]

    for name, split_dict, out_dir, prefix in splits:
        os.makedirs(out_dir, exist_ok=True)
        num_events_split = len(split_dict["label"])
        num_chunks = int(np.ceil(num_events_split / chunk_size))

        print(
            f"Writing {num_events_split} events to '{out_dir}' in {num_chunks} chunk(s)..."
        )
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, num_events_split)

            main_chunk = {k: split_dict[k][start:end] for k in MAIN_KEYS}
            extra_chunk = {k: split_dict[k][start:end] for k in EXTRA_KEYS}

            out_path = os.path.join(out_dir, f"{prefix}_chunk_{i}.root")

            with uproot.recreate(out_path) as out_f:
                out_f[tree_name] = main_chunk
                out_f[extra_tree_name] = extra_chunk

    print("\nData preparation, mixing, and chunked export complete!")


def run_gatr_preprocessing(
    signal_files: list[str],
    background_files: list[str],
    output_dir: str,
    seed: int = 42,
    chunk_size: int = 5000,
    files_per_chunk: int = 25,
    version_tag: str = "v1",
    preselection_fn=clue_physics_preselection,  # Pass preselection function here
    extra_trees: dict = None,
    merge_back_layers: bool = False,
):
    """Executes GATr data pipeline with preselection filtering and cutflow logging."""
    sig_cutflow = Counter()
    bkg_cutflow = Counter()

    processed_dir = os.path.join(output_dir, "intermediate_processed")
    split_dir = os.path.join(output_dir, "split")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    def _batch_and_preprocess(file_list: list[str], class_label: str, logger: Counter) -> list[str]:
        if not file_list:
            return []
        chunks = [file_list[i : i + files_per_chunk] for i in range(0, len(file_list), files_per_chunk)]
        out_paths = [
            os.path.join(processed_dir, f"preprocessed_{class_label}_chunk_{idx}.root")
            for idx in range(len(chunks))
        ]

        for chunk_files, out_file in zip(chunks, out_paths):
            preprocess_clue_root_file(
                chunk_files,
                out_file,
                preselection_fn=preselection_fn,
                extra_trees_to_load=extra_trees,
                cutflow_logger=logger,
                merge_back_layers=merge_back_layers,
            )
        return out_paths

    proc_sig_files = _batch_and_preprocess(signal_files, "pion_sig", sig_cutflow)
    proc_bkg_files = _batch_and_preprocess(background_files, "photon_bkg", bkg_cutflow)

    # Print summary tables
    print_cutflow_table(sig_cutflow, "Signal (Pion)")
    print_cutflow_table(bkg_cutflow, "Background (Photon)")

    print(f"\n---> [Stage 2/2] Running 50/50 balance and 3-way split...")

    process_mix_and_split(
        signal_files=proc_sig_files,
        background_files=proc_bkg_files,
        base_output_dir=split_dir,
        tree_name="CLUEShowers",
        extra_tree_name="CLUEExtra",
        chunk_size=chunk_size,
        random_seed=seed,
    )

    print(f"\n[GATr Preprocessing] Complete! Splitted datasets saved to: {split_dir}")