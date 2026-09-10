"""Pipeline for evaluating calorimeter shower reconstruction efficiency,

model identification efficiency, and combined efficiency.

Targets:
    1 = pi0 (Signal)
    0 = photon / gamma (Background)
"""

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import uproot
import vector

vector.register_awkward()

from configs.GATr_config import DEBUG_CONFIG, SCALE_UP_CONFIG
import gatr.utils.einsum


def native_torch_einsum(equation, *operands):
    return torch.einsum(equation, *operands)


gatr.utils.einsum.gatr_einsum = native_torch_einsum
gatr.utils.einsum._gatr_einsum = native_torch_einsum


# =====================================================================
# 1. CLUE Diphoton Invariant Mass Reconstruction
# =====================================================================
def reconstruct_diphoton_invariant_mass(
    x_hits: np.ndarray,
    y_hits: np.ndarray,
    z_hits: np.ndarray,
    e_hits: np.ndarray,
    cluster_ids: np.ndarray,
    mask: np.ndarray,
    vertex_pos: tuple = (0.0, 0.0, 0.0),
) -> tuple[int, float]:
    """Calculates CLUE cluster count and diphoton m_yy if n_clusters >= 2."""
    valid = mask.astype(bool) & (e_hits > 0) & (cluster_ids >= 0)
    if not np.any(valid):
        return 0, 0.0

    valid_cids = cluster_ids[valid]
    unique_clusters = np.unique(valid_cids)
    n_clusters = int(len(unique_clusters))

    if n_clusters < 2:
        return n_clusters, 0.0

    x, y, z, E = x_hits[valid], y_hits[valid], z_hits[valid], e_hits[valid]

    cluster_energies = []
    cluster_positions = []

    for cid in unique_clusters:
        c_mask = valid_cids == cid
        E_tot = np.sum(E[c_mask])
        if E_tot <= 0:
            continue

        pos = np.average(
            np.column_stack([x[c_mask], y[c_mask], z[c_mask]]),
            axis=0,
            weights=E[c_mask],
        )
        cluster_energies.append(E_tot)
        cluster_positions.append(pos)

    if len(cluster_energies) < 2:
        return n_clusters, 0.0

    sorted_indices = np.argsort(cluster_energies)[::-1]
    E1, r1 = cluster_energies[sorted_indices[0]], cluster_positions[sorted_indices[0]]
    E2, r2 = cluster_energies[sorted_indices[1]], cluster_positions[sorted_indices[1]]

    v1 = r1 - np.array(vertex_pos)
    v2 = r2 - np.array(vertex_pos)

    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return n_clusters, 0.0

    u1 = v1 / norm1
    u2 = v2 / norm2

    cos_theta = np.clip(np.dot(u1, u2), -1.0, 1.0)
    m_inv = np.sqrt(2.0 * E1 * E2 * (1.0 - cos_theta))
    return n_clusters, float(m_inv)


# =====================================================================
# 2. Data Loading & Inference Loop
# =====================================================================
def load_and_reconstruct_test_data(
    test_dir: str,
    model_wrapper,
    tree_name: str = "CLUEShowers",
    extra_tree_name: str = "CLUEExtra",
    batch_size: int = 32,
) -> pd.DataFrame:
    """Loads ROOT chunks, runs GATr inference, and extracts cluster counts & m_yy."""
    root_files = sorted(glob.glob(os.path.join(test_dir, "*.root")))
    if not root_files:
        raise FileNotFoundError(f"No ROOT files found in: {test_dir}")

    all_true_energy, all_labels, all_scores = [], [], []
    all_n_clusters, all_m_inv = [], []

    print(f"--> Processing {len(root_files)} test ROOT file(s) from '{test_dir}'...")

    for fpath in root_files:
        filename = os.path.basename(fpath)
        with uproot.open(fpath) as f:
            if tree_name not in f or extra_tree_name not in f:
                print(f"--> [WARNING] Skipping {filename}: Missing trees.")
                continue

            tree = f[tree_name]
            extra_tree = f[extra_tree_name]

            x_norm = tree["hit_x_norm"].array(library="np")
            y_norm = tree["hit_y_norm"].array(library="np")
            z_norm = tree["hit_z_norm"].array(library="np")
            E_raw = tree["hit_E_raw"].array(library="np")
            E_norm = tree["hit_E_norm"].array(library="np")
            mask = tree["hit_mask"].array(library="np")
            labels = tree["label"].array(library="np")
            mc_energy = tree["true_mc_energy"].array(library="np")

            x_phys = extra_tree["hit_x"].array(library="np")
            y_phys = extra_tree["hit_y"].array(library="np")
            z_phys = extra_tree["hit_z"].array(library="np")

            if "hit_cluster_id" in extra_tree:
                cluster_ids = extra_tree["hit_cluster_id"].array(library="np")
            elif "cluster_id" in extra_tree:
                cluster_ids = extra_tree["cluster_id"].array(library="np")
            else:
                raise KeyError(f"No cluster ID branch found in '{extra_tree_name}'.")

            num_events = len(labels)
            valid_events = []
            valid_event_indices = []

            for idx in range(num_events):
                x_arr = x_norm[idx]
                mask_arr = mask[idx]
                e_raw_arr = E_raw[idx]

                if x_arr.size == 0 or mask_arr.size == 0:
                    num_valid_hits = 0
                else:
                    valid_hit_mask = (mask_arr > 0) & (e_raw_arr > 0)
                    num_valid_hits = np.sum(valid_hit_mask)

                if num_valid_hits == 0:
                    continue

                valid_event_indices.append(idx)
                t_norm = np.zeros_like(x_arr)
                extra_scalars = np.stack([E_norm[idx], t_norm], axis=-1)

                valid_events.append({
                    "x": torch.from_numpy(x_arr).float(),
                    "y": torch.from_numpy(y_norm[idx]).float(),
                    "z": torch.from_numpy(z_norm[idx]).float(),
                    "energy_raw": torch.from_numpy(e_raw_arr).float(),
                    "extra_scalars": torch.from_numpy(extra_scalars).float(),
                    "mask": torch.from_numpy(mask_arr).float(),
                })

            dataset_list = []
            for i in range(0, len(valid_events), batch_size):
                batch_items = valid_events[i : i + batch_size]
                dataset_list.append({
                    "x": torch.stack([b["x"] for b in batch_items]),
                    "y": torch.stack([b["y"] for b in batch_items]),
                    "z": torch.stack([b["z"] for b in batch_items]),
                    "energy_raw": torch.stack([b["energy_raw"] for b in batch_items]),
                    "extra_scalars": torch.stack([b["extra_scalars"] for b in batch_items]),
                    "mask": torch.stack([b["mask"] for b in batch_items]),
                })

            sig_scores = np.zeros(num_events, dtype=np.float32)

            if dataset_list:
                probs = model_wrapper.predict_proba(dataset_list)
                valid_scores = probs[:, 1]

                for valid_i, orig_idx in enumerate(valid_event_indices):
                    sig_scores[orig_idx] = valid_scores[valid_i]

            reco_results = [
                reconstruct_diphoton_invariant_mass(
                    x_hits=x_phys[i],
                    y_hits=y_phys[i],
                    z_hits=z_phys[i],
                    e_hits=E_raw[i],
                    cluster_ids=cluster_ids[i],
                    mask=mask[i],
                )
                for i in range(num_events)
            ]

            n_clusters_list = [res[0] for res in reco_results]
            m_inv_list = [res[1] for res in reco_results]

            all_true_energy.append(mc_energy)
            all_labels.append(labels)
            all_scores.append(sig_scores)
            all_n_clusters.append(np.array(n_clusters_list))
            all_m_inv.append(np.array(m_inv_list))

    return pd.DataFrame(
        {
            "true_energy": np.concatenate(all_true_energy),
            "label": np.concatenate(all_labels),
            "score": np.concatenate(all_scores),
            "n_clusters": np.concatenate(all_n_clusters),
            "m_inv": np.concatenate(all_m_inv),
        }
    )


# =====================================================================
# 3. Binned Multi-Stage Metrics Calculation
# =====================================================================
def compute_binned_metrics(
    df_pi0: pd.DataFrame,
    df_gamma: pd.DataFrame,
    energy_bins: np.ndarray,
    mass_window: tuple = (0.110, 0.160),
    gatr_threshold: float = 0.5,
):
    """Computes binned selection efficiencies across sequential pipeline stages."""
    bin_centers = 0.5 * (energy_bins[:-1] + energy_bins[1:])

    df_pi0["pass_reco"] = df_pi0["n_clusters"] > 0
    df_pi0["pass_gatr"] = df_pi0["pass_reco"] & (df_pi0["score"] >= gatr_threshold)
    df_pi0["pass_multi"] = df_pi0["n_clusters"] >= 2
    df_pi0["pass_mass"] = (
        df_pi0["pass_multi"]
        & (df_pi0["m_inv"] >= mass_window[0])
        & (df_pi0["m_inv"] <= mass_window[1])
    )
    df_pi0["pass_comb_or"] = df_pi0["pass_gatr"] | df_pi0["pass_mass"]

    abs_eff_reco, abs_eff_gatr, abs_eff_mass, abs_eff_comb_or = [], [], [], []

    def safe_pct(num, den):
        return (num / den * 100.0) if den > 0 else 0.0

    for i in range(len(energy_bins) - 1):
        e_low, e_high = energy_bins[i], energy_bins[i + 1]
        pi0_bin = df_pi0[
            (df_pi0["true_energy"] >= e_low) & (df_pi0["true_energy"] < e_high)
        ]

        n_tot = len(pi0_bin)
        n_reco = pi0_bin["pass_reco"].sum()
        n_gatr = pi0_bin["pass_gatr"].sum()
        n_mass = pi0_bin["pass_mass"].sum()
        n_comb_or = pi0_bin["pass_comb_or"].sum()

        abs_eff_reco.append(safe_pct(n_reco, n_tot))
        abs_eff_gatr.append(safe_pct(n_gatr, n_tot))
        abs_eff_mass.append(safe_pct(n_mass, n_tot))
        abs_eff_comb_or.append(safe_pct(n_comb_or, n_tot))

    metrics_abs = {
        "eff_reco": np.array(abs_eff_reco),
        "eff_gatr": np.array(abs_eff_gatr),
        "eff_mass": np.array(abs_eff_mass),
        "eff_comb_or": np.array(abs_eff_comb_or),
    }

    return bin_centers, metrics_abs


# =====================================================================
# 4. Decoupled HEP Plot Generators
# =====================================================================
def plot_clue_reconstruction_efficiency(
    bin_centers: np.ndarray,
    metrics_abs: dict,
    exp_text: str = "CLUE GATr-Eval",
    output_path: str = "clue_reco_efficiency.pdf",
):
    """Plot 1: Baseline CLUE pre-identification reconstruction efficiency."""
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=120)

    # Blue Dot: CLUE cluster reconstruction efficiency (N_c > 0)
    ax.plot(
        bin_centers,
        metrics_abs["eff_reco"],
        color="#1f77b4",
        marker="o",
        linestyle="None",
        linewidth=1.8,
        label=r"CLUE Cluster Reco ($N_c > 0$)",
        zorder=3,
    )

    # Red Diamond: Mass Window Cut (N_c >= 2 & m_yy in window)
    ax.plot(
        bin_centers,
        metrics_abs["eff_mass"],
        color="#d62728",
        marker="D",
        linestyle="None",
        linewidth=2.0,
        label=r"Mass Window Cut ($N_c \geq 2$)",
        zorder=4,
    )

    max_e = max(bin_centers) + 2.5 if len(bin_centers) > 0 else 80.0
    ax.set_xlim(0, max_e)
    ax.set_ylim(-2, 102)

    ax.set_xlabel("Energy [GeV]", fontsize=14, labelpad=6)
    ax.set_ylabel("Efficiency [%]", fontsize=14, labelpad=8)

    ax.tick_params(axis="both", which="major", direction="in", length=7, width=1.2, labelsize=12)
    ax.tick_params(axis="both", which="minor", direction="in", length=4, width=0.8)
    ax.minorticks_on()

    words = exp_text.split()
    first_word = words[0] if words else ""
    rest_words = " ".join(words[1:]) if len(words) > 1 else ""
    ax.text(
        0.04,
        0.95,
        rf"$\mathbf{{{first_word}}}$ {rest_words}",
        transform=ax.transAxes,
        fontsize=14,
        va="top",
    )

    ax.legend(
        loc="center right",
        bbox_to_anchor=(0.96, 0.55),
        frameon=True,
        facecolor="white",
        framealpha=0.85,
        fontsize=11,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"--> Saved CLUE Reconstruction plot to: {output_path}")
    plt.close()


def plot_combined_efficiency(
    bin_centers: np.ndarray,
    metrics_abs: dict,
    exp_text: str = "CLUE GATr-Eval",
    output_path: str = "model_combined_efficiency.pdf",
):
    """Plot 2: GATr Model ID, Mass Window Cut, and Combined Efficiency."""
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=120)

    # Orange Triangle: GATr Model ID
    ax.plot(
        bin_centers,
        metrics_abs["eff_gatr"],
        color="#ff7f0e",
        marker="^",
        linestyle="None",
        linewidth=1.8,
        label=r"GATr Model ID",
        zorder=3,
    )

    # Red Diamond: Mass Window Cut
    ax.plot(
        bin_centers,
        metrics_abs["eff_mass"],
        color="#d62728",
        marker="D",
        linestyle="None",
        linewidth=2.0,
        label=r"Mass Window Cut",
        zorder=4,
    )

    # Purple Cross: Combined Efficiency (GATr OR Mass Window)
    ax.plot(
        bin_centers,
        metrics_abs["eff_comb_or"],
        color="#9467bd",
        marker="P",
        linestyle="None",
        linewidth=2.2,
        label=r"Combined Eff (GATr or Mass)",
        zorder=5,
    )

    max_e = max(bin_centers) + 2.5 if len(bin_centers) > 0 else 80.0
    ax.set_xlim(0, max_e)
    ax.set_ylim(-2, 102)

    ax.set_xlabel("Energy [GeV]", fontsize=14, labelpad=6)
    ax.set_ylabel("Efficiency [%]", fontsize=14, labelpad=8)

    ax.tick_params(axis="both", which="major", direction="in", length=7, width=1.2, labelsize=12)
    ax.tick_params(axis="both", which="minor", direction="in", length=4, width=0.8)
    ax.minorticks_on()

    words = exp_text.split()
    first_word = words[0] if words else ""
    rest_words = " ".join(words[1:]) if len(words) > 1 else ""
    ax.text(
        0.04,
        0.95,
        rf"$\mathbf{{{first_word}}}$ {rest_words}",
        transform=ax.transAxes,
        fontsize=14,
        va="top",
    )

    ax.legend(
        loc="center right",
        bbox_to_anchor=(0.96, 0.55),
        frameon=True,
        facecolor="white",
        framealpha=0.85,
        fontsize=11,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"--> Saved Model & Combined Efficiency plot to: {output_path}")
    plt.close()


# =====================================================================
# 5. End-to-End Execution
# =====================================================================
def run_evaluation(
    test_dir: str = "./data/split/Testing",
    model_path: str = "./best_gatr_model.pt",
    output_dir: str = "./plots",
    tree_name: str = "CLUEShowers",
    extra_tree_name: str = "CLUEExtra",
    gatr_config: dict = None,
    mass_window: tuple = (0.110, 0.160),
    gatr_threshold: float = 0.5,
    num_bins: int = 20,
    exp_text: str = "CLUE GATr-Eval",
    device: torch.device = None,
) -> pd.DataFrame:
    from src.models.gatr_models import CalorimeterGATrClassifier, GATrWrapper

    print("\n==========================================")
    print("Starting End-to-End Evaluation Pipeline")
    print("==========================================\n")

    state_dict = None

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            if "config" in checkpoint:
                gatr_config = checkpoint["config"]
                print("--> Overriding config with saved checkpoint architecture.")
        else:
            state_dict = checkpoint

    if gatr_config is None:
        gatr_config = {
            "in_s_channels": 3,
            "out_s_channels": 16,
            "hidden_s_channels": 32,
            "in_mv_channels": 1,
            "out_mv_channels": 1,
            "hidden_mv_channels": 8,
            "num_blocks": 2,
        }

    raw_model = CalorimeterGATrClassifier(gatr_config)
    if state_dict is not None:
        raw_model.load_state_dict(state_dict)
        print(f"Successfully loaded trained weights from: {model_path}")

    model_wrapper = GATrWrapper(raw_model, device=device)

    df_results = load_and_reconstruct_test_data(
        test_dir=test_dir,
        model_wrapper=model_wrapper,
        tree_name=tree_name,
        extra_tree_name=extra_tree_name,
    )

    df_pi0 = df_results[df_results["label"] == 1].copy()
    df_gamma = df_results[df_results["label"] == 0].copy()

    energy_bins = np.linspace(
        df_results["true_energy"].min(), df_results["true_energy"].max(), num_bins
    )

    bin_centers, metrics_abs = compute_binned_metrics(
        df_pi0,
        df_gamma,
        energy_bins=energy_bins,
        mass_window=mass_window,
        gatr_threshold=gatr_threshold,
    )

    plot_clue_reconstruction_efficiency(
        bin_centers,
        metrics_abs,
        exp_text=exp_text,
        output_path=os.path.join(output_dir, "clue_reco_efficiency.pdf"),
    )

    plot_combined_efficiency(
        bin_centers,
        metrics_abs,
        exp_text=exp_text,
        output_path=os.path.join(output_dir, "model_combined_efficiency.pdf"),
    )

    print("\n--- DIAGNOSTIC SUMMARY ---")
    print(df_results.groupby("label")[["n_clusters", "m_inv", "score"]].describe())
    print("---------------------------\n")
    print("\n[Evaluation Complete] Pipeline execution finished successfully!\n")
    return df_results


if __name__ == "__main__":
    run_evaluation(
        test_dir="/eos/user/m/mgroning/ML_summer_project/CERN_summer_project/data/GATr/dataset_preselection_0.5-80GeV/split/Testing",
        model_path="/eos/user/m/mgroning/ML_summer_project/CERN_summer_project/trained_models/GATr/model_dataset_preselection_0.5-80GeV_v1/gatr_best_model.pt",
        output_dir="/eos/user/m/mgroning/ML_summer_project/CERN_summer_project/plots",
        tree_name="CLUEShowers",
        extra_tree_name="CLUEExtra",
        mass_window=(0.110, 0.160),
        gatr_config=DEBUG_CONFIG,
        gatr_threshold=0.5,
        num_bins=20,
    )