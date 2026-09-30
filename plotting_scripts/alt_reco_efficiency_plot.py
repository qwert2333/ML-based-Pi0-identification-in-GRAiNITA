"""Pipeline for evaluating calorimeter shower reconstruction using per-cluster

GATr classification and orthogonal diphoton mass reconstruction on photon-like clusters.

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

from configs.GATr_config import DEBUG_CONFIG
import gatr.utils.einsum


def native_torch_einsum(equation, *operands):
    return torch.einsum(equation, *operands)


gatr.utils.einsum.gatr_einsum = native_torch_einsum
gatr.utils.einsum._gatr_einsum = native_torch_einsum


# =====================================================================
# 1. Invariant Mass Reconstruction for Selected Clusters
# =====================================================================
def compute_mass_for_cids(
    x_phys: np.ndarray,
    y_phys: np.ndarray,
    z_phys: np.ndarray,
    e_hits: np.ndarray,
    cluster_ids: np.ndarray,
    valid_mask: np.ndarray,
    selected_cids: list,
    vertex_pos: tuple = (0.0, 0.0, 0.0),
) -> float:
    """Calculates invariant mass m_yy using only hits belonging to selected_cids."""
    if len(selected_cids) < 2:
        return 0.0

    cluster_energies = []
    cluster_positions = []

    for cid in selected_cids:
        c_mask = valid_mask & (cluster_ids == cid)
        E_tot = np.sum(e_hits[c_mask])
        if E_tot <= 0:
            continue

        pos = np.average(
            np.column_stack([x_phys[c_mask], y_phys[c_mask], z_phys[c_mask]]),
            axis=0,
            weights=e_hits[c_mask],
        )
        cluster_energies.append(E_tot)
        cluster_positions.append(pos)

    if len(cluster_energies) < 2:
        return 0.0

    sorted_indices = np.argsort(cluster_energies)[::-1]
    E1, r1 = cluster_energies[sorted_indices[0]], cluster_positions[sorted_indices[0]]
    E2, r2 = cluster_energies[sorted_indices[1]], cluster_positions[sorted_indices[1]]

    v1 = r1 - np.array(vertex_pos)
    v2 = r2 - np.array(vertex_pos)

    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0

    u1, u2 = v1 / norm1, v2 / norm2
    cos_theta = np.clip(np.dot(u1, u2), -1.0, 1.0)
    return float(np.sqrt(2.0 * E1 * E2 * (1.0 - cos_theta)))


# =====================================================================
# 2. Per-Cluster Inference & Data Loading Loop
# =====================================================================
def load_and_eval_cluster_gatr_data(
    test_dir: str,
    model_wrapper,
    tree_name: str = "CLUEShowers",
    extra_tree_name: str = "CLUEExtra",
    gatr_threshold: float = 0.5,
    mass_window: tuple = (0.110, 0.160),
    batch_size: int = 64,
) -> pd.DataFrame:
    """Evaluates GATr per-cluster and applies the orthogonal selection workflow."""
    root_files = sorted(glob.glob(os.path.join(test_dir, "*.root")))
    if not root_files:
        raise FileNotFoundError(f"No ROOT files found in: {test_dir}")

    records = []
    print(f"--> Processing {len(root_files)} ROOT file(s) from '{test_dir}'...")

    for fpath in root_files:
        filename = os.path.basename(fpath)
        with uproot.open(fpath) as f:
            if tree_name not in f or extra_tree_name not in f:
                print(f"--> [WARNING] Skipping {filename}: Missing trees.")
                continue

            tree, extra_tree = f[tree_name], f[extra_tree_name]

            E_raw = tree["hit_E_raw"].array(library="np")
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

            for event_idx in range(len(labels)):
                valid = (mask[event_idx] > 0) & (E_raw[event_idx] > 0) & (cluster_ids[event_idx] >= 0)
                if not np.any(valid):
                    records.append({
                        "true_energy": mc_energy[event_idx],
                        "label": labels[event_idx],
                        "n_clusters": 0,
                        "n_pi0_clusters": 0,
                        "n_gamma_clusters": 0,
                        "pass_gatr_cluster": False,
                        "pass_mass_rem": False,
                        "pass_combined": False,
                        "m_inv_rem": 0.0,
                    })
                    continue

                valid_cids = cluster_ids[event_idx][valid]
                unique_cids = np.unique(valid_cids)
                n_clusters = len(unique_cids)

                # Step 1: Prepare cluster batch for GATr
                cluster_batch = []
                for cid in unique_cids:
                    c_mask = valid & (cluster_ids[event_idx] == cid)
                    c_mask_float = c_mask.astype(np.float32)

                    cluster_batch.append({
                        "x": torch.from_numpy(x_phys[event_idx]).float(),
                        "y": torch.from_numpy(y_phys[event_idx]).float(),
                        "z": torch.from_numpy(z_phys[event_idx]).float(),
                        "energy_raw": torch.from_numpy(E_raw[event_idx] * c_mask_float).float(),
                        "mask": torch.from_numpy(c_mask_float).float(),
                    })

                # Run GATr per cluster
                batched_data = []
                for i in range(0, len(cluster_batch), batch_size):
                    b = cluster_batch[i : i + batch_size]
                    batched_data.append({
                        "x": torch.stack([item["x"] for item in b]),
                        "y": torch.stack([item["y"] for item in b]),
                        "z": torch.stack([item["z"] for item in b]),
                        "energy_raw": torch.stack([item["energy_raw"] for item in b]),
                        "mask": torch.stack([item["mask"] for item in b]),
                    })

                probs = model_wrapper.predict_proba(batched_data)[:, 1]

                pi0_cids = [cid for cid, score in zip(unique_cids, probs) if score >= gatr_threshold]
                gamma_cids = [cid for cid, score in zip(unique_cids, probs) if score < gatr_threshold]

                n_pi0, n_gamma = len(pi0_cids), len(gamma_cids)

                pass_gatr_cluster = False
                pass_mass_rem = False
                m_inv_rem = 0.0

                # Step 2: Decision Tree Execution
                if (n_clusters == 1 and n_pi0 == 1) or (n_clusters > 0 and n_pi0 == n_clusters):
                    pass_gatr_cluster = True
                elif n_gamma >= 2:
                    m_inv_rem = compute_mass_for_cids(
                        x_phys=x_phys[event_idx],
                        y_phys=y_phys[event_idx],
                        z_phys=z_phys[event_idx],
                        e_hits=E_raw[event_idx],
                        cluster_ids=cluster_ids[event_idx],
                        valid_mask=valid,
                        selected_cids=gamma_cids,
                    )
                    if mass_window[0] <= m_inv_rem <= mass_window[1]:
                        pass_mass_rem = True

                records.append({
                    "true_energy": mc_energy[event_idx],
                    "label": labels[event_idx],
                    "n_clusters": n_clusters,
                    "n_pi0_clusters": n_pi0,
                    "n_gamma_clusters": n_gamma,
                    "pass_gatr_cluster": pass_gatr_cluster,
                    "pass_mass_rem": pass_mass_rem,
                    "pass_combined": pass_gatr_cluster or pass_mass_rem,
                    "m_inv_rem": m_inv_rem,
                })

    return pd.DataFrame(records)


# =====================================================================
# 3. Additive Binned Efficiency & Mis-ID Calculation
# =====================================================================
def compute_binned_additive_metrics(
    df_pi0: pd.DataFrame,
    df_gamma: pd.DataFrame,
    energy_bins: np.ndarray,
):
    """Computes strictly additive efficiencies for the orthogonal decision paths."""
    bin_centers = 0.5 * (energy_bins[:-1] + energy_bins[1:])

    eff_gatr_cluster, eff_mass_rem, eff_combined = [], [], []
    misid_gamma = []

    def safe_pct(num, den):
        return (num / den * 100.0) if den > 0 else 0.0

    for i in range(len(energy_bins) - 1):
        e_low, e_high = energy_bins[i], energy_bins[i + 1]

        pi0_bin = df_pi0[(df_pi0["true_energy"] >= e_low) & (df_pi0["true_energy"] < e_high)]
        gamma_bin = df_gamma[(df_gamma["true_energy"] >= e_low) & (df_gamma["true_energy"] < e_high)]

        n_tot = len(pi0_bin)
        n_gatr = pi0_bin["pass_gatr_cluster"].sum()
        n_mass = pi0_bin["pass_mass_rem"].sum()
        n_comb = pi0_bin["pass_combined"].sum()

        eff_gatr_cluster.append(safe_pct(n_gatr, n_tot))
        eff_mass_rem.append(safe_pct(n_mass, n_tot))
        eff_combined.append(safe_pct(n_comb, n_tot))

        misid_gamma.append(safe_pct(gamma_bin["pass_combined"].sum(), len(gamma_bin)))

    metrics = {
        "eff_gatr_cluster": np.array(eff_gatr_cluster),
        "eff_mass_rem": np.array(eff_mass_rem),
        "eff_combined": np.array(eff_combined),
    }

    return bin_centers, metrics, np.array(misid_gamma)


# =====================================================================
# 4. HEP Plotting Function
# =====================================================================
def plot_additive_efficiencies(
    bin_centers: np.ndarray,
    metrics: dict,
    misid_gamma: np.ndarray,
    exp_text: str = "CLUE Cluster-GATr Eval",
    output_path: str = "cluster_gatr_efficiency.pdf",
):
    """Generates HEP publication plot illustrating orthogonal additive efficiencies."""
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    fig, ax1 = plt.subplots(figsize=(8.5, 6), dpi=120)
    ax2 = ax1.twinx()

    # Stage 1: GATr Cluster Selection
    ax1.plot(
        bin_centers,
        metrics["eff_gatr_cluster"],
        color="#ff7f0e",
        marker="^",
        linestyle="None",
        linewidth=1.8,
        label=r"Eff: Cluster GATr Tag ($N_c=1$ or All-$\pi^0$)",
        zorder=4,
    )
    # Stage 2: Invariant Mass on Remaining Photons
    ax1.plot(
        bin_centers,
        metrics["eff_mass_rem"],
        color="#d62728",
        marker="D",
        linestyle="None",
        linewidth=1.8,
        label=r"Eff: $m_{\gamma\gamma}$ Mass Cut (Remaining Photons)",
        zorder=5,
    )
    # Stage 3: Total Combined Additive Efficiency
    ax1.plot(
        bin_centers,
        metrics["eff_combined"],
        color="#9467bd",
        marker="P",
        linestyle="None",
        linewidth=2.2,
        label=r"Eff: Total Combined ($\epsilon_{\text{GATr}} + \epsilon_{\text{Mass}}$)",
        zorder=6,
    )

    # Background Mis-ID Rate
    ax2.plot(
        bin_centers,
        misid_gamma,
        color="#2ca02c",
        marker="v",
        linestyle="None",
        linewidth=1.8,
        label=r"$\gamma \rightarrow \pi^0$ Mis-ID rate",
        zorder=3,
    )

    max_e = max(bin_centers) + 2.5 if len(bin_centers) > 0 else 80.0
    ax1.set_xlim(0, max_e)
    ax1.set_ylim(-2, 102)
    ax2.set_ylim(-2, 102)

    ax1.set_xlabel("Energy [GeV]", fontsize=14, labelpad=6)
    ax1.set_ylabel("Efficiency [%]", fontsize=14, labelpad=8)
    ax2.set_ylabel("Mis-identification rate [%]", fontsize=14, labelpad=8, rotation=270)

    for ax in [ax1, ax2]:
        ax.tick_params(axis="both", which="major", direction="in", length=7, width=1.2, labelsize=12)
        ax.tick_params(axis="both", which="minor", direction="in", length=4, width=0.8)
        ax.minorticks_on()

    words = exp_text.split()
    first_word = words[0] if words else ""
    rest_words = " ".join(words[1:]) if len(words) > 1 else ""
    ax1.text(0.04, 0.95, rf"$\mathbf{{{first_word}}}$ {rest_words}", transform=ax1.transAxes, fontsize=14, va="top")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="center right",
        bbox_to_anchor=(0.96, 0.55),
        frameon=True,
        facecolor="white",
        framealpha=0.85,
        fontsize=10.5,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"--> Saved additive efficiency plot to: {output_path}")
    plt.close()


# =====================================================================
# 5. End-to-End Orchestrator
# =====================================================================
def run_cluster_evaluation(
    test_dir: str,
    model_path: str,
    output_plot_path: str = "cluster_gatr_efficiency.pdf",
    tree_name: str = "CLUEShowers",
    extra_tree_name: str = "CLUEExtra",
    gatr_config: dict = None,
    mass_window: tuple = (0.110, 0.160),
    gatr_threshold: float = 0.5,
    num_bins: int = 20,
    exp_text: str = "CLUE Cluster-GATr",
    device: torch.device = None,
) -> pd.DataFrame:
    from src.models.gatr_models import CalorimeterGATrClassifier, GATrWrapper

    print("\n==========================================")
    print("Launching Cluster-Level GATr Evaluation")
    print("==========================================\n")

    state_dict = None
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            gatr_config = checkpoint["config"]

    gatr_config = gatr_config or DEBUG_CONFIG
    raw_model = CalorimeterGATrClassifier(gatr_config)
    if state_dict is not None:
        raw_model.load_state_dict(state_dict)

    model_wrapper = GATrWrapper(raw_model, device=device)

    df_results = load_and_eval_cluster_gatr_data(
        test_dir=test_dir,
        model_wrapper=model_wrapper,
        tree_name=tree_name,
        extra_tree_name=extra_tree_name,
        gatr_threshold=gatr_threshold,
        mass_window=mass_window,
    )

    df_pi0 = df_results[df_results["label"] == 1].copy()
    df_gamma = df_results[df_results["label"] == 0].copy()

    energy_bins = np.linspace(df_results["true_energy"].min(), df_results["true_energy"].max(), num_bins)

    bin_centers, metrics, misid_gamma = compute_binned_additive_metrics(
        df_pi0, df_gamma, energy_bins=energy_bins
    )

    plot_additive_efficiencies(
        bin_centers, metrics, misid_gamma, exp_text=exp_text, output_path=output_plot_path
    )

    print("\n--- CLUSTER-LEVEL SUMMARY ---")
    print(df_results.groupby("label")[["pass_gatr_cluster", "pass_mass_rem", "pass_combined"]].mean() * 100)
    print("-----------------------------\n")

    return df_results


if __name__ == "__main__":
    run_cluster_evaluation(
        test_dir="/eos/user/m/mgroning/ML_summer_project/CERN_summer_project/data/GATr/dataset_preselection_0.5-80GeV/split/Testing",
        model_path="/eos/user/m/mgroning/ML_summer_project/CERN_summer_project/trained_models/GATr/model_dataset_preselection_0.5-80GeV_v1/gatr_best_model.pt",
        output_plot_path="/eos/user/m/mgroning/ML_summer_project/CERN_summer_project/plots/cluster_gatr_efficiency.pdf",
    )