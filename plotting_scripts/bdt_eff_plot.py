"""Pipeline for evaluating calorimeter shower reconstruction efficiency,

BDT model identification efficiency, and combined efficiency using m_inv.

Targets:
    1 = pi0 (Signal)
    0 = photon / gamma (Background)
"""

import glob
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
import vector

vector.register_awkward()

from models.bdt_models import BDTClassifier


# =====================================================================
# 1. Column Matcher Helper
# =====================================================================
def _extract_column(
    df_raw: pd.DataFrame, metadata, candidate_names: list, var_label: str
) -> np.ndarray:
    """Extracts evaluation quantities across df_raw and metadata containers."""
    # 1. Search df_raw columns
    for col in candidate_names:
        if col in df_raw.columns:
            print(f"--> [Info] Matched '{var_label}' in df_raw: '{col}'")
            return df_raw[col].values

    # 2. Search metadata DataFrame
    if isinstance(metadata, pd.DataFrame):
        for col in candidate_names:
            if col in metadata.columns:
                print(f"--> [Info] Matched '{var_label}' in metadata: '{col}'")
                return metadata[col].values

    # 3. Search metadata dictionary
    if isinstance(metadata, dict):
        for col in candidate_names:
            if col in metadata:
                print(f"--> [Info] Matched '{var_label}' in metadata dict: '{col}'")
                return np.array(metadata[col])

    # 4. Physics Fallbacks
    if var_label == "true_energy" and "total_reco_energy" in df_raw.columns:
        print("--> [WARNING] True MC energy not found in metadata. Falling back to 'total_reco_energy'.")
        return df_raw["total_reco_energy"].values

    if var_label == "m_inv":
        print("--> [WARNING] Diphoton mass 'm_inv' not found in dataset. Defaulting to 0.0.")
        return np.zeros(len(df_raw), dtype=np.float32)

    meta_keys = (
        list(metadata.columns)
        if isinstance(metadata, pd.DataFrame)
        else (list(metadata.keys()) if isinstance(metadata, dict) else [])
    )
    raise KeyError(
        f"Could not find column for '{var_label}'. Tried candidates: {candidate_names}.\n"
        f"Available df_raw columns: {list(df_raw.columns)}\n"
        f"Available metadata keys: {meta_keys}"
    )


# =====================================================================
# 2. Data Loading & BDT Evaluation
# =====================================================================
def load_and_evaluate_bdt_test_data(
    test_dir: str,
    bdt_model,
) -> pd.DataFrame:
    """Loads BDT test chunks, aligns tabular features, predicts signal scores,

    and extracts physical evaluation quantities (true_energy, label, n_clusters, m_inv).
    """
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory missing: {test_dir}")

    print(f"--> Loading BDT test dataset chunks from: '{test_dir}'...")

    wrapper = BDTClassifier()
    df_raw, y_test, metadata = wrapper.load_dataset_from_chunks(test_dir)

    model_obj = bdt_model.model if hasattr(bdt_model, "model") else bdt_model

    # Feature Alignment
    if hasattr(model_obj, "feature_names_in_"):
        expected_features = list(model_obj.feature_names_in_)
        missing = [f for f in expected_features if f not in df_raw.columns]
        if missing:
            raise ValueError(f"Test dataset is missing required BDT features: {missing}")
        X_test = df_raw[expected_features]
        print(f"--> Successfully aligned {len(expected_features)} tabular features for BDT inference.")
    else:
        # Exclude metadata/target columns if present in df_raw
        non_feature_cols = ["target_pdg", "target_energy", "true_mc_energy", "m_inv", "label"]
        X_test = df_raw.drop(columns=[c for c in non_feature_cols if c in df_raw.columns])

    print("--> Running BDT inference...")
    probs = model_obj.predict_proba(X_test)[:, 1]

    # Robust multi-container extraction for physics quantities
    true_energy = _extract_column(
        df_raw,
        metadata,
        ["true_mc_energy", "true_energy", "mc_energy", "gen_energy", "true_e", "energy", "E_true"],
        "true_energy",
    )
    n_clusters = _extract_column(
        df_raw,
        metadata,
        ["n_clusters", "n_reco_clusters", "n_clue_clusters", "num_clusters", "n_cluster"],
        "n_clusters",
    )
    m_inv = _extract_column(
        df_raw,
        metadata,
        ["m_inv", "m_yy", "invariant_mass", "diphoton_mass", "m_reco"],
        "m_inv",
    )

    labels = y_test.values if hasattr(y_test, "values") else np.array(y_test)

    return pd.DataFrame(
        {
            "true_energy": true_energy,
            "label": labels,
            "score": probs,
            "n_clusters": n_clusters,
            "m_inv": m_inv,
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
    bdt_threshold: float = 0.5,
):
    """Computes binned selection efficiencies and background rejection across pipeline stages."""
    bin_centers = 0.5 * (energy_bins[:-1] + energy_bins[1:])

    # Signal (pi0) selection conditions
    df_pi0["pass_reco"] = df_pi0["n_clusters"] > 0
    df_pi0["pass_bdt"] = df_pi0["pass_reco"] & (df_pi0["score"] >= bdt_threshold)
    df_pi0["pass_multi"] = df_pi0["n_clusters"] >= 2
    df_pi0["pass_mass"] = (
        df_pi0["pass_multi"]
        & (df_pi0["m_inv"] >= mass_window[0])
        & (df_pi0["m_inv"] <= mass_window[1])
    )
    df_pi0["pass_comb_or"] = df_pi0["pass_bdt"] | df_pi0["pass_mass"]

    # Background (gamma) rejection conditions
    df_gamma["pass_multi"] = df_gamma["n_clusters"] >= 2
    df_gamma["pass_mass"] = (
        df_gamma["pass_multi"]
        & (df_gamma["m_inv"] >= mass_window[0])
        & (df_gamma["m_inv"] <= mass_window[1])
    )
    df_gamma["pass_bdt"] = (df_gamma["n_clusters"] > 0) & (df_gamma["score"] >= bdt_threshold)

    abs_eff_reco, abs_eff_bdt, abs_eff_mass, abs_eff_comb_or = [], [], [], []
    bkg_rej_mass, bkg_rej_bdt = [], []

    def safe_pct(num, den):
        return (num / den * 100.0) if den > 0 else 0.0

    for i in range(len(energy_bins) - 1):
        e_low, e_high = energy_bins[i], energy_bins[i + 1]

        # Pi0 Signal metrics
        pi0_bin = df_pi0[
            (df_pi0["true_energy"] >= e_low) & (df_pi0["true_energy"] < e_high)
        ]
        n_tot_pi0 = len(pi0_bin)
        abs_eff_reco.append(safe_pct(pi0_bin["pass_reco"].sum(), n_tot_pi0))
        abs_eff_bdt.append(safe_pct(pi0_bin["pass_bdt"].sum(), n_tot_pi0))
        abs_eff_mass.append(safe_pct(pi0_bin["pass_mass"].sum(), n_tot_pi0))
        abs_eff_comb_or.append(safe_pct(pi0_bin["pass_comb_or"].sum(), n_tot_pi0))

        # Gamma Background Rejection metrics (1 - Pass Rate)
        gamma_bin = df_gamma[
            (df_gamma["true_energy"] >= e_low) & (df_gamma["true_energy"] < e_high)
        ]
        n_tot_gamma = len(gamma_bin)
        bkg_rej_mass.append(100.0 - safe_pct(gamma_bin["pass_mass"].sum(), n_tot_gamma))
        bkg_rej_bdt.append(safe_pct(gamma_bin["pass_bdt"].sum(), n_tot_gamma))

    metrics_abs = {
        "eff_reco": np.array(abs_eff_reco),
        "eff_bdt": np.array(abs_eff_bdt),
        "eff_mass": np.array(abs_eff_mass),
        "eff_comb_or": np.array(abs_eff_comb_or),
        "bkg_rej_mass": np.array(bkg_rej_mass),
        "bkg_rej_bdt": np.array(bkg_rej_bdt),
    }

    return bin_centers, metrics_abs


# =====================================================================
# 4. HEP Plot Generators
# =====================================================================
def plot_diphoton_mass_spectrum(
    df_results: pd.DataFrame,
    mass_window: tuple = (0.110, 0.160),
    exp_text: str = "CLUE BDT-Eval",
    output_path: str = "diphoton_mass_spectrum.pdf",
):
    """Plot 0: Diphoton invariant mass distribution comparing pi0 signal vs photon background."""
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=120)

    pi0_mass = df_results[(df_results["label"] == 1) & (df_results["n_clusters"] >= 2)]["m_inv"]
    gamma_mass = df_results[(df_results["label"] == 0) & (df_results["n_clusters"] >= 2)]["m_inv"]

    bins = np.linspace(0.0, 0.30, 60)

    ax.hist(
        pi0_mass,
        bins=bins,
        histtype="step",
        linewidth=2.0,
        color="#1f77b4",
        label=r"$\pi^0$ Signal ($N_c \geq 2$)",
        density=True,
    )
    ax.hist(
        gamma_mass,
        bins=bins,
        histtype="step",
        linewidth=2.0,
        linestyle="--",
        color="#ff7f0e",
        label=r"$\gamma$ Background ($N_c \geq 2$)",
        density=True,
    )

    # Highlight invariant mass window
    ax.axvspan(mass_window[0], mass_window[1], color="#d62728", alpha=0.15, label=f"Mass Window ({mass_window[0]}–{mass_window[1]} GeV)")
    ax.axvline(0.135, color="black", linestyle=":", alpha=0.7, label=r"Nominal $m_{\pi^0}$ (135 MeV)")

    ax.set_xlim(0.0, 0.30)
    ax.set_xlabel(r"Invariant Mass $m_{\gamma\gamma}$ [GeV]", fontsize=14, labelpad=6)
    ax.set_ylabel("Normalized Density", fontsize=14, labelpad=8)

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

    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.85, fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"--> Saved Diphoton Mass Spectrum plot to: {output_path}")
    plt.close()


def plot_clue_reconstruction_efficiency(
    bin_centers: np.ndarray,
    metrics_abs: dict,
    exp_text: str = "CLUE BDT-Eval",
    output_path: str = "clue_reco_efficiency.pdf",
):
    """Plot 1: Baseline CLUE pre-identification reconstruction efficiency."""
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=120)

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
    exp_text: str = "CLUE BDT-Eval",
    output_path: str = "bdt_combined_efficiency.pdf",
):
    """Plot 2: BDT Model ID, Mass Window Cut, and Combined Efficiency."""
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=120)

    ax.plot(
        bin_centers,
        metrics_abs["eff_bdt"],
        color="#ff7f0e",
        marker="^",
        linestyle="None",
        linewidth=1.8,
        label=r"BDT Model ID",
        zorder=3,
    )

    ax.plot(
        bin_centers,
        metrics_abs["bkg_rej_bdt"],
        color="#d62728",
        marker="D",
        linestyle="None",
        linewidth=2.0,
        label=r"Purity BDT ($\epsilon_\gamma$)",
        zorder=4,
    )

    #ax.plot(
    #    bin_centers,
    #    metrics_abs["eff_comb_or"],
    #    color="#9467bd",
    #    marker="P",
    #    linestyle="None",
    #    linewidth=2.2,
    #    label=r"Combined Eff (BDT or Mass)",
    #    zorder=5,
    #)

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
    print(f"--> Saved BDT Combined Efficiency plot to: {output_path}")
    plt.close()


# =====================================================================
# 5. End-to-End Execution
# =====================================================================
def run_bdt_evaluation(
    test_dir: str = None,
    model_path: str = None,
    output_dir: str = "./plots",
    mass_window: tuple = (0.110, 0.160),
    bdt_threshold: float = 0.5,
    num_bins: int = 20,
    exp_text: str = "CLUE BDT-Eval",
) -> pd.DataFrame:
    print("\n==========================================")
    print("Starting BDT Physics Evaluation Pipeline")
    print("==========================================\n")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"--> Loading trained BDT model checkpoint from: {model_path}")
    loaded_model = joblib.load(model_path)

    df_results = load_and_evaluate_bdt_test_data(
        test_dir=test_dir,
        bdt_model=loaded_model,
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
        bdt_threshold=bdt_threshold,
    )

    # Plot 0: Invariant Mass Spectrum
    plot_diphoton_mass_spectrum(
        df_results,
        mass_window=mass_window,
        exp_text=exp_text,
        output_path=os.path.join(output_dir, "diphoton_mass_spectrum.pdf"),
    )

    # Plot 1: CLUE Reconstruction Baseline
    plot_clue_reconstruction_efficiency(
        bin_centers,
        metrics_abs,
        exp_text=exp_text,
        output_path=os.path.join(output_dir, "clue_reco_efficiency.pdf"),
    )

    # Plot 2: BDT Model ID and Combined Efficiency
    plot_combined_efficiency(
        bin_centers,
        metrics_abs,
        exp_text=exp_text,
        output_path=os.path.join(output_dir, "bdt_combined_efficiency.pdf"),
    )

    print("\n--- BDT DIAGNOSTIC SUMMARY ---")
    print(df_results.groupby("label")[["n_clusters", "m_inv", "score"]].describe())
    print("-------------------------------\n")
    print("\n[Evaluation Complete] BDT pipeline execution finished successfully!\n")
    return df_results


if __name__ == "__main__":
    run_bdt_evaluation(
        test_dir="/eos/user/m/mgroning/ML_summer_project/CERN_summer_project/data/BDT/dataset_bdt_with_inv_mass/Testing",
        model_path="/eos/user/m/mgroning/ML_summer_project/CERN_summer_project/trained_models/BDT/model_bdt_preselection/clue_gamma_pi0_bdt_model.joblib",
        output_dir="/eos/user/m/mgroning/ML_summer_project/CERN_summer_project/trained_models/BDT/model_bdt_preselection/",
        mass_window=(0.110, 0.160),
        bdt_threshold=0.5,
        num_bins=20,
    )