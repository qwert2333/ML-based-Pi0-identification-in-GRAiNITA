import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot
import pandas as pd


def get_event_hits_spherical(
    root_filepath: str,
    tree_name: str = "CLUEClustersHits",
    event_idx: int = 0,
):
    """Extracts hit positions and energies for a single event and converts them to

    relative theta, phi in milliradians.
    """
    with uproot.open(root_filepath) as f:
        tree = f[tree_name]
        hits = tree.arrays(
            ["x", "y", "z", "energy"], entry_start=event_idx, entry_stop=event_idx + 1
        )

    x = ak.to_numpy(hits["x"][0])
    y = ak.to_numpy(hits["y"][0])
    z = ak.to_numpy(hits["z"][0])
    e = ak.to_numpy(hits["energy"][0])

    # Filter non-zero energy hits
    valid = e > 0
    x, y, z, e = x[valid], y[valid], z[valid], e[valid]

    # Calculate spherical coordinates (theta, phi)
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)

    # Shift coordinates relative to energy-weighted centroid (in mrad)
    e_tot = np.sum(e)
    mean_theta = np.sum(theta * e) / e_tot
    mean_phi = np.sum(phi * e) / e_tot

    delta_theta_mrad = (theta - mean_theta) * 1e3
    delta_phi_mrad = (np.arctan2(np.sin(phi - mean_phi), np.cos(phi - mean_phi))) * 1e3

    return delta_theta_mrad, delta_phi_mrad, e

def get_event_hits_spherical_merged(
    root_filepath: str,
    tree_name: str = "CLUEClustersHits",
    event_idx: int = 0,
    decimals: int = 2,  # Precision to round mrad coordinates for tower grouping
):
    """Extracts hit positions for a single event, converts them to relative theta, phi

    in mrad, and merges multi-layer hits within the same transverse cell tower.
    """
    with uproot.open(root_filepath) as f:
        tree = f[tree_name]
        hits = tree.arrays(
            ["x", "y", "z", "energy"], entry_start=event_idx, entry_stop=event_idx + 1
        )

    x = ak.to_numpy(hits["x"][0])
    y = ak.to_numpy(hits["y"][0])
    z = ak.to_numpy(hits["z"][0])
    e = ak.to_numpy(hits["energy"][0])

    valid = e > 0
    x, y, z, e = x[valid], y[valid], z[valid], e[valid]

    # Calculate spherical coordinates (theta, phi)
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)

    # Shift coordinates relative to energy-weighted centroid (in mrad)
    e_tot = np.sum(e)
    mean_theta = np.sum(theta * e) / e_tot
    mean_phi = np.sum(phi * e) / e_tot

    delta_theta_mrad = (theta - mean_theta) * 1e3
    delta_phi_mrad = (np.arctan2(np.sin(phi - mean_phi), np.cos(phi - mean_phi))) * 1e3

    # --- AGGREGATION / MERGING LAYER HITS PER 2D CELL TOWER ---
    df_hits = pd.DataFrame(
        {
            "d_theta": np.round(delta_theta_mrad, decimals),
            "d_phi": np.round(delta_phi_mrad, decimals),
            "energy": e,
        }
    )

    # Sum energy over layers for identical transverse tower positions
    df_merged = df_hits.groupby(["d_theta", "d_phi"], as_index=False).sum()

    return (
        df_merged["d_theta"].to_numpy(),
        df_merged["d_phi"].to_numpy(),
        df_merged["energy"].to_numpy(),
    )

def plot_pi0_vs_gamma_event_display(
    gamma_file: str,
    pion_file: str,
    event_idx: int = 0,
    tree_name: str = "CLUEClustersHits",
    output_png: str = "event_display_80GeV_gamma_vs_pi0.png",
):
    """Plots side-by-side 2D hit maps in (Delta theta, Delta phi) weighted by energy."""
    th_g, ph_g, e_g = get_event_hits_spherical_merged(gamma_file, tree_name, event_idx)
    th_p, ph_p, e_p = get_event_hits_spherical_merged(pion_file, tree_name, event_idx)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    # 1. Single Photon Plot
    sc1 = ax1.scatter(
        th_g,
        ph_g,
        c=e_g,
        s=e_g * 150 + 10,
        cmap="viridis",
        alpha=0.85,
        edgecolor="k",
        linewidth=0.3,
    )
    cbar1 = fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Hit Energy [GeV]", fontsize=11)
    ax1.set_title(
        rf"Single Photon $\gamma$ (80 GeV) — Event #{event_idx}",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlabel(r"$\Delta\Theta$ relative to centroid [mrad]", fontsize=11)
    ax1.set_ylabel(r"$\Delta\Phi$ relative to centroid [mrad]", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Pi0 Plot
    sc2 = ax2.scatter(
        th_p,
        ph_p,
        c=e_p,
        s=e_p * 150 + 10,
        cmap="plasma",
        alpha=0.85,
        edgecolor="k",
        linewidth=0.3,
    )
    cbar2 = fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Hit Energy [GeV]", fontsize=11)
    ax2.set_title(
        rf"Neutral Pion $\pi^0 \to \gamma\gamma$ (80 GeV) — Event #{event_idx}",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlabel(r"$\Delta\Theta$ relative to centroid [mrad]", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Set symmetric axis limits centered on shower core (e.g. +/- 10 mrad window)
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)

    plt.suptitle(
        r"Calorimeter Hit Display: $80\text{ GeV}$ Single Photon vs. $\pi^0 \to \gamma\gamma$ ($\theta_{\gamma\gamma} \approx 3.4\text{ mrad}$)",
        fontsize=13,
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.show()
    print(f"Saved event display plot to: {output_png}")




def find_event_index_by_energy(
    root_filepath: str,
    target_energy: float = 80.0,
    tolerance: float = 1.0,
    mc_tree: str = "MCParticles",
) -> int:
    """Scans the MCParticles tree and returns the index of the first event

    matching the target primary energy within tolerance [GeV].
    """
    with uproot.open(root_filepath) as f:
        mc = f[mc_tree].arrays(["energy", "primary"])

    # Extract energy of the primary particle per event
    primary_energies = mc["energy"][mc["primary"]]
    event_energies = ak.fill_none(ak.firsts(primary_energies), 0.0)

    # Apply energy window mask
    energy_mask = (event_energies >= (target_energy - tolerance)) & (
        event_energies <= (target_energy + tolerance)
    )

    matching_indices = ak.where(energy_mask)[0]

    if len(matching_indices) == 0:
        raise ValueError(
            f"No events found with E = {target_energy} ± {tolerance} GeV in {root_filepath}"
        )

    selected_idx = int(matching_indices[0])
    found_e = float(event_energies[selected_idx])
    print(
        f"Found target event at index {selected_idx} (True MC Energy: {found_e:.2f} GeV)"
    )

    return selected_idx


# --- Example Integration with Event Display ---

gamma_file = "/eos/user/m/mgroning/DD4hep_Grainita/output/job_20201270_final/ClueNtuple_proc_3.root"
pi0_file = "/eos/user/m/mgroning/DD4hep_Grainita/output/job_20201269_final/ClueNtuple_proc_3.root"  # Update with actual path


# 1. Identify ~80 GeV event indices automatically
gamma_idx = find_event_index_by_energy(
    gamma_file, target_energy=80.0, tolerance=1.0
)
pi0_idx = find_event_index_by_energy(pi0_file, target_energy=80.0, tolerance=1.0)

# 2. Extract hits and plot side-by-side using the identified event indices
th_g, ph_g, e_g = get_event_hits_spherical_merged(
    gamma_file, tree_name="CLUEClustersHits", event_idx=gamma_idx
)
th_p, ph_p, e_p = get_event_hits_spherical_merged(
    pi0_file, tree_name="CLUEClustersHits", event_idx=pi0_idx
)

# Plotting call using the identified indices
plot_pi0_vs_gamma_event_display(
    gamma_file,
    pi0_file,
    event_idx=pi0_idx,  # Or run separately for gamma_idx if they differ
)