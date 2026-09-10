import os
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import uproot

# ==========================================
# Configuration: 1D Crack Angular Resolution Profile
# ==========================================
FILE_10GEV = "/eos/user/f/faguo/FCCSW/DD4hep_Grainita/run_performance_condor/performance_ntuples/performance_ntuple_gamma_10GeV.root"

PHI_MIN, PHI_MAX = -0.15, 0.15
N_PHI_BINS = 15  # 30 bins over 0.3 rad gives ~0.01 rad (~0.57 deg) per bin for good statistics

THETA_MIN_DEG, THETA_MAX_DEG = 70.0, 110.0
THETA_MIN, THETA_MAX = np.radians(THETA_MIN_DEG), np.radians(THETA_MAX_DEG)

phi_edges = np.linspace(PHI_MIN, PHI_MAX, N_PHI_BINS + 1)
phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])

def wrap_phi(dphi):
    return np.arctan2(np.sin(dphi), np.cos(dphi))

def calc_68_interval_with_err(data, n_boot=50):
    """Computes 68.3% effective resolution and bootstrap error estimate."""
    data = data[~np.isnan(data) & ~np.isinf(data)]
    n = len(data)
    if n < 15:
        return np.nan, np.nan
    
    def get_sig(arr):
        s_arr = np.sort(arr)
        n68 = int(np.floor(0.682689 * len(arr)))
        if n68 < 1:
            return np.nan
        intervals = s_arr[n68:] - s_arr[:-n68]
        return np.min(intervals) / 2.0

    sig_eff = get_sig(data)
    if np.isnan(sig_eff):
        return np.nan, np.nan

    # Quick Bootstrap for error
    rng = np.random.default_rng(42)
    boot_sigs = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        s_b = get_sig(sample)
        if not np.isnan(s_b):
            boot_sigs.append(s_b)
            
    sig_err = np.std(boot_sigs) if len(boot_sigs) > 5 else np.nan
    return sig_eff, sig_err

# ==========================================
# Data Extraction & Processing
# ==========================================
branches = [
    "primary_px", "primary_py", "primary_pz",
    "sim_hits_raw_log_weighted_theta", "sim_hits_raw_log_weighted_phi",
    "digi_from_raw_digihit_log_weighted_theta", "digi_from_raw_digihit_log_weighted_phi",
    "rec_from_raw_cluster_x", "rec_from_raw_cluster_y", "rec_from_raw_cluster_z"
]

with uproot.open(FILE_10GEV) as f:
    df = f["events"].arrays(branches, library="np")

p_pt = np.hypot(df["primary_px"], df["primary_py"])
p_p = np.hypot(p_pt, df["primary_pz"])
true_theta = np.arccos(np.clip(df["primary_pz"] / p_p, -1.0, 1.0))
true_phi = np.arctan2(df["primary_py"], df["primary_px"])

r = np.hypot(np.hypot(df["rec_from_raw_cluster_x"], df["rec_from_raw_cluster_y"]), df["rec_from_raw_cluster_z"])
rec_theta = np.arccos(np.clip(df["rec_from_raw_cluster_z"] / np.where(r > 0, r, 1.0), -1.0, 1.0))
rec_phi = np.arctan2(df["rec_from_raw_cluster_y"], df["rec_from_raw_cluster_x"])

# Storage arrays [mrad]
results = {
    "sim_th": ([], []), "sim_ph": ([], []),
    "digi_th": ([], []), "digi_ph": ([], []),
    "rec_th": ([], []), "rec_ph": ([], [])
}

# Restrict to central barrel slice in Theta to isolate Phi crack effect
central_mask = (true_theta >= THETA_MIN) & (true_theta < THETA_MAX)

for i in range(N_PHI_BINS):
    p_low, p_high = phi_edges[i], phi_edges[i+1]
    bin_mask = central_mask & (true_phi >= p_low) & (true_phi < p_high)
    
    # Calculate Theta/Phi residuals per level
    d_th_sim = df["sim_hits_raw_log_weighted_theta"][bin_mask] - true_theta[bin_mask]
    d_ph_sim = wrap_phi(df["sim_hits_raw_log_weighted_phi"][bin_mask] - true_phi[bin_mask])
    
    d_th_digi = df["digi_from_raw_digihit_log_weighted_theta"][bin_mask] - true_theta[bin_mask]
    d_ph_digi = wrap_phi(df["digi_from_raw_digihit_log_weighted_phi"][bin_mask] - true_phi[bin_mask])
    
    d_th_rec = rec_theta[bin_mask] - true_theta[bin_mask]
    d_ph_rec = wrap_phi(rec_phi[bin_mask] - true_phi[bin_mask])
    
    # Fit 68% containment (converted to mrad)
    for key, data in zip(["sim_th", "sim_ph", "digi_th", "digi_ph", "rec_th", "rec_ph"],
                         [d_th_sim, d_ph_sim, d_th_digi, d_ph_digi, d_th_rec, d_ph_rec]):
        sig, err = calc_68_interval_with_err(data)
        results[key][0].append(sig * 1e3 if not np.isnan(sig) else np.nan)
        results[key][1].append(err * 1e3 if not np.isnan(err) else np.nan)

# ==========================================
# Plotting 1D Angular Profiles Across Crack
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

# 1. Theta Angular Resolution vs Phi
ax1.errorbar(phi_centers, results["sim_th"][0], yerr=results["sim_th"][1], fmt="o-", color="tab:blue", capsize=3, label="Sim Level")
ax1.errorbar(phi_centers, results["digi_th"][0], yerr=results["digi_th"][1], fmt="s--", color="tab:orange", capsize=3, label="Digi Level")
ax1.errorbar(phi_centers, results["rec_th"][0], yerr=results["rec_th"][1], fmt="^:", color="tab:green", capsize=3, label="Rec Level")
ax1.set_xlabel(r"$\Phi_{\mathrm{true}}$ [rad]", fontsize=11)
ax1.set_ylabel(r"$\sigma_{\Theta, \mathrm{eff}}$ [mrad]", fontsize=11)
ax1.set_title(r"Polar Angular Resolution ($\sigma_\Theta$) Profile across Crack", fontsize=12, fontweight='bold')
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(fontsize=10)

# 2. Phi Angular Resolution vs Phi
ax2.errorbar(phi_centers, results["sim_ph"][0], yerr=results["sim_ph"][1], fmt="o-", color="tab:blue", capsize=3, label="Sim Level")
ax2.errorbar(phi_centers, results["digi_ph"][0], yerr=results["digi_ph"][1], fmt="s--", color="tab:orange", capsize=3, label="Digi Level")
ax2.errorbar(phi_centers, results["rec_ph"][0], yerr=results["rec_ph"][1], fmt="^:", color="tab:green", capsize=3, label="Rec Level")
ax2.set_xlabel(r"$\Phi_{\mathrm{true}}$ [rad]", fontsize=11)
ax2.set_ylabel(r"$\sigma_{\Phi, \mathrm{eff}}$ [mrad]", fontsize=11)
ax2.set_title(r"Azimuthal Angular Resolution ($\sigma_\Phi$) Profile across Crack", fontsize=12, fontweight='bold')
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(fontsize=10)

# Indicate known crack locations from earlier scans
for crack_phi in [-0.13, -0.04, 0.05, 0.14]:
    if PHI_MIN <= crack_phi <= PHI_MAX:
        ax1.axvline(crack_phi, color='red', linestyle=':', alpha=0.5)
        ax2.axvline(crack_phi, color='red', linestyle=':', alpha=0.5)

plt.suptitle(f"GRAiNITA 10 GeV 1D Angular Crack Scan (Central Barrel: $\\Theta \\in [{THETA_MIN_DEG}^\\circ, {THETA_MAX_DEG}^\\circ]$)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("grainita_angular_resolution_1d_crack_scan.png", dpi=300)
plt.close(fig)

print("Saved 1D angular resolution crack profile to 'grainita_angular_resolution_1d_crack_scan.png'")