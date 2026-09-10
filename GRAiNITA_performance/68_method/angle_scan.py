import os
import matplotlib
matplotlib.use('Agg')  # Headless backend for LXPLUS/Condor
import matplotlib.pyplot as plt
import numpy as np
import uproot

# ==========================================
# Configuration: 10 GeV High-Granularity Scan
# ==========================================
FILE_10GEV = "/eos/user/f/faguo/FCCSW/DD4hep_Grainita/run_performance_condor/performance_ntuples/performance_ntuple_gamma_10GeV.root"

N_PHI_BINS = 20
N_THETA_BINS = 15

# Set angular bounds: adjust PHI_MIN/PHI_MAX to target a single module or crack
# e.g., Full azimuth: [-np.pi, np.pi] | Single module crack zoom: [-np.pi/18, np.pi/18]
PHI_MIN, PHI_MAX = -0.15, 0.15
THETA_MIN_DEG, THETA_MAX_DEG = 70.0, 110.0

THETA_MIN = np.radians(THETA_MIN_DEG)
THETA_MAX = np.radians(THETA_MAX_DEG)

phi_edges = np.linspace(PHI_MIN, PHI_MAX, N_PHI_BINS + 1)
theta_edges = np.linspace(THETA_MIN, THETA_MAX, N_THETA_BINS + 1)

def calc_68_interval(data):
    """Computes shortest 68.3% containment interval for energy resolution."""
    n = len(data)
    if n < 10:
        return np.nan, np.nan
    sorted_data = np.sort(data)
    n_68 = int(np.floor(0.682689 * n))
    if n_68 < 1:
        return np.nan, np.nan
    
    intervals = sorted_data[n_68:] - sorted_data[:-n_68]
    min_idx = np.argmin(intervals)
    
    sigma_eff = (sorted_data[min_idx + n_68] - sorted_data[min_idx]) / 2.0
    peak_val = float(np.median(sorted_data[min_idx : min_idx + n_68]))
    return sigma_eff, peak_val

# ==========================================
# Data Extraction & Grid Calculation
# ==========================================
branches = [
    "primary_px", "primary_py", "primary_pz", "primary_energy",
    "digi_from_raw_digihit_energy_sum", "rec_from_raw_cluster_energy"
]

with uproot.open(FILE_10GEV) as f:
    df = f["events"].arrays(branches, library="np")

p_pt = np.hypot(df["primary_px"], df["primary_py"])
p_p = np.hypot(p_pt, df["primary_pz"])
true_theta = np.arccos(np.clip(df["primary_pz"] / p_p, -1.0, 1.0))
true_phi = np.arctan2(df["primary_py"], df["primary_px"])

e_true = df["primary_energy"]
e_digi_norm = df["digi_from_raw_digihit_energy_sum"] / e_true
e_rec_norm = df["rec_from_raw_cluster_energy"] / e_true

# Arrays shape: [N_THETA_BINS, N_PHI_BINS]
resp_digi = np.full((N_THETA_BINS, N_PHI_BINS), np.nan)
res_digi = np.full((N_THETA_BINS, N_PHI_BINS), np.nan)
resp_rec = np.full((N_THETA_BINS, N_PHI_BINS), np.nan)
res_rec = np.full((N_THETA_BINS, N_PHI_BINS), np.nan)

for i in range(N_THETA_BINS):
    t_low, t_high = theta_edges[i], theta_edges[i+1]
    for j in range(N_PHI_BINS):
        p_low, p_high = phi_edges[j], phi_edges[j+1]
        
        mask = (true_theta >= t_low) & (true_theta < t_high) & \
               (true_phi >= p_low) & (true_phi < p_high)
        
        if np.sum(mask) >= 10:
            sig_d, peak_d = calc_68_interval(e_digi_norm[mask])
            sig_r, peak_r = calc_68_interval(e_rec_norm[mask])
            
            if peak_d > 0:
                resp_digi[i, j] = peak_d
                res_digi[i, j] = (sig_d / peak_d) * 100.0
            if peak_r > 0:
                resp_rec[i, j] = peak_r
                res_rec[i, j] = (sig_r / peak_r) * 100.0

# ==========================================
# Heatmap Plotting
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey=True)
extent = [PHI_MIN, PHI_MAX, THETA_MIN_DEG, THETA_MAX_DEG]

# 1. Digi Peak Response
im0 = axes[0, 0].imshow(resp_digi, origin='lower', aspect='auto', extent=extent, cmap='viridis')
axes[0, 0].set_title(r"Digi Level Peak Response ($E_{\mathrm{meas}}/E_{\mathrm{true}}$)", fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel(r"$\Theta_{\mathrm{true}}$ [deg]", fontsize=11)
fig.colorbar(im0, ax=axes[0, 0], label="Response Peak")

# 2. Rec Peak Response
im1 = axes[0, 1].imshow(resp_rec, origin='lower', aspect='auto', extent=extent, cmap='viridis')
axes[0, 1].set_title(r"Rec Level Peak Response ($E_{\mathrm{meas}}/E_{\mathrm{true}}$)", fontsize=11, fontweight='bold')
fig.colorbar(im1, ax=axes[0, 1], label="Response Peak")

# 3. Digi Resolution
im2 = axes[1, 0].imshow(res_digi, origin='lower', aspect='auto', extent=extent, cmap='plasma')
axes[1, 0].set_title(r"Digi Level Energy Resolution $\sigma_{\mathrm{eff}}/E$ [%]", fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel(r"$\Phi_{\mathrm{true}}$ [rad]", fontsize=11)
axes[1, 0].set_ylabel(r"$\Theta_{\mathrm{true}}$ [deg]", fontsize=11)
fig.colorbar(im2, ax=axes[1, 0], label="Resolution [%]")

# 4. Rec Resolution
im3 = axes[1, 1].imshow(res_rec, origin='lower', aspect='auto', extent=extent, cmap='plasma')
axes[1, 1].set_title(r"Rec Level Energy Resolution $\sigma_{\mathrm{eff}}/E$ [%]", fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel(r"$\Phi_{\mathrm{true}}$ [rad]", fontsize=11)
fig.colorbar(im3, ax=axes[1, 1], label="Resolution [%]")

plt.suptitle("GRAiNITA 10 GeV Crack Scan (72 $\Phi$ Bins $\\times$ 15 $\Theta$ Bins)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("grainita_crack_scan_10GeV_72x15_singular.png", dpi=300)
plt.close(fig)

print("Saved 2D crack scan heatmap to 'grainita_crack_scan_10GeV_72x15_singular.png'")