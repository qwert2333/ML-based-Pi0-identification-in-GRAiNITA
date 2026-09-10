import glob
import os
import re
import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
import matplotlib.pyplot as plt
import uproot

# ==========================================
# Configuration & File Discovery
# ==========================================
NTUPLE_DIR = "/eos/user/f/faguo/FCCSW/DD4hep_Grainita/run_performance_condor/performance_ntuples"
E_MIN, E_MAX = 1, 50
LOG_FILE = "grainita_fit_summary.txt"

file_pattern = os.path.join(NTUPLE_DIR, "performance_ntuple_gamma_*GeV.root")
all_files = glob.glob(file_pattern)

file_dict = {}
for filepath in all_files:
    match = re.search(r"gamma_(\d+)GeV\.root", filepath)
    if match:
        e_val = int(match.group(1))
        if E_MIN <= e_val <= E_MAX:
            file_dict[e_val] = filepath

energies = sorted(file_dict.keys())

# Initialize text log file
log = open(LOG_FILE, "w")
log.write("=" * 115 + "\n")
log.write("GRAINITA PERFORMANCE FITTING LOG WITH GOODNESS OF FIT (GoF)\n")
log.write(f"Analyzed Energies: {energies} GeV\n")
log.write("=" * 115 + "\n\n")

# ==========================================
# Helper Functions: Resolution Fitting + GoF
# ==========================================



def crystal_ball(x, N, alpha, n, mean, sigma):
    """
    Vectorized Crystal Ball function (Gaussian core with power-law low-energy tail).
    """
    t = (x - mean) / sigma
    abs_alpha = np.abs(alpha)
    
    # Gaussian core for t > -alpha
    gauss = np.exp(-0.5 * t**2)
    
    # Power-law tail for t <= -alpha
    A = (n / abs_alpha)**n * np.exp(-0.5 * abs_alpha**2)
    B = n / abs_alpha - abs_alpha
    
    # Protect against non-positive bases in fractional powers
    base = np.maximum(B - t, 1e-6)
    tail = A * (base**(-n))
    
    return N * np.where(t > -abs_alpha, gauss, tail)


def fit_resolution(data, num_sigmas=2.0):
    """
    Crystal Ball fit returning:
    (sigma_core, mean_core, resolution, sigma_err, mean_err, rel_err, chi2_ndf, p_val)
    """
    data = data[~np.isnan(data) & ~np.isinf(data)]
    if len(data) < 50:
        return (np.nan,) * 8
    
    mean_est = np.median(data)
    std_est = np.std(data)
    
    # Estimate peak position
    for _ in range(2):
        mask = np.abs(data - mean_est) < num_sigmas * std_est
        if np.sum(mask) < 20:
            break
        mean_est = np.mean(data[mask])
        std_est = np.std(data[mask])
        
    # Asymmetric range: Extend lower bound to capture the low-energy tail
    bins_range = (mean_est - 5.0 * std_est, mean_est + 2.5 * std_est)
    counts, bin_edges = np.histogram(data, bins=60, range=bins_range)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    try:
        # Initial guesses: [N, alpha, n, mean, sigma]
        p0 = [np.max(counts), 1.2, 3.0, mean_est, std_est]
        
        # Lower and upper bounds for fit stability
        lower_bounds = [0.0, 0.1, 1.01, mean_est - 2 * std_est, 1e-5]
        upper_bounds = [np.max(counts) * 2, 5.0, 20.0, mean_est + 2 * std_est, std_est * 3]
        
        popt, pcov = opt.curve_fit(
            crystal_ball, 
            bin_centers, 
            counts, 
            p0=p0, 
            bounds=(lower_bounds, upper_bounds), 
            maxfev=5000
        )
        perr = np.sqrt(np.diag(pcov))
        
        # Extract Gaussian core parameters (indices 3 and 4)
        mean_val, sigma_val = popt[3], abs(popt[4])
        mean_err, sigma_err = perr[3], perr[4]
        
        rel_res = sigma_val / mean_val if mean_val != 0 else np.nan
        rel_err = rel_res * np.sqrt((sigma_err / sigma_val)**2 + (mean_err / mean_val)**2) if mean_val != 0 else np.nan
        
        # Goodness of Fit (GoF) using 5 degrees of freedom
        expected = crystal_ball(bin_centers, *popt)
        y_err = np.sqrt(np.maximum(counts, 1.0))
        chi2 = np.sum(((counts - expected) / y_err) ** 2)
        ndf = max(1, len(counts) - 5)
        chi2_ndf = chi2 / ndf
        p_val = stats.chi2.sf(chi2, ndf)
        
        return sigma_val, mean_val, rel_res, sigma_err, mean_err, rel_err, chi2_ndf, p_val

    except Exception:
        return std_est, mean_est, std_est / mean_est if mean_est != 0 else np.nan, 0.0, 0.0, 0.0, np.nan, np.nan

def res_func(E, a, b):
    return np.sqrt((a / np.sqrt(E))**2 + b**2)

def wrap_phi(dphi):
    return np.arctan2(np.sin(dphi), np.cos(dphi))

# ==========================================
# Data Processing & Energy Fits
# ==========================================
branches = [
    "primary_px", "primary_py", "primary_pz", "primary_energy",
    "sim_hits_raw_energy_sum", "digi_from_raw_digihit_energy_sum", "rec_from_raw_cluster_energy",
    "sim_hits_raw_log_weighted_theta", "sim_hits_raw_log_weighted_phi",
    "digi_from_raw_digihit_log_weighted_theta", "digi_from_raw_digihit_log_weighted_phi",
    "rec_from_raw_cluster_x", "rec_from_raw_cluster_y", "rec_from_raw_cluster_z"
]

res_e_sim, res_e_digi, res_e_rec = [], [], []
err_e_sim, err_e_digi, err_e_rec = [], [], []

res_th_sim, res_th_digi, res_th_rec = [], [], []
res_ph_sim, res_ph_digi, res_ph_rec = [], [], []

cache_10GeV = None

log.write("--- 1. GLOBAL PER-ENERGY GAUSSIAN FIT RESULTS & GoF ---\n")
log.write(f"{'E [GeV]':<8} | {'Level':<6} | {'Mean':<9} | {'Mean Err':<9} | {'Sigma':<9} | {'Sigma Err':<9} | {'sig/E [%]':<10} | {'chi2/ndf':<10} | {'p-value':<10}\n")
log.write("-" * 115 + "\n")

for E in energies:
    fpath = file_dict[E]
    with uproot.open(fpath) as f:
        tree = f["events"]
        df = tree.arrays(branches, library="np")
        
        p_pt = np.hypot(df["primary_px"], df["primary_py"])
        p_p = np.hypot(p_pt, df["primary_pz"])
        true_theta = np.arccos(df["primary_pz"] / p_p)
        true_phi = np.arctan2(df["primary_py"], df["primary_px"])
        
        r = np.sqrt(df["rec_from_raw_cluster_x"]**2 + df["rec_from_raw_cluster_y"]**2 + df["rec_from_raw_cluster_z"]**2)
        rec_theta = np.arccos(df["rec_from_raw_cluster_z"] / r)
        rec_phi = np.arctan2(df["rec_from_raw_cluster_y"], df["rec_from_raw_cluster_x"])
        
        # Energy fits
        for level, key in zip(["Sim", "Digi", "Rec"], 
                              ["sim_hits_raw_energy_sum", "digi_from_raw_digihit_energy_sum", "rec_from_raw_cluster_energy"]):
            sig, mean, rel, sig_e, mean_e, rel_e, chi2_ndf, p_val = fit_resolution(df[key] / E)
            
            if level == "Sim":
                res_e_sim.append(rel); err_e_sim.append(rel_e)
            elif level == "Digi":
                res_e_digi.append(rel); err_e_digi.append(rel_e)
            elif level == "Rec":
                res_e_rec.append(rel); err_e_rec.append(rel_e)
            
            log.write(f"{E:<8} | {level:<6} | {mean:<9.5f} | {mean_e:<9.5f} | {sig:<9.5f} | {sig_e:<9.5f} | {rel*100:<10.3f} | {chi2_ndf:<10.2f} | {p_val:<10.4e}\n")

        # Angular fits
        sig_th_s, _, _, _, _, _, _, _ = fit_resolution(df["sim_hits_raw_log_weighted_theta"] - true_theta)
        sig_th_d, _, _, _, _, _, _, _ = fit_resolution(df["digi_from_raw_digihit_log_weighted_theta"] - true_theta)
        sig_th_r, _, _, _, _, _, _, _ = fit_resolution(rec_theta - true_theta)
        
        sig_ph_s, _, _, _, _, _, _, _ = fit_resolution(wrap_phi(df["sim_hits_raw_log_weighted_phi"] - true_phi))
        sig_ph_d, _, _, _, _, _, _, _ = fit_resolution(wrap_phi(df["digi_from_raw_digihit_log_weighted_phi"] - true_phi))
        sig_ph_r, _, _, _, _, _, _, _ = fit_resolution(wrap_phi(rec_phi - true_phi))
        
        res_th_sim.append(sig_th_s * 1e3); res_th_digi.append(sig_th_d * 1e3); res_th_rec.append(sig_th_r * 1e3)
        res_ph_sim.append(sig_ph_s * 1e3); res_ph_digi.append(sig_ph_d * 1e3); res_ph_rec.append(sig_ph_r * 1e3)
        
        if E == 10:
            cache_10GeV = {
                "true_theta": true_theta, "true_phi": true_phi,
                "sim_th": df["sim_hits_raw_log_weighted_theta"], "sim_ph": df["sim_hits_raw_log_weighted_phi"],
                "digi_th": df["digi_from_raw_digihit_log_weighted_theta"], "digi_ph": df["digi_from_raw_digihit_log_weighted_phi"],
                "rec_th": rec_theta, "rec_ph": rec_phi
            }

energies = np.array(energies)

# ==========================================
# Plot 1: Energy Resolution Curves & Global Parametric Fits
# ==========================================
log.write("\n--- 2. ENERGY RESOLUTION PARAMETRIZATION FITS & GoF ---\n")
log.write("Model: sigma_E / E = a / sqrt(E) (+) b\n")
log.write(f"{'Level':<8} | {'Stochastic a [% GeV^0.5]':<25} | {'Constant b [%]':<20} | {'chi2/ndf':<10} | {'p-value':<10}\n")
log.write("-" * 85 + "\n")

fig, ax = plt.subplots(figsize=(8, 6))
for res_e, err_e, label, fmt in zip([res_e_sim, res_e_digi, res_e_rec], 
                                    [err_e_sim, err_e_digi, err_e_rec], 
                                    ["Sim", "Digi", "Rec"], ["o", "s", "^"]):
    res_e, err_e = np.array(res_e), np.array(err_e)
    mask = ~np.isnan(res_e) & ~np.isnan(err_e) & (err_e > 0)
    
    ax.errorbar(energies[mask], res_e[mask] * 100, yerr=err_e[mask] * 100, fmt=fmt, label=f"{label} Level")
    
    try:
        err_e_total = np.sqrt(err_e[mask]**2 + 0.0003**2)
        popt, pcov = opt.curve_fit(res_func, energies[mask], res_e[mask], sigma=err_e_total, absolute_sigma=True, p0=[0.1, 0.01])
        perr = np.sqrt(np.diag(pcov))
        
        residuals = (res_e[mask] - res_func(energies[mask], *popt)) / err_e_total
        chi2 = np.sum(residuals**2)
        ndf = max(1, np.sum(mask) - 2)
        chi2_ndf = chi2 / ndf
        p_val = stats.chi2.sf(chi2, ndf)
        
        a_pct, a_err = popt[0] * 100, perr[0] * 100
        b_pct, b_err = popt[1] * 100, perr[1] * 100
        
        log.write(f"{label:<8} | {a_pct:.3f} +/- {a_err:.3f}{'':<12} | {b_pct:.3f} +/- {b_err:.3f}{'':<7} | {chi2_ndf:<10.2f} | {p_val:<10.4e}\n")
        
        e_smooth = np.linspace(1, 50, 100)
        ax.plot(e_smooth, res_func(e_smooth, *popt)*100, "--", 
                label=f"{label} Fit: {a_pct:.2f}\%/\sqrt{{E}} \oplus {b_pct:.2f}\%")
    except Exception as e:
        log.write(f"{label:<8} | FIT FAILED ({e})\n")

ax.set_xlabel("Beam Energy [GeV]", fontsize=12)
ax.set_ylabel("Energy Resolution $\sigma_E / E$ [%]", fontsize=12)
ax.set_title("Grainita Single Photon Energy Resolution", fontsize=14)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=10)
plt.savefig("grainita_energy_resolution.png", dpi=300)
plt.close()

# ==========================================
# Plot 2: Stochastic and Constant Terms vs Theta/Phi
# ==========================================
theta_bins = np.radians(np.linspace(60, 120, 11))
phi_bins = np.linspace(-np.pi, np.pi, 11)

def get_terms_vs_var(var_type="theta"):
    bins = theta_bins if var_type == "theta" else phi_bins
    a_terms, b_terms, centers = [], [], []
    
    log.write(f"\n--- BINNED RESOLUTION TERMS VS {var_type.upper()} & GoF ---\n")
    log.write(f"{'Bin Center':<12} | {'a [% GeV^0.5]':<20} | {'b [%]':<20} | {'chi2/ndf':<10} | {'p-value':<10}\n")
    log.write("-" * 80 + "\n")
    
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        center_val = np.degrees(0.5*(low+high)) if var_type == "theta" else 0.5*(low+high)
        centers.append(center_val)
        
        e_binned, res_binned, err_binned = [], [], []
        for E in energies:
            fpath = file_dict[E]
            with uproot.open(fpath) as f:
                tree = f["events"]
                px = tree["primary_px"].array(library="np")
                py = tree["primary_py"].array(library="np")
                pz = tree["primary_pz"].array(library="np")
                rec_e = tree["rec_from_raw_cluster_energy"].array(library="np")
                
                val = np.arccos(pz / np.sqrt(px**2 + py**2 + pz**2)) if var_type == "theta" else np.arctan2(py, px)
                mask = (val >= low) & (val < high)
                
                if np.sum(mask) > 50:
                    _, _, r_rec, _, _, rel_e, _, _ = fit_resolution(rec_e[mask] / E)
                    if not np.isnan(r_rec) and rel_e > 0:
                        e_binned.append(E)
                        res_binned.append(r_rec)
                        err_binned.append(rel_e)
        
        if len(e_binned) >= 3:
            try:
                popt, pcov = opt.curve_fit(res_func, e_binned, res_binned, sigma=err_binned, absolute_sigma=True, p0=[0.1, 0.01])
                perr = np.sqrt(np.diag(pcov))
                
                residuals = (np.array(res_binned) - res_func(np.array(e_binned), *popt)) / np.array(err_binned)
                chi2 = np.sum(residuals**2)
                ndf = max(1, len(e_binned) - 2)
                chi2_ndf = chi2 / ndf
                p_val = stats.chi2.sf(chi2, ndf)
                
                a_terms.append(popt[0] * 100)
                b_terms.append(popt[1] * 100)
                log.write(f"{center_val:<12.3f} | {popt[0]*100:.3f} +/- {perr[0]*100:.3f}{'':<5} | {popt[1]*100:.3f} +/- {perr[1]*100:.3f}{'':<3} | {chi2_ndf:<10.2f} | {p_val:<10.4e}\n")
            except Exception:
                a_terms.append(np.nan); b_terms.append(np.nan)
                log.write(f"{center_val:<12.3f} | FIT FAILED\n")
        else:
            a_terms.append(np.nan); b_terms.append(np.nan)
            log.write(f"{center_val:<12.3f} | INSUFFICIENT DATA\n")
            
    return np.array(centers), np.array(a_terms), np.array(b_terms)

th_centers, a_th, b_th = get_terms_vs_var("theta")
ph_centers, a_ph, b_ph = get_terms_vs_var("phi")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(th_centers, a_th, "o-", label="Stochastic $a$ [% $\sqrt{\mathrm{GeV}}$]")
ax1.plot(th_centers, b_th, "s-", label="Constant $b$ [%]")
ax1.set_xlabel("$\Theta_{\mathrm{true}}$ [deg]", fontsize=12); ax1.set_ylabel("Resolution Parameters [%]", fontsize=12)
ax1.set_title("Rec Energy Resolution Terms vs Theta", fontsize=13); ax1.grid(True, linestyle="--"); ax1.legend()

ax2.plot(ph_centers, a_ph, "o-", label="Stochastic $a$ [% $\sqrt{\mathrm{GeV}}$]")
ax2.plot(ph_centers, b_ph, "s-", label="Constant $b$ [%]")
ax2.set_xlabel("$\Phi_{\mathrm{true}}$ [rad]", fontsize=12); ax2.set_ylabel("Resolution Parameters [%]", fontsize=12)
ax2.set_title("Rec Energy Resolution Terms vs Phi", fontsize=13); ax2.grid(True, linestyle="--"); ax2.legend()

plt.tight_layout()
plt.savefig("grainita_resolution_terms_theta_phi.png", dpi=300)
plt.close()

# ==========================================
# Plot 3: Global Angular Resolution vs Energy
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for res, label, fmt in zip([res_th_sim, res_th_digi, res_th_rec], ["Sim", "Digi", "Rec"], ["o", "s", "^"]):
    ax1.plot(energies, res, fmt + "-", label=f"{label} Level")
for res, label, fmt in zip([res_ph_sim, res_ph_digi, res_ph_rec], ["Sim", "Digi", "Rec"], ["o", "s", "^"]):
    ax2.plot(energies, res, fmt + "-", label=f"{label} Level")

ax1.set_xlabel("Beam Energy [GeV]", fontsize=12); ax1.set_ylabel("$\sigma_{\Theta}$ [mrad]", fontsize=12)
ax1.set_title("Angular Resolution ($\Theta$) vs Energy", fontsize=13); ax1.grid(True, linestyle="--"); ax1.legend()
ax2.set_xlabel("Beam Energy [GeV]", fontsize=12); ax2.set_ylabel("$\sigma_{\Phi}$ [mrad]", fontsize=12)
ax2.set_title("Angular Resolution ($\Phi$) vs Energy", fontsize=13); ax2.grid(True, linestyle="--"); ax2.legend()
plt.tight_layout()
plt.savefig("grainita_angular_resolution_vs_energy.png", dpi=300)
plt.close()

# ==========================================
# Plot 4: Angular Resolution vs Theta / Phi at 10 GeV
# ==========================================
if cache_10GeV is not None:
    c = cache_10GeV
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for level, th_key, ph_key, fmt in [("Sim", "sim_th", "sim_ph", "o"),
                                       ("Digi", "digi_th", "digi_ph", "s"),
                                       ("Rec", "rec_th", "rec_ph", "^")]:
        res_th_bins, res_ph_bins = [], []
        for i in range(len(theta_bins)-1):
            mask = (c["true_theta"] >= theta_bins[i]) & (c["true_theta"] < theta_bins[i+1])
            sig, _, _, _, _, _, _, _ = fit_resolution((c[th_key][mask] - c["true_theta"][mask]))
            res_th_bins.append(sig * 1e3)
        for i in range(len(phi_bins)-1):
            mask = (c["true_phi"] >= phi_bins[i]) & (c["true_phi"] < phi_bins[i+1])
            sig, _, _, _, _, _, _, _ = fit_resolution(wrap_phi(c[ph_key][mask] - c["true_phi"][mask]))
            res_ph_bins.append(sig * 1e3)
            
        th_c_deg = np.degrees(0.5*(theta_bins[:-1] + theta_bins[1:]))
        ph_c = 0.5*(phi_bins[:-1] + phi_bins[1:])
        ax1.plot(th_c_deg, res_th_bins, fmt + "-", label=f"{level} Level")
        ax2.plot(ph_c, res_ph_bins, fmt + "-", label=f"{level} Level")

    ax1.set_xlabel("$\Theta_{\mathrm{true}}$ [deg]", fontsize=12); ax1.set_ylabel("$\sigma_{\Theta}$ [mrad]", fontsize=12)
    ax1.set_title("Angular Resolution ($\Theta$) at 10 GeV", fontsize=13); ax1.grid(True, linestyle="--"); ax1.legend()
    ax2.set_xlabel("$\Phi_{\mathrm{true}}$ [rad]", fontsize=12); ax2.set_ylabel("$\sigma_{\Phi}$ [mrad]", fontsize=12)
    ax2.set_title("Angular Resolution ($\Phi$) at 10 GeV", fontsize=13); ax2.grid(True, linestyle="--"); ax2.legend()
    plt.tight_layout()
    plt.savefig("grainita_angular_resolution_at_10GeV.png", dpi=300)
    plt.close()

log.close()
print(f"Execution complete! Plots saved as PNG files, fit details and GoF logged to '{LOG_FILE}'.")