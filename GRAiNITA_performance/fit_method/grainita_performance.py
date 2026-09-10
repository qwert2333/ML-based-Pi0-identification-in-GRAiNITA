import glob
import os
import re
import numpy as np
import scipy.optimize as opt
import scipy.stats as stats

# Force non-interactive backend BEFORE importing pyplot for headless environments (LXPLUS / Condor)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import uproot

# ==========================================
# Configuration & File Discovery
# ==========================================
NTUPLE_DIR = "/eos/user/f/faguo/FCCSW/DD4hep_Grainita/run_performance_condor/performance_ntuples"
E_MIN, E_MAX = 1, 50
LOG_FILE = "grainita_fit_summary.txt"
PLOT_DIR_1D = os.path.abspath("plots_1d")

os.makedirs(PLOT_DIR_1D, exist_ok=True)

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
# Unified Crystal Ball & Dynamic Fitting Engine
# ==========================================
def crystal_ball(x, amplitude, alpha, n, mean, sigma):
    """Vectorized Crystal Ball function."""
    z = (x - mean) / sigma
    abs_alpha = np.abs(alpha)
    A = (n / abs_alpha)**n * np.exp(-abs_alpha**2 / 2.0)
    B = n / abs_alpha - abs_alpha
    gaussian_part = np.exp(-z**2 / 2.0)
    safe_tail = np.maximum(B - z, 1e-10)
    tail_part = A * (safe_tail**(-n))
    return amplitude * np.where(z > -abs_alpha, gaussian_part, tail_part)


def fit_resolution(data, low_bound=1.5, up_bound=1.2, energy_gev=None, level_label=None, save_dir=PLOT_DIR_1D):
    """
    Fits Crystal Ball distribution with dynamic percentile range estimation, 
    calculates GoF statistics, and optionally exports 2-panel diagnostic plots.
    """
    data = data[~np.isnan(data) & ~np.isinf(data)]
    if len(data) < 50:
        return (np.nan,) * 8

    # Dynamic percentile estimation around peak
    mu_approx = float(np.median(data))
    q75, q25 = np.percentile(data, [75, 25])
    sigma_approx = (q75 - q25) / 1.349
    if sigma_approx <= 0:
        sigma_approx = np.std(data) if np.std(data) > 0 else 1e-3

    fit_min = mu_approx - low_bound * sigma_approx
    fit_max = mu_approx + up_bound * sigma_approx

    counts, bin_edges = np.histogram(data, bins=35, range=(fit_min, fit_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_widths = np.diff(bin_edges)
    data_errors = np.sqrt(np.maximum(counts, 1.0))

    # Initial Guesses & Bounds
    amp_guess = float(np.max(counts))
    mean_guess = np.clip(mu_approx, fit_min, fit_max)
    sigma_guess = np.clip(sigma_approx, 1e-5, (fit_max - fit_min) / 2.0)

    p0 = [amp_guess, 1.2, 3.0, mean_guess, sigma_guess]
    bounds = (
        [0, 0.1, 0.5, fit_min, 1e-5],
        [np.inf, 5.0, 10.0, fit_max, (fit_max - fit_min) / 2.0]
    )

    try:
        popt, pcov = opt.curve_fit(
            crystal_ball, bin_centers, counts, 
            p0=p0, bounds=bounds, maxfev=10000
        )
        amp, fit_alpha, fit_n, mean_cb, sigma_cb = popt
        sigma_cb = abs(sigma_cb)

        perr = np.sqrt(np.diag(pcov)) if not np.isinf(pcov).any() else np.zeros(5)
        mean_err, sigma_err = perr[3], perr[4]

        rel_res = sigma_cb / mean_cb if mean_cb != 0 else np.nan
        rel_err = rel_res * np.sqrt((sigma_err / sigma_cb)**2 + (mean_err / mean_cb)**2) if mean_cb != 0 else np.nan

        # Goodness of Fit Calculation
        fit_values = crystal_ball(bin_centers, *popt)
        mask = (counts >= 5) & (fit_values > 0)
        chi2_val = np.sum(((counts[mask] - fit_values[mask]) ** 2) / fit_values[mask])
        ndf = max(1, np.count_nonzero(mask) - len(popt))
        chi2_ndf = chi2_val / ndf
        p_val = 1.0 - stats.chi2.cdf(chi2_val, ndf)

        # Produce 2-Panel Spectrum & Pull Residual Plot
        if energy_gev is not None and level_label is not None:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(8, 7), sharex=True, 
                gridspec_kw={'height_ratios': [3, 1]}
            )
            fig.subplots_adjust(hspace=0.05)

            # Spectrum & Fit Curve
            ax1.errorbar(bin_centers, counts, yerr=data_errors, fmt='k.', capsize=2, label=f'Data ({energy_gev} GeV {level_label})')
            x_fine = np.linspace(fit_min, fit_max, 500)
            ax1.plot(x_fine, crystal_ball(x_fine, *popt), 'r-', lw=2, label='Crystal Ball Fit')
            ax1.set_ylabel(f'Events / ({bin_widths[0]:.4f})', fontsize=11)
            ax1.set_title(f'Grainita {energy_gev} GeV [{level_label}] Response Spectrum', fontsize=12, fontweight='bold')
            ax1.grid(True, linestyle=':', alpha=0.5)
            ax1.legend(loc='upper right', fontsize=10)

            textstr = '\n'.join((
                r'$\alpha = %.3f$' % (fit_alpha, ),
                r'$n = %.3f$' % (fit_n, ),
                r'$\mu = %.4f \pm %.4f$' % (mean_cb, mean_err),
                r'$\sigma = %.4f \pm %.4f$' % (sigma_cb, sigma_err),
                r'$\sigma/E = %.2f \pm %.2f\%%$' % (rel_res * 100.0, rel_err * 100.0),
                r'$\chi^2/\mathrm{ndf} = %.1f / %d = %.2f$' % (chi2_val, ndf, chi2_ndf),
                r'$p\mathrm{-value} = %.4f$' % (p_val, )
            ))
            ax1.text(
                0.04, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
            )

            # Residual Pull Panel
            residuals = np.zeros_like(counts, dtype=float)
            has_data_mask = data_errors > 0
            residuals[has_data_mask] = (counts[has_data_mask] - fit_values[has_data_mask]) / data_errors[has_data_mask]

            ax2.errorbar(bin_centers[has_data_mask], residuals[has_data_mask], yerr=np.ones_like(residuals[has_data_mask]), fmt='g.', capsize=2)
            ax2.axhline(0, color='black', linestyle='--', lw=1.2)
            ax2.axhspan(-1, 1, color='gray', alpha=0.2)
            ax2.axhline(2, color='gray', linestyle=':', alpha=0.5)
            ax2.axhline(-2, color='gray', linestyle=':', alpha=0.5)

            ax2.set_xlabel(r'$E_{\mathrm{meas}} / E_{\mathrm{true}}$', fontsize=11)
            ax2.set_ylabel(r'$\Delta / \sigma_{\mathrm{stat}}$', fontsize=11)
            ax2.set_ylim(-4, 4)
            ax2.grid(True, linestyle=':', alpha=0.5)

            output_filepath = os.path.join(save_dir, f"fit_{level_label}_{energy_gev}GeV.pdf")
            plt.savefig(output_filepath, bbox_inches='tight', dpi=300)
            plt.close(fig)

        return sigma_cb, mean_cb, rel_res, sigma_err, mean_err, rel_err, chi2_ndf, p_val

    except Exception as e:
        if energy_gev is not None and level_label is not None:
            print(f"[WARNING] Fit failed for {energy_gev} GeV [{level_label}]: {e}")
        return sigma_approx, mu_approx, sigma_approx / mu_approx if mu_approx != 0 else np.nan, 0.0, 0.0, 0.0, np.nan, np.nan


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
err_th_sim, err_th_digi, err_th_rec = [], [], []

res_ph_sim, res_ph_digi, res_ph_rec = [], [], []
err_ph_sim, err_ph_digi, err_ph_rec = [], [], []

cache_10GeV = None

log.write("--- 1. GLOBAL PER-ENERGY CRYSTAL BALL FIT RESULTS & GoF ---\n")
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
        z_over_r = np.divide(
            df["rec_from_raw_cluster_z"], r, 
            out=np.zeros_like(r, dtype=float), 
            where=r > 0
        )
        rec_theta = np.arccos(np.clip(z_over_r, -1.0, 1.0))
        rec_phi = np.arctan2(df["rec_from_raw_cluster_y"], df["rec_from_raw_cluster_x"])
        
        # Global energy fits (Generates 2-panel spectrum plots dynamically in PLOT_DIR_1D)
        for level, key in zip(["Sim", "Digi", "Rec"], 
                              ["sim_hits_raw_energy_sum", "digi_from_raw_digihit_energy_sum", "rec_from_raw_cluster_energy"]):
            sig, mean, rel, sig_e, mean_e, rel_e, chi2_ndf, p_val = fit_resolution(df[key] / E, energy_gev=E, level_label=level)
            
            if level == "Sim":
                res_e_sim.append(rel); err_e_sim.append(rel_e)
            elif level == "Digi":
                res_e_digi.append(rel); err_e_digi.append(rel_e)
            elif level == "Rec":
                res_e_rec.append(rel); err_e_rec.append(rel_e)
            
            log.write(f"{E:<8} | {level:<6} | {mean:<9.5f} | {mean_e:<9.5f} | {sig:<9.5f} | {sig_e:<9.5f} | {rel*100:<10.3f} | {chi2_ndf:<10.2f} | {p_val:<10.4e}\n")

        # Angular fits with uncertainty extraction
        sig_th_s, _, _, err_th_s, _, _, _, _ = fit_resolution(df["sim_hits_raw_log_weighted_theta"] - true_theta)
        sig_th_d, _, _, err_th_d, _, _, _, _ = fit_resolution(df["digi_from_raw_digihit_log_weighted_theta"] - true_theta)
        sig_th_r, _, _, err_th_r, _, _, _, _ = fit_resolution(rec_theta - true_theta)
        
        sig_ph_s, _, _, err_ph_s, _, _, _, _ = fit_resolution(wrap_phi(df["sim_hits_raw_log_weighted_phi"] - true_phi))
        sig_ph_d, _, _, err_ph_d, _, _, _, _ = fit_resolution(wrap_phi(df["digi_from_raw_digihit_log_weighted_phi"] - true_phi))
        sig_ph_r, _, _, err_ph_r, _, _, _, _ = fit_resolution(wrap_phi(rec_phi - true_phi))
        
        res_th_sim.append(sig_th_s * 1e3); err_th_sim.append(err_th_s * 1e3)
        res_th_digi.append(sig_th_d * 1e3); err_th_digi.append(err_th_d * 1e3)
        res_th_rec.append(sig_th_r * 1e3); err_th_rec.append(err_th_r * 1e3)

        res_ph_sim.append(sig_ph_s * 1e3); err_ph_sim.append(err_ph_s * 1e3)
        res_ph_digi.append(sig_ph_d * 1e3); err_ph_digi.append(err_ph_d * 1e3)
        res_ph_rec.append(sig_ph_r * 1e3); err_ph_rec.append(err_ph_r * 1e3)
        
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
    
    ax.errorbar(energies[mask], res_e[mask] * 100, yerr=err_e[mask] * 100, fmt=fmt, capsize=3, label=f"{label} Level")
    
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
                label=rf"{label} Fit: {a_pct:.2f}\%/\sqrt{{E}} \oplus {b_pct:.2f}\%")
    except Exception as e:
        log.write(f"{label:<8} | FIT FAILED ({e})\n")

ax.set_xlabel("Beam Energy [GeV]", fontsize=12)
ax.set_ylabel(r"Energy Resolution $\sigma_E / E$ [%]", fontsize=12)
ax.set_title("Grainita Single Photon Energy Resolution", fontsize=14)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=10)
plt.savefig("grainita_energy_resolution.png", dpi=300)
plt.close(fig)

# ==========================================
# Plot 2: Stochastic and Constant Terms vs Theta/Phi (Digi Level)
# ==========================================
theta_bins = np.radians(np.linspace(60, 120, 11))
phi_bins = np.linspace(-np.pi, np.pi, 11)

def get_terms_vs_var(var_type="theta"):
    bins = theta_bins if var_type == "theta" else phi_bins
    a_terms, a_errs, b_terms, b_errs, centers = [], [], [], [], []
    
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
                digi_e = tree["digi_from_raw_digihit_energy_sum"].array(library="np")
                
                val = np.arccos(pz / np.sqrt(px**2 + py**2 + pz**2)) if var_type == "theta" else np.arctan2(py, px)
                mask = (val >= low) & (val < high)
                
                if np.sum(mask) > 50:
                    _, _, r_rec, _, _, rel_e, _, _ = fit_resolution(digi_e[mask] / E)
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
                a_errs.append(perr[0] * 100)
                b_terms.append(popt[1] * 100)
                b_errs.append(perr[1] * 100)
                log.write(f"{center_val:<12.3f} | {popt[0]*100:.3f} +/- {perr[0]*100:.3f}{'':<5} | {popt[1]*100:.3f} +/- {perr[1]*100:.3f}{'':<3} | {chi2_ndf:<10.2f} | {p_val:<10.4e}\n")
            except Exception as e:
                a_terms.append(np.nan); a_errs.append(np.nan)
                b_terms.append(np.nan); b_errs.append(np.nan)
                log.write(f"{center_val:<12.3f} | FIT FAILED ({e})\n")
        else:
            a_terms.append(np.nan); a_errs.append(np.nan)
            b_terms.append(np.nan); b_errs.append(np.nan)
            log.write(f"{center_val:<12.3f} | INSUFFICIENT DATA\n")
            
    return np.array(centers), np.array(a_terms), np.array(a_errs), np.array(b_terms), np.array(b_errs)

th_centers, a_th, a_th_err, b_th, b_th_err = get_terms_vs_var("theta")
ph_centers, a_ph, a_ph_err, b_ph, b_ph_err = get_terms_vs_var("phi")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.errorbar(th_centers, a_th, yerr=a_th_err, fmt="o-", capsize=3, label=r"Stochastic $a$ [% $\sqrt{\mathrm{GeV}}$]")
ax1.errorbar(th_centers, b_th, yerr=b_th_err, fmt="s-", capsize=3, label=r"Constant $b$ [%]")
ax1.set_xlabel(r"$\Theta_{\mathrm{true}}$ [deg]", fontsize=12); ax1.set_ylabel("Resolution Parameters [%]", fontsize=12)
ax1.set_title("Digi Energy Resolution Terms vs Theta", fontsize=13); ax1.grid(True, linestyle="--"); ax1.legend()

ax2.errorbar(ph_centers, a_ph, yerr=a_ph_err, fmt="o-", capsize=3, label=r"Stochastic $a$ [% $\sqrt{\mathrm{GeV}}$]")
ax2.errorbar(ph_centers, b_ph, yerr=b_ph_err, fmt="s-", capsize=3, label=r"Constant $b$ [%]")
ax2.set_xlabel(r"$\Phi_{\mathrm{true}}$ [rad]", fontsize=12); ax2.set_ylabel("Resolution Parameters [%]", fontsize=12)
ax2.set_title("Digi Energy Resolution Terms vs Phi", fontsize=13); ax2.grid(True, linestyle="--"); ax2.legend()

plt.tight_layout()
plt.savefig("grainita_resolution_terms_theta_phi.png", dpi=300)
plt.close(fig)

# ==========================================
# Plot 3: Global Angular Resolution vs Energy
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for res, err, label, fmt in zip([res_th_sim, res_th_digi, res_th_rec],
                                [err_th_sim, err_th_digi, err_th_rec],
                                ["Sim", "Digi", "Rec"], ["o", "s", "^"]):
    ax1.errorbar(energies, res, yerr=err, fmt=fmt + "-", capsize=3, label=f"{label} Level")

for res, err, label, fmt in zip([res_ph_sim, res_ph_digi, res_ph_rec],
                                [err_ph_sim, err_ph_digi, err_ph_rec],
                                ["Sim", "Digi", "Rec"], ["o", "s", "^"]):
    ax2.errorbar(energies, res, yerr=err, fmt=fmt + "-", capsize=3, label=f"{label} Level")

ax1.set_xlabel("Beam Energy [GeV]", fontsize=12); ax1.set_ylabel(r"$\sigma_{\Theta}$ [mrad]", fontsize=12)
ax1.set_title(r"Angular Resolution ($\Theta$) vs Energy", fontsize=13); ax1.grid(True, linestyle="--"); ax1.legend()
ax2.set_xlabel("Beam Energy [GeV]", fontsize=12); ax2.set_ylabel(r"$\sigma_{\Phi}$ [mrad]", fontsize=12)
ax2.set_title(r"Angular Resolution ($\Phi$) vs Energy", fontsize=13); ax2.grid(True, linestyle="--"); ax2.legend()
plt.tight_layout()
plt.savefig("grainita_angular_resolution_vs_energy.png", dpi=300)
plt.close(fig)

# ==========================================
# Plot 4: Angular Resolution vs Theta / Phi at 10 GeV
# ==========================================
if cache_10GeV is not None:
    c = cache_10GeV
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for level, th_key, ph_key, fmt in [("Sim", "sim_th", "sim_ph", "o"),
                                       ("Digi", "digi_th", "digi_ph", "s"),
                                       ("Rec", "rec_th", "rec_ph", "^")]:
        res_th_bins, err_th_bins = [], []
        res_ph_bins, err_ph_bins = [], []
        
        for i in range(len(theta_bins)-1):
            mask = (c["true_theta"] >= theta_bins[i]) & (c["true_theta"] < theta_bins[i+1])
            sig, _, _, sig_err, _, _, _, _ = fit_resolution((c[th_key][mask] - c["true_theta"][mask]))
            res_th_bins.append(sig * 1e3)
            err_th_bins.append(sig_err * 1e3)
            
        for i in range(len(phi_bins)-1):
            mask = (c["true_phi"] >= phi_bins[i]) & (c["true_phi"] < phi_bins[i+1])
            sig, _, _, sig_err, _, _, _, _ = fit_resolution(wrap_phi(c[ph_key][mask] - c["true_phi"][mask]))
            res_ph_bins.append(sig * 1e3)
            err_ph_bins.append(sig_err * 1e3)
            
        th_c_deg = np.degrees(0.5*(theta_bins[:-1] + theta_bins[1:]))
        ph_c = 0.5*(phi_bins[:-1] + phi_bins[1:])
        
        ax1.errorbar(th_c_deg, res_th_bins, yerr=err_th_bins, fmt=fmt + "-", capsize=3, label=f"{level} Level")
        ax2.errorbar(ph_c, res_ph_bins, yerr=err_ph_bins, fmt=fmt + "-", capsize=3, label=f"{level} Level")

    ax1.set_xlabel(r"$\Theta_{\mathrm{true}}$ [deg]", fontsize=12); ax1.set_ylabel(r"$\sigma_{\Theta}$ [mrad]", fontsize=12)
    ax1.set_title(r"Angular Resolution ($\Theta$) at 10 GeV", fontsize=13); ax1.grid(True, linestyle="--"); ax1.legend()
    ax2.set_xlabel(r"$\Phi_{\mathrm{true}}$ [rad]", fontsize=12); ax2.set_ylabel(r"$\sigma_{\Phi}$ [mrad]", fontsize=12)
    ax2.set_title(r"Angular Resolution ($\Phi$) at 10 GeV", fontsize=13); ax2.grid(True, linestyle="--"); ax2.legend()
    plt.tight_layout()
    plt.savefig("grainita_angular_resolution_at_10GeV.png", dpi=300)
    plt.close(fig)

log.close()
print(f"Execution complete!\n- 2-panel 1D diagnostic PDF plots saved in '{PLOT_DIR_1D}/'\n- Performance summary figures with error bars saved to working directory\n- Fit log exported to '{LOG_FILE}'.")