#!/usr/bin/env python3
"""Generate the baseline A -> gamma gamma multi-task training grid."""

import argparse
import csv
import math
from pathlib import Path

from generate_Ayy_hepmc import C_MM_PER_NS, generate


DEFAULT_MASSES_GEV = (0.05, 0.10, 0.135, 0.20, 0.50, 1.00, 2.00)
DEFAULT_MEAN_LAB_LENGTHS_MM = (0.0, 100.0, 300.0, 800.0, 1600.0)


def tag(value):
    return f"{value:g}".replace(".", "p")


def proper_lifetime_ns(mass_gev, reference_energy_gev, mean_lab_length_mm):
    if mean_lab_length_mm == 0.0:
        return 0.0
    momentum = math.sqrt(reference_energy_gev**2 - mass_gev**2)
    return mean_lab_length_mm * mass_gev / (C_MM_PER_NS * momentum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--events-per-point", type=int, default=1000)
    parser.add_argument("--energy-max", type=float, default=80.0)
    parser.add_argument("--reference-energy", type=float, default=20.0)
    parser.add_argument("--theta", type=float, nargs=2, default=(1.50, 1.65))
    parser.add_argument("--phi", type=float, nargs=2, default=(1.50, 1.65))
    parser.add_argument("--seed", type=int, default=310000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    point = 0
    for mass in DEFAULT_MASSES_GEV:
        energy_min = max(1.0, 5.0 * mass)
        if energy_min >= args.energy_max:
            raise ValueError(f"Empty energy range for mass {mass:g} GeV")
        for mean_length in DEFAULT_MEAN_LAB_LENGTHS_MM:
            tau = proper_lifetime_ns(
                mass, args.reference_energy, mean_length
            )
            seed = args.seed + point
            name = (
                f"Atoyy_mA{tag(mass)}GeV_"
                f"tau{tag(tau)}ns_Lref{tag(mean_length)}mm_"
                f"E{tag(energy_min)}to{tag(args.energy_max)}GeV.hepmc3"
            )
            output = args.output_dir / name
            generate(
                output=str(output),
                nevents=args.events_per_point,
                mA=mass,
                energy_range=(energy_min, args.energy_max),
                theta_range=tuple(args.theta),
                phi_range=tuple(args.phi),
                proper_lifetime_ns=tau,
                seed=seed,
                energy_sampling="log",
            )
            rows.append(
                {
                    "file": name,
                    "events": args.events_per_point,
                    "mass_GeV": mass,
                    "proper_lifetime_ns": tau,
                    "reference_mean_lab_length_mm": mean_length,
                    "reference_energy_GeV": args.reference_energy,
                    "energy_min_GeV": energy_min,
                    "energy_max_GeV": args.energy_max,
                    "energy_sampling": "log",
                    "theta_min": args.theta[0],
                    "theta_max": args.theta[1],
                    "phi_min": args.phi[0],
                    "phi_max": args.phi[1],
                    "seed": seed,
                }
            )
            point += 1

    manifest = args.output_dir / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} files and {sum(r['events'] for r in rows)} events")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
