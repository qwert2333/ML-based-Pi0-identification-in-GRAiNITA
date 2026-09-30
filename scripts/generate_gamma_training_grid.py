#!/usr/bin/env python3
"""Generate energy-matched single-photon background files."""

import argparse
import csv
from pathlib import Path

from generate_single_gamma_hepmc import generate


DEFAULT_COMPONENTS = (
    (1.0, 80.0, 20000),
    (2.5, 80.0, 5000),
    (5.0, 80.0, 5000),
    (10.0, 80.0, 5000),
)


def tag(value):
    return f"{value:g}".replace(".", "p")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-scale", type=float, default=1.0)
    parser.add_argument("--theta", type=float, nargs=2, default=(1.50, 1.65))
    parser.add_argument("--phi", type=float, nargs=2, default=(1.50, 1.65))
    parser.add_argument("--seed", type=int, default=410000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, (energy_min, energy_max, base_events) in enumerate(DEFAULT_COMPONENTS):
        events = max(1, round(base_events * args.event_scale))
        seed = args.seed + index
        name = (
            f"gamma_E{tag(energy_min)}to{tag(energy_max)}GeV_"
            f"logE_n{events}.hepmc3"
        )
        generate(
            output=str(args.output_dir / name),
            nevents=events,
            energy_range=(energy_min, energy_max),
            theta_range=tuple(args.theta),
            phi_range=tuple(args.phi),
            seed=seed,
            energy_sampling="log",
        )
        rows.append(
            {
                "file": name,
                "events": events,
                "energy_min_GeV": energy_min,
                "energy_max_GeV": energy_max,
                "energy_sampling": "log",
                "theta_min": args.theta[0],
                "theta_max": args.theta[1],
                "phi_min": args.phi[0],
                "phi_max": args.phi[1],
                "seed": seed,
            }
        )

    manifest = args.output_dir / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} files and {sum(r['events'] for r in rows)} events")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
