#!/usr/bin/env python3
"""Generate single-photon HepMC3 background with log-uniform energy."""

import argparse
import math
import random

import pyhepmc as HepMC3


def generate(
    output,
    nevents,
    energy_range,
    theta_range=(1.50, 1.65),
    phi_range=(1.50, 1.65),
    seed=12345,
    energy_sampling="log",
):
    if nevents <= 0:
        raise ValueError("Number of events must be positive.")
    if energy_range[0] <= 0 or energy_range[0] > energy_range[1]:
        raise ValueError("Energy bounds must be positive and ordered.")
    if energy_sampling not in {"log", "uniform"}:
        raise ValueError("Energy sampling must be 'log' or 'uniform'.")
    random.seed(seed)

    writer = HepMC3.open(output, "w")
    for event_number in range(nevents):
        if energy_sampling == "log":
            energy = math.exp(
                random.uniform(math.log(energy_range[0]), math.log(energy_range[1]))
            )
        else:
            energy = random.uniform(*energy_range)
        theta = random.uniform(*theta_range)
        phi = random.uniform(*phi_range)
        momentum = HepMC3.FourVector(
            energy * math.sin(theta) * math.cos(phi),
            energy * math.sin(theta) * math.sin(phi),
            energy * math.cos(theta),
            energy,
        )
        event = HepMC3.GenEvent(HepMC3.Units.GEV, HepMC3.Units.MM)
        event.event_number = event_number
        # A generator-level photon may have no explicit production vertex;
        # HepMC then treats it as an orphan particle originating at the IP.
        event.add_particle(HepMC3.GenParticle(momentum, 22, 1))
        writer.write(event)
    writer.close()

    print(f"Output: {output}")
    print(f"Events: {nevents}")
    print(f"Energy: {energy_range} GeV ({energy_sampling})")
    print(f"Theta: {theta_range} rad")
    print(f"Phi: {phi_range} rad")
    print(f"Seed: {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-n", "--nevents", type=int, default=5000)
    parser.add_argument("--energy", type=float, nargs=2, required=True)
    parser.add_argument(
        "--energy-sampling", choices=("log", "uniform"), default="log"
    )
    parser.add_argument("--theta", type=float, nargs=2, default=(1.50, 1.65))
    parser.add_argument("--phi", type=float, nargs=2, default=(1.50, 1.65))
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    generate(
        output=args.output,
        nevents=args.nevents,
        energy_range=tuple(args.energy),
        theta_range=tuple(args.theta),
        phi_range=tuple(args.phi),
        seed=args.seed,
        energy_sampling=args.energy_sampling,
    )
