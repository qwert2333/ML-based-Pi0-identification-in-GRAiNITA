#!/usr/bin/env python3

import argparse
import math
import random
import pyhepmc as HepMC3

C_MM_PER_NS = 299.792458


# ------------------------------------------------------------
# Lorentz boost
# ------------------------------------------------------------
def boost(p4, beta):
    """
    Boost four-vector p4 = (px, py, pz, E)
    by velocity beta = (bx, by, bz).

    Input/output energy and momentum units are GeV.
    """
    px, py, pz, E = p4
    bx, by, bz = beta

    b2 = bx*bx + by*by + bz*bz

    if b2 < 1e-20:
        return p4

    if b2 >= 1.0:
        raise ValueError("|beta| must be smaller than 1.")

    gamma = 1.0 / math.sqrt(1.0 - b2)
    bp = bx*px + by*py + bz*pz

    factor = ((gamma - 1.0) * bp / b2) + gamma * E

    px_new = px + factor * bx
    py_new = py + factor * by
    pz_new = pz + factor * bz
    E_new  = gamma * (E + bp)

    return px_new, py_new, pz_new, E_new


# ------------------------------------------------------------
# Generate A -> gamma gamma
# ------------------------------------------------------------
def decay_A_to_gg(mA, EA, thetaA, phiA):
    """
    Generate isotropic A -> gamma gamma.

    mA, EA: GeV
    thetaA, phiA: radians

    Returns:
        pA, p1, p2
    where each is (px, py, pz, E).
    """

    if EA < mA:
        raise ValueError(
            f"EA ({EA} GeV) must be >= mA ({mA} GeV)."
        )

    # --------------------------------------------------------
    # A momentum in lab frame
    # --------------------------------------------------------
    pA_mag = math.sqrt(max(EA*EA - mA*mA, 0.0))

    pxA = pA_mag * math.sin(thetaA) * math.cos(phiA)
    pyA = pA_mag * math.sin(thetaA) * math.sin(phiA)
    pzA = pA_mag * math.cos(thetaA)

    pA = (pxA, pyA, pzA, EA)

    # --------------------------------------------------------
    # Two-body decay in A rest frame
    #
    # For massless photons:
    #
    #       E_gamma* = |p_gamma*| = mA / 2
    # --------------------------------------------------------
    Egamma = mA / 2.0
    pgamma = Egamma

    cos_theta = random.uniform(-1.0, 1.0)
    sin_theta = math.sqrt(1.0 - cos_theta*cos_theta)
    phi = random.uniform(0.0, 2.0*math.pi)

    px = pgamma * sin_theta * math.cos(phi)
    py = pgamma * sin_theta * math.sin(phi)
    pz = pgamma * cos_theta

    p1_rest = (
        px,
        py,
        pz,
        Egamma
    )

    p2_rest = (
        -px,
        -py,
        -pz,
        Egamma
    )

    # --------------------------------------------------------
    # Boost from A rest frame to lab
    # --------------------------------------------------------
    beta = (
        pxA / EA,
        pyA / EA,
        pzA / EA
    )

    p1_lab = boost(p1_rest, beta)
    p2_lab = boost(p2_rest, beta)

    return pA, p1_lab, p2_lab


# ------------------------------------------------------------
# Main generator
# ------------------------------------------------------------
def generate(
    output,
    nevents,
    mA,
    energy_range,
    theta_range,
    phi_range,
    proper_lifetime_ns,
    seed,
    energy_sampling="log",
):
    if not math.isfinite(mA) or mA <= 0.0:
        raise ValueError("Mass of A must be finite and positive.")
    if nevents <= 0:
        raise ValueError("Number of events must be positive.")
    if not math.isfinite(proper_lifetime_ns) or proper_lifetime_ns < 0.0:
        raise ValueError("Proper mean lifetime must be finite and non-negative.")
    for name, bounds in (
        ("energy", energy_range),
        ("theta", theta_range),
        ("phi", phi_range),
    ):
        if len(bounds) != 2 or not all(math.isfinite(value) for value in bounds):
            raise ValueError(f"{name} range must have two finite bounds.")
        if bounds[0] > bounds[1]:
            raise ValueError(f"{name} range must satisfy min < max.")
    if energy_range[0] < mA:
        raise ValueError("Minimum energy must be at least the mass of A.")
    if energy_sampling not in {"log", "uniform"}:
        raise ValueError("Energy sampling must be either 'log' or 'uniform'.")
    if energy_sampling == "log" and energy_range[0] <= 0.0:
        raise ValueError("Log-uniform energy sampling requires a positive minimum.")
    if not (0.0 <= theta_range[0] < theta_range[1] <= math.pi):
        raise ValueError("Theta range must lie within [0, pi] radians.")

    random.seed(seed)

    writer = HepMC3.open(output, "w")

    for ievt in range(nevents):

        event = HepMC3.GenEvent(
            HepMC3.Units.GEV,
            HepMC3.Units.MM
        )

        event.event_number = ievt

        # Generate kinematics. Log-uniform is the default so broad ranges
        # contain comparable statistics per multiplicative energy interval.
        if energy_sampling == "log":
            log_energy = random.uniform(
                math.log(energy_range[0]), math.log(energy_range[1])
            )
            EA = math.exp(log_energy)
        else:
            EA = random.uniform(*energy_range)
        thetaA = random.uniform(*theta_range)
        phiA = random.uniform(*phi_range)
        pA, p1, p2 = decay_A_to_gg(
            mA,
            EA,
            thetaA,
            phiA
        )

        # A particle
        # A has no explicit production vertex, so HepMC3 places it at the
        # implicit root vertex: the IP at (0, 0, 0, 0).
        # Use arbitrary BSM PDG ID 9000005
        particle_A = HepMC3.GenParticle(
            HepMC3.FourVector(
                pA[0],
                pA[1],
                pA[2],
                pA[3]
            ),
            9000005,
            2
        )

        # ----------------------------------------------------
        # A decay vertex
        #
        # tau is the mean proper lifetime (the exponential scale parameter),
        # following the standard ALP/LLP convention. It is not a half-life.
        # HepMC3 vertex time is ct in the event's length unit (mm).
        # ----------------------------------------------------
        if proper_lifetime_ns <= 0.0:
            proper_time_ns = 0.
        else:
            proper_time_ns = random.expovariate(1.0 / proper_lifetime_ns)
        proper_ct_mm = C_MM_PER_NS * proper_time_ns
        decay_position = (
            pA[0] / mA * proper_ct_mm,
            pA[1] / mA * proper_ct_mm,
            pA[2] / mA * proper_ct_mm,
            pA[3] / mA * proper_ct_mm,
        )
        if not all(math.isfinite(value) for value in decay_position):
            raise ValueError("Decay vertex is not finite; check lifetime and energy ranges.")
        decay_vertex = HepMC3.GenVertex(
            HepMC3.FourVector(*decay_position)
        )

        decay_vertex.add_particle_in(particle_A)

        # Photon 1
        gamma1 = HepMC3.GenParticle(
            HepMC3.FourVector(
                p1[0],
                p1[1],
                p1[2],
                p1[3]
            ),
            22,
            1
        )

        # Photon 2
        gamma2 = HepMC3.GenParticle(
            HepMC3.FourVector(
                p2[0],
                p2[1],
                p2[2],
                p2[3]
            ),
            22,
            1
        )

        decay_vertex.add_particle_out(gamma1)
        decay_vertex.add_particle_out(gamma2)

        event.add_vertex(decay_vertex)

        writer.write(event)

    writer.close()

    print("----------------------------------------")
    print("Generation finished")
    print("----------------------------------------")
    print(f"Output        : {output}")
    print(f"Events        : {nevents}")
    print(f"mA            : {mA} GeV")
    print(f"EA range      : {energy_range} GeV")
    print(f"EA sampling   : {energy_sampling}")
    print(f"thetaA range  : {theta_range} rad")
    print(f"phiA range    : {phi_range} rad")
    print(f"proper tau    : {proper_lifetime_ns} ns (mean proper lifetime)")
    print("production    : IP (0, 0, 0, 0) mm")
    print(f"random seed   : {seed}")
    print("----------------------------------------")


# ------------------------------------------------------------
# Command line
# ------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate A -> gamma gamma events in HepMC3 format."
    )

    parser.add_argument(
        "-o", "--output",
        default="A_to_gg.hepmc3",
        help="Output HepMC3 file"
    )

    parser.add_argument(
        "-n", "--nevents",
        type=int,
        default=1000,
        help="Number of events"
    )

    parser.add_argument(
        "--mass",
        type=float,
        default=1.0,
        help="Mass of A [GeV]"
    )

    parser.add_argument(
        "--energy",
        type=float, nargs=2, metavar=("MIN", "MAX"), required=True,
        help="Energy bounds of A [GeV]"
    )

    parser.add_argument(
        "--energy-sampling",
        choices=("log", "uniform"),
        default="log",
        help=(
            "Energy distribution inside --energy bounds. Default: log, "
            "which samples uniformly in log(E)."
        )
    )

    parser.add_argument(
        "--theta",
        type=float, nargs=2, metavar=("MIN", "MAX"), default=(1.50, 1.65),
        help="Uniform polar-angle range of A [rad] (default: 1.50 1.65)"
    )

    parser.add_argument(
        "--phi",
        type=float, nargs=2, metavar=("MIN", "MAX"), default=(1.50, 1.65),
        help="Uniform azimuthal-angle range of A [rad] (default: 1.50 1.65)"
    )

    parser.add_argument(
        "--proper-lifetime-ns",
        type=float,
        required=True,
        help=(
            "Mean proper lifetime tau of A [ns]; each event samples an "
            "exponential proper decay time with mean tau"
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed"
    )

    args = parser.parse_args()

    generate(
        output=args.output,
        nevents=args.nevents,
        mA=args.mass,
        energy_range=args.energy,
        theta_range=args.theta,
        phi_range=args.phi,
        proper_lifetime_ns=args.proper_lifetime_ns,
        seed=args.seed,
        energy_sampling=args.energy_sampling,
    )
