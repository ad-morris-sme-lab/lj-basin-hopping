"""Run a reproducible basin-hopping search for the 13-atom LJ cluster."""
import argparse
import numpy as np
from lj_basin_hopping import basin_hopping

REFERENCE_ENERGY = -44.326801

def main() -> None:
    """Optimize LJ13 from a seeded random configuration and report the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=80, help="number of basin-hopping moves")
    parser.add_argument("--seed", type=int, default=13, help="random seed")
    parser.add_argument("--plot", metavar="FILE", help="optional path for a convergence plot")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    initial = rng.normal(0.0, 1.4, size=(13, 3))
    result = basin_hopping(initial, n_steps=args.steps, seed=args.seed)
    print(f"LJ13 best energy: {result.energy:.6f} epsilon")
    print(f"Reference energy: {REFERENCE_ENERGY:.6f} epsilon")
    print(f"Absolute error: {abs(result.energy - REFERENCE_ENERGY):.3e} epsilon")
    print(f"Accepted moves: {result.accepted_moves}/{args.steps}")
    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(result.best_energy_trace)
        plt.xlabel("Basin-hopping step")
        plt.ylabel("Best-so-far energy (epsilon)")
        plt.tight_layout()
        plt.savefig(args.plot, dpi=160)

if __name__ == "__main__":
    main()
