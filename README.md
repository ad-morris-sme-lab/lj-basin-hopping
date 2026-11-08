# Basin-Hopping Optimization of an LJ13 Cluster

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A reproducible global-search demonstration that builds on a pairwise Lennard-Jones energy model by combining random basin-hopping moves with local L-BFGS-B geometry optimization. It uses N=13 particles and reduced Lennard-Jones units.

## Why this problem and system size

Basin hopping is a natural next step after calculating Lennard-Jones energies because it uses the energy and gradient to search across multiple local minima. I chose N=13 because it is small enough for fast, reproducible runs and has a well-established icosahedral global-minimum energy for validation.

## How to run locally

The following commands create a fresh Conda environment, install the project, run the example, and execute the tests:

```powershell
conda create -n lj-basin-hopping -c conda-forge python=3.12 numpy=2.5.4 scipy=1.18.1 matplotlib=3.11.2 pytest=9.2.0 pip
conda activate lj-basin-hopping
python -m pip install -e .
python examples/basin_hopping_lj13.py
python -m pytest -q
```

For a standard virtual environment, install the pinned versions from `requirements.txt` instead.

## Run

```bash
python examples/basin_hopping_lj13.py
python -m pytest -q
```

Example output with the default seed:

```text
LJ13 best energy: -44.326801 epsilon
Reference energy: -44.326801 epsilon
Absolute error: 4.195e-07 epsilon
Accepted moves: 29/80
```

The example uses a seeded random start, so its output is deterministic for the pinned numerical stack. It reports the best energy found, the reference energy, the absolute difference, and accepted moves. To save an optional convergence plot, pass a path: `python examples/basin_hopping_lj13.py --plot convergence.png`.

The test suite checks finite results and monotone best-energy traces for three fixed random initial geometries (seeds 1, 7, and 13). A separate 80-step run from the documented seed 13 checks recovery of the LJ13 reference energy. Short heuristic searches are not required to find the global minimum from every random geometry.

Reference LJ13 minimum: `-44.326801 epsilon`. Lengths are in sigma, energies in epsilon, and reduced temperature is k_B T / epsilon. The heuristic search can find a local minimum and does not guarantee discovery of the global minimum for every random start or finite run length.

## Abstract

This project implements a reproducible basin-hopping search for the 13-particle Lennard-Jones cluster. Each random displacement is followed by analytic-gradient L-BFGS-B relaxation, and a Metropolis criterion controls moves between local minima. The seeded example recovers the reported icosahedral LJ13 minimum near -44.326801 in reduced energy units. Tests cover the pair minimum, gradient consistency, translational invariance, malformed geometries, deterministic optimization, and local-minimizer failure handling. The calculation uses classical pairwise interactions in reduced units and is intended as a compact teaching example; finite search effort does not certify global optimality for arbitrary initial structures.

## Cite this work

```bibtex
@software{lj_basin_hopping_2026,
  author = {{Lennard-Jones Basin-Hopping contributors}},
  title = {Basin-Hopping Optimization of an LJ13 Cluster},
  version = {1.0.0},
  year = {2026},
  url = {https://github.com/ad-morris-sme-lab/lj-basin-hopping}
}
```

## Theory and references

See [docs/theory.md](docs/theory.md) for the model, equations, assumptions, and references. The reference N=13 minimum is listed by the [Cambridge Cluster Database](https://doye.chem.ox.ac.uk/jon/structures/LJ/tables.150.html). Basin hopping is described by Wales and Doye, [J. Phys. Chem. A 101, 5111-5116 (1997)](https://doi.org/10.1021/jp970984n).

## License

MIT, see [LICENSE](LICENSE).
