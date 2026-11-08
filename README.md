# Basin-Hopping Optimization of an LJ13 Cluster

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A reproducible global-search demonstration that builds on a pairwise Lennard-Jones energy model by combining random basin-hopping moves with local L-BFGS-B geometry optimization. It uses N=13 particles and reduced Lennard-Jones units.

## Why this problem and system size

Basin hopping is a natural next step after calculating Lennard-Jones energies because it uses the energy and gradient to search across multiple local minima. I chose N=13 because it is small enough for fast, reproducible runs and has a well-established icosahedral global-minimum energy for validation.

## Install

Python 3.10 or newer is supported. Dependencies are pinned with Python-version markers for compatible wheels.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Run

```bash
python examples/basin_hopping_lj13.py
python -m pytest -q
```

The example uses a seeded random start, so its output is deterministic for the pinned numerical stack. It reports the best energy found, the reference energy, the absolute difference, and accepted moves. To save an optional convergence plot, pass a path: `python examples/basin_hopping_lj13.py --plot convergence.png`.

Reference LJ13 minimum: `-44.326801 epsilon`. Lengths are in sigma, energies in epsilon, and reduced temperature is k_B T / epsilon. The heuristic search can find a local minimum and does not guarantee discovery of the global minimum for every random start or finite run length.

## Theory and citation

See [docs/theory.md](docs/theory.md) for the model, equations, assumptions, and references. The reference N=13 minimum is listed by the [Cambridge Cluster Database](https://doye.chem.ox.ac.uk/jon/structures/LJ/tables.150.html). Basin hopping is described by Wales and Doye, [J. Phys. Chem. A 101, 5111-5116 (1997)](https://doi.org/10.1021/jp970984n).

## License

MIT, see [LICENSE](LICENSE).
