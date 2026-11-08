"""Basin-hopping global search with local L-BFGS-B minimization."""
from dataclasses import dataclass
import math
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from .potential import FloatArray, lennard_jones_energy_gradient, validate_positions

@dataclass(frozen=True)
class BasinHoppingResult:
    """Best minimum and search trace returned by basin hopping.

    Attributes
    ----------
    energy : float
        Best energy in units of epsilon.
    positions : numpy.ndarray
        Best coordinates in units of sigma, shape (N, 3).
    best_energy_trace : numpy.ndarray
        Best-so-far energy after initialization and every hop.
    accepted_moves : int
        Number of proposed local minima accepted as the current state.
    """
    energy: float
    positions: FloatArray
    best_energy_trace: NDArray[np.float64]
    accepted_moves: int

def _local_minimum(positions: FloatArray, max_iterations: int) -> tuple[float, FloatArray]:
    """Minimize one configuration and remove center-of-mass drift."""
    atom_count = len(positions)
    def objective(flat: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        energy, gradient = lennard_jones_energy_gradient(flat.reshape(atom_count, 3))
        return energy, gradient.ravel()
    result = minimize(objective, positions.ravel(), method="L-BFGS-B", jac=True,
                      options={"maxiter": max_iterations, "ftol": 1e-12, "gtol": 1e-7})
    coords = result.x.reshape(atom_count, 3)
    coords -= np.mean(coords, axis=0)
    energy, _ = lennard_jones_energy_gradient(coords)
    if not np.isfinite(energy):
        raise RuntimeError("local minimizer returned a non-finite energy")
    return energy, coords

def basin_hopping(
    initial_positions: ArrayLike,
    n_steps: int = 100,
    temperature: float = 1.0,
    step_size: float = 0.35,
    seed: int = 0,
    local_max_iterations: int = 300,
) -> BasinHoppingResult:
    """Search for a low-energy cluster using basin hopping.

    Parameters
    ----------
    initial_positions : array_like
        Initial N by 3 coordinates in reduced length units.
    n_steps : int, default=100
        Number of perturbed configurations to locally minimize.
    temperature : float, default=1.0
        Metropolis acceptance temperature in reduced energy units.
    step_size : float, default=0.35
        Standard deviation of independent coordinate displacements in sigma.
    seed : int, default=0
        Seed for the NumPy random generator.
    local_max_iterations : int, default=300
        Maximum L-BFGS-B iterations for each local minimization.

    Returns
    -------
    BasinHoppingResult
        Lowest minimum found, its coordinates, best-so-far trace, and accepts.

    Raises
    ------
    ValueError
        If the initial geometry or search parameters are invalid.
    """
    positions = validate_positions(initial_positions).copy()
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if isinstance(local_max_iterations, bool) or not isinstance(local_max_iterations, int) or local_max_iterations < 1:
        raise ValueError("local_max_iterations must be a positive integer")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and positive")
    rng = np.random.default_rng(seed)
    current_energy, current_positions = _local_minimum(positions, local_max_iterations)
    best_energy, best_positions = current_energy, current_positions.copy()
    trace = [best_energy]
    accepted_moves = 0
    for _ in range(n_steps):
        proposal = current_positions + rng.normal(0.0, step_size, current_positions.shape)
        proposal -= np.mean(proposal, axis=0)
        proposal_energy, proposal_positions = _local_minimum(proposal, local_max_iterations)
        delta_energy = proposal_energy - current_energy
        if delta_energy <= 0.0 or rng.random() < math.exp(-delta_energy / temperature):
            current_energy, current_positions = proposal_energy, proposal_positions
            accepted_moves += 1
        if proposal_energy < best_energy:
            best_energy, best_positions = proposal_energy, proposal_positions.copy()
        trace.append(best_energy)
    return BasinHoppingResult(best_energy, best_positions, np.asarray(trace), accepted_moves)
