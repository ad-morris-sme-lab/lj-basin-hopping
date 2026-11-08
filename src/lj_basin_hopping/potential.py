"""Reduced-unit Lennard-Jones energy and analytic gradient."""
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

def validate_positions(positions: ArrayLike) -> FloatArray:
    """Convert and validate an N by 3 coordinate array.

    Parameters
    ----------
    positions : array_like
        Cartesian coordinates in reduced length units (sigma).

    Returns
    -------
    numpy.ndarray
        Finite float64 array with shape (N, 3), where N is at least two.

    Raises
    ------
    ValueError
        If the shape is invalid or any coordinate is non-finite.
    """
    coords = np.asarray(positions, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] < 2:
        raise ValueError("positions must have shape (N, 3), N >= 2")
    if not np.all(np.isfinite(coords)):
        raise ValueError("positions must contain only finite coordinates")
    return coords

def lennard_jones_energy_gradient(positions: ArrayLike) -> tuple[float, FloatArray]:
    """Calculate total LJ energy and Cartesian gradient in reduced units.

    Parameters
    ----------
    positions : array_like
        N by 3 coordinates in units of sigma.

    Returns
    -------
    tuple of float and numpy.ndarray
        Energy in units of epsilon and gradient in epsilon per sigma.

    Raises
    ------
    ValueError
        For malformed coordinates, coincident particles, or float64 overflow.
    """
    coords = validate_positions(positions)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            delta = coords[:, None, :] - coords[None, :, :]
            distances = np.hypot(np.hypot(delta[..., 0], delta[..., 1]), delta[..., 2])
            i, j = np.triu_indices(len(coords), k=1)
            rij = distances[i, j]
            if np.any(rij == 0.0):
                raise ValueError("distinct particles must not occupy identical positions")
            inv6 = (1.0 / rij) ** 6
            inv12 = inv6**2
            energy = 4.0 * np.sum(inv12 - inv6)
            pair_factor = 24.0 * (inv6 - 2.0 * inv12) / rij**2
            pair_gradient = pair_factor[:, None] * delta[i, j]
            gradient = np.zeros_like(coords)
            np.add.at(gradient, i, pair_gradient)
            np.add.at(gradient, j, -pair_gradient)
    except FloatingPointError as exc:
        raise ValueError("energy or gradient exceeds the float64 range") from exc
    if not np.isfinite(energy) or not np.all(np.isfinite(gradient)):
        raise ValueError("energy or gradient exceeds the float64 range")
    return float(energy), gradient
