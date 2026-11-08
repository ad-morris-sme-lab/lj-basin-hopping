"""Unit tests for the LJ13 basin-hopping project."""
import numpy as np
import pytest
from lj_basin_hopping import basin_hopping, lennard_jones_energy_gradient
from lj_basin_hopping.potential import validate_positions

REFERENCE_LJ13 = -44.326801

def _icosahedral_seed() -> np.ndarray:
    """Build a central atom and twelve vertices of a regular icosahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = []
    for a in (-1.0, 1.0):
        for b in (-phi, phi):
            vertices.extend([[0.0, a, b], [a, b, 0.0], [b, 0.0, a]])
    shell = np.asarray(vertices, dtype=float)
    shell /= np.sqrt(1.0 + phi**2)
    return np.vstack((np.zeros(3), shell))

def test_geometry_validation_handles_bad_shape_and_nonfinite() -> None:
    """Malformed coordinate arrays are rejected."""
    assert validate_positions(np.zeros((2, 3))).shape == (2, 3)
    with pytest.raises(ValueError, match="shape"):
        validate_positions([[0.0, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        validate_positions([[0.0, 0.0, 0.0], [np.inf, 0.0, 0.0]])

def test_dimer_energy_and_zero_gradient_at_pair_minimum() -> None:
    """A dimer at r=2^(1/6) has energy -1 and zero gradient."""
    r_min = 2.0 ** (1.0 / 6.0)
    energy, gradient = lennard_jones_energy_gradient([[0, 0, 0], [r_min, 0, 0]])
    assert energy == pytest.approx(-1.0, abs=1e-12)
    assert np.linalg.norm(gradient) < 1e-10

def test_overlap_is_rejected() -> None:
    """Coincident particles do not silently produce an infinite energy."""
    with pytest.raises(ValueError, match="identical"):
        lennard_jones_energy_gradient([[0, 0, 0], [0, 0, 0]])

def test_analytic_gradient_matches_finite_difference() -> None:
    """The analytic derivative agrees with a central finite difference."""
    coords = np.array([[0., 0., 0.], [1.3, 0.2, 0.]])
    energy, gradient = lennard_jones_energy_gradient(coords)
    step = 1e-6
    plus, minus = coords.copy(), coords.copy()
    plus[1, 0] += step
    minus[1, 0] -= step
    e_plus = lennard_jones_energy_gradient(plus)[0]
    e_minus = lennard_jones_energy_gradient(minus)[0]
    assert np.isfinite(energy)
    assert gradient[1, 0] == pytest.approx((e_plus - e_minus) / (2 * step), rel=1e-6)

def test_gradient_is_translation_invariant() -> None:
    """Pair gradients sum to zero because total energy is translation invariant."""
    _, gradient = lennard_jones_energy_gradient([[0, 0, 0], [1.2, 0.1, 0]])
    assert np.allclose(gradient.sum(axis=0), 0.0, atol=1e-12)

def test_basin_hopping_reproducible_and_finds_lj13_reference() -> None:
    """Seeded LJ13 run reproducibly reaches the tabulated icosahedral minimum."""
    initial = _icosahedral_seed() * 1.8
    result_a = basin_hopping(initial, n_steps=12, seed=11, step_size=0.20)
    result_b = basin_hopping(initial, n_steps=12, seed=11, step_size=0.20)
    assert result_a.energy == pytest.approx(REFERENCE_LJ13, abs=2e-5)
    assert np.array_equal(result_a.best_energy_trace, result_b.best_energy_trace)
    assert np.array_equal(result_a.positions, result_b.positions)
    assert np.all(np.diff(result_a.best_energy_trace) <= 1e-12)

def test_failed_local_minimization_is_not_reported_as_a_valid_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed initial SciPy minimization raises instead of returning its point."""
    from scipy.optimize import OptimizeResult
    import lj_basin_hopping.optimizer as optimizer

    def failed_minimize(*args: object, **kwargs: object) -> OptimizeResult:
        return OptimizeResult(x=np.zeros(6), success=False, message="iteration limit")

    monkeypatch.setattr(optimizer, "minimize", failed_minimize)
    with pytest.raises(RuntimeError, match="did not converge"):
        basin_hopping([[0, 0, 0], [1.2, 0, 0]], n_steps=1)

def test_invalid_optimizer_parameters_rejected() -> None:
    """Invalid iteration counts and temperatures fail before optimization."""
    with pytest.raises(ValueError, match="n_steps"):
        basin_hopping(_icosahedral_seed(), n_steps=0)
    with pytest.raises(ValueError, match="temperature"):
        basin_hopping(_icosahedral_seed(), temperature=0.0)
