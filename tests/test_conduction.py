import numpy as np
import pytest

from crystallite.conduction import SteadyConduction
from crystallite.elastic_deformation import ElasticDeformation
from crystallite.grid import Grid


def _smooth(grid, field):
    """Band-limit a field (the discrete operator is only exact on it)."""
    return np.real(np.asarray(grid.ifft(grid.fft(np.asarray(field, dtype=np.float32)) * grid.lanczos_filter)))


def _disk(grid, radius, inside, outside):
    x, y = np.asarray(grid.x[0]), np.asarray(grid.x[1])
    r = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    return _smooth(grid, np.where(r < radius, inside, outside)), r


@pytest.fixture
def grid():
    return Grid(shape=(64, 64, 1), lengths=(1.0, 1.0, 1.0))


def test_homogeneous_medium_needs_no_iterations_and_flux_is_kappa_times_field(grid):
    solution = SteadyConduction(grid, 2.5).solve([1.0, 0.5, 0.0])
    assert solution.converged and solution.iterations == 0
    np.testing.assert_allclose(np.asarray(solution.flux)[0], 2.5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(solution.flux)[1], 1.25, atol=1e-6)


def test_anisotropic_homogeneous_flux_follows_the_tensor(grid):
    kappa = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 1.0]])
    field = np.array([1.0, -0.4, 0.0])
    solution = SteadyConduction(grid, kappa).solve(field)
    q = np.asarray(solution.flux)
    for i in range(3):
        np.testing.assert_allclose(q[i], kappa[i] @ field, atol=1e-6)


def test_reference_green_converts_a_far_field_flux_to_a_driving_force(grid):
    kappa = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 1.0]])
    solver = SteadyConduction(grid, kappa)
    field = np.asarray(solver.reference_green.apply_compliance(np.array([1.0, 0.0, 0.0])))
    np.testing.assert_allclose(kappa @ field, [1.0, 0.0, 0.0], atol=1e-6)


@pytest.mark.parametrize("contrast", [0.1, 4.0])
def test_disk_inclusion_matches_the_mode_iii_elastic_solver(grid, contrast):
    # antiplane elasticity is this problem: sigma_i3 = mu (X_bar_i + d_i u_3), mu <-> kappa
    kappa, _ = _disk(grid, 0.2, contrast, 1.0)
    conduction = SteadyConduction(grid, kappa, reference_conductivity=1.0).solve(
        [1.0, 0.0, 0.0], tol=1e-6, max_iterations=3000
    )
    strain = np.zeros((3, 3))
    strain[0, 2] = strain[2, 0] = 0.5
    elastic = ElasticDeformation(
        grid, lame_lambda=0.0 * kappa, lame_mu=kappa, reference_lame_lambda=0.0,
        reference_lame_mu=1.0,
    ).solve(strain, tol=1e-6, max_iterations=3000)
    assert conduction.converged and elastic.converged
    np.testing.assert_allclose(
        np.asarray(conduction.flux)[0], np.asarray(elastic.stress)[0, 2], atol=2e-4
    )
    np.testing.assert_allclose(
        np.asarray(conduction.flux)[1], np.asarray(elastic.stress)[1, 2], atol=2e-4
    )


def test_flux_is_divergence_free_and_its_mean_is_the_effective_conductivity(grid):
    kappa, _ = _disk(grid, 0.2, 5.0, 1.0)
    solver = SteadyConduction(grid, kappa, reference_conductivity=1.0)
    solution = solver.solve([1.0, 0.0, 0.0], tol=1e-6, max_iterations=3000)
    divergence = np.asarray(solver._divergence(solution.flux))
    assert np.abs(divergence).max() < 1e-3 * np.abs(np.asarray(solution.flux)).max() * 64
    # mean driving force is the prescribed one; a conducting inclusion raises the mean flux
    np.testing.assert_allclose(np.asarray(solution.driving_force)[0].mean(), 1.0, atol=1e-5)
    assert np.asarray(solution.flux)[0].mean() > 1.0


def test_source_is_balanced_by_the_divergence_of_the_flux(grid):
    source, _ = _disk(grid, 0.15, 1.0, 0.0)
    solver = SteadyConduction(grid, 1.7)
    solution = solver.solve([0.0, 0.0, 0.0], source=source, tol=1e-6)
    assert solution.converged
    balance = np.asarray(solver._divergence(solution.flux)) - (source - source.mean())
    assert np.abs(balance).max() < 1e-3 * np.abs(source).max()
    np.testing.assert_allclose(np.asarray(solution.driving_force).mean(axis=(1, 2, 3)), 0.0, atol=1e-6)


def test_a_uniform_source_is_dropped_not_diverged(grid):
    solution = SteadyConduction(grid, 1.0).solve(
        [0.0, 0.0, 0.0], source=np.full(grid.shape, 3.0)
    )
    assert solution.converged and solution.iterations == 0


def test_heterogeneous_conductivity_needs_a_reference(grid):
    kappa, _ = _disk(grid, 0.2, 5.0, 1.0)
    with pytest.raises(ValueError, match="reference"):
        SteadyConduction(grid, kappa)


@pytest.mark.parametrize("bad", [np.ones((3, 4)), np.ones((5, 5, 5)), np.ones((8, 8, 1))])
def test_rejects_a_malformed_conductivity(grid, bad):
    with pytest.raises(ValueError):
        SteadyConduction(grid, bad, reference_conductivity=1.0)


def test_rejects_a_malformed_load(grid):
    solver = SteadyConduction(grid, 1.0)
    with pytest.raises(ValueError, match="macro_field"):
        solver.solve([1.0, 0.0])
    with pytest.raises(ValueError, match="source"):
        solver.solve([1.0, 0.0, 0.0], source=np.ones((4, 4, 1)))
    with pytest.raises(ValueError, match="initial_potential"):
        solver.solve([1.0, 0.0, 0.0], initial_potential=np.ones((4, 4, 1)))
