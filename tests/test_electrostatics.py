import numpy as np
import pytest

from crystallite.conduction import SteadyConduction
from crystallite.electrostatics import Electrostatics
from crystallite.grid import Grid
from crystallite.spectral.short_range import DifferentialOperators


@pytest.fixture
def grid():
    return Grid(shape=(64, 64, 1), lengths=(1.0, 1.0, 1.0))


def _disk(grid, radius=0.15):
    x, y = np.asarray(grid.x[0]), np.asarray(grid.x[1])
    indicator = np.where(np.hypot(x - 0.5, y - 0.5) < radius, 1.0, 0.0).astype(np.float32)
    return np.real(np.asarray(grid.ifft(grid.fft(indicator) * grid.lanczos_filter)))


def _divergence(grid, vector):
    return np.real(np.asarray(grid.ifft(DifferentialOperators(grid).div(grid.fft(vector)))))


def _curl_z(grid, vector):
    ops = DifferentialOperators(grid)
    return np.real(np.asarray(grid.ifft(ops.curl(grid.fft(vector))[2])))


def test_gauss_law_holds_with_the_mean_charge_dropped(grid):
    rho = _disk(grid)
    solution = Electrostatics(grid, 2.5).solve(charge_density=rho)
    residual = _divergence(grid, np.asarray(solution.displacement)) - (rho - rho.mean())
    assert np.abs(residual).max() < 1e-4 * np.abs(rho).max() * 64


def test_the_field_is_curl_free_and_scales_inversely_with_permittivity(grid):
    rho = _disk(grid)
    weak = Electrostatics(grid, 1.0).solve(charge_density=rho)
    strong = Electrostatics(grid, 4.0).solve(charge_density=rho)
    field = np.asarray(weak.electric_field)
    assert np.abs(_curl_z(grid, field)).max() < 1e-4 * np.abs(field).max() * 64
    np.testing.assert_allclose(np.asarray(strong.electric_field) * 4.0, field, atol=1e-6)
    np.testing.assert_allclose(np.asarray(strong.displacement), np.asarray(weak.displacement), atol=1e-6)


def test_polarization_acts_through_its_bound_charge_and_displacement_is_solenoidal(grid):
    polarization = np.zeros((3,) + grid.shape, dtype=np.float32)
    polarization[0] = _disk(grid)
    polarization[1] = 0.4 * _disk(grid)
    solution = Electrostatics(grid, 2.0).solve(polarization=polarization)
    displacement = np.asarray(solution.displacement)
    # no free charge: div D = 0, i.e. div(eps E) = -div P
    assert np.abs(_divergence(grid, displacement)).max() < 1e-4 * np.abs(polarization).max() * 64
    # bound charge only: same field as the equivalent free charge -div P
    bound = -_divergence(grid, polarization)
    equivalent = Electrostatics(grid, 2.0).solve(charge_density=bound)
    np.testing.assert_allclose(
        np.asarray(solution.electric_field), np.asarray(equivalent.electric_field),
        atol=1e-4 * np.abs(np.asarray(solution.electric_field)).max(),
    )


def test_a_uniform_polarization_has_no_field_and_displacement_equals_it(grid):
    polarization = np.zeros((3,) + grid.shape, dtype=np.float32)
    polarization[0], polarization[2] = 0.7, 0.3
    solution = Electrostatics(grid).solve(polarization=polarization)
    np.testing.assert_allclose(np.asarray(solution.electric_field), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(solution.displacement), polarization, atol=1e-6)


def test_a_far_field_is_the_mean_field(grid):
    rho = _disk(grid)
    far = [0.3, -0.2, 0.0]
    solution = Electrostatics(grid, 3.0).solve(charge_density=rho, far_field=far)
    field = np.asarray(solution.electric_field)
    np.testing.assert_allclose(field.mean(axis=(1, 2, 3)), far, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(solution.displacement).mean(axis=(1, 2, 3)), 3.0 * np.array(far), atol=1e-5
    )


def test_a_uniform_charge_is_neutralized_not_diverged(grid):
    solution = Electrostatics(grid).solve(charge_density=np.full(grid.shape, 2.0))
    np.testing.assert_allclose(np.asarray(solution.electric_field), 0.0, atol=1e-6)


def test_matches_steady_conduction_with_permittivity_as_conductivity_and_flux_as_displacement(grid):
    # a dielectric problem is steady conduction: kappa <-> eps, source <-> rho, flux <-> D
    epsilon, rho = 2.5, _disk(grid)
    electro = Electrostatics(grid, epsilon).solve(charge_density=rho)
    conduction = SteadyConduction(grid, epsilon).solve([0.0, 0.0, 0.0], source=rho, tol=1e-6)
    assert conduction.converged
    peak = np.abs(np.asarray(electro.displacement)).max()
    np.testing.assert_allclose(
        np.asarray(conduction.flux)[:2], np.asarray(electro.displacement)[:2], atol=1e-4 * peak
    )


@pytest.mark.parametrize("permittivity", [0.0, -1.0])
def test_rejects_a_nonpositive_permittivity(grid, permittivity):
    with pytest.raises(ValueError, match="positive"):
        Electrostatics(grid, permittivity)


def test_rejects_malformed_sources(grid):
    solver = Electrostatics(grid)
    with pytest.raises(ValueError, match="charge_density"):
        solver.solve(charge_density=np.ones((4, 4, 1)))
    with pytest.raises(ValueError, match="polarization"):
        solver.solve(polarization=np.ones(grid.shape))
    with pytest.raises(ValueError, match="far_field"):
        solver.solve(far_field=[1.0, 0.0])
