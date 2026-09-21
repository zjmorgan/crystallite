import numpy as np
import pytest

from crystallite.electrostatics import Electrostatics
from crystallite.grid import Grid
from crystallite.magnetostatics import Magnetostatics
from crystallite.spectral.short_range import DifferentialOperators


@pytest.fixture
def grid():
    return Grid(shape=(64, 64, 1), lengths=(1.0, 1.0, 1.0))


def _magnetization(grid):
    x, y = np.asarray(grid.x[0]), np.asarray(grid.x[1])
    disk = np.where(np.hypot(x - 0.5, y - 0.5) < 0.15, 1.0, 0.0).astype(np.float32)
    disk = np.real(np.asarray(grid.ifft(grid.fft(disk) * grid.lanczos_filter)))
    m = np.zeros((3,) + grid.shape, dtype=np.float32)
    m[0], m[1] = disk, 0.4 * disk
    return m


def test_flux_density_is_solenoidal_and_the_field_curl_free(grid):
    magnetization = _magnetization(grid)
    solution = Magnetostatics(grid, permeability=3.0).solve(magnetization)
    ops = DifferentialOperators(grid)
    divergence = np.real(np.asarray(grid.ifft(ops.div(grid.fft(np.asarray(solution.flux_density))))))
    field = np.asarray(solution.magnetic_field)
    curl = np.real(np.asarray(grid.ifft(ops.curl(grid.fft(field))[2])))
    assert np.abs(divergence).max() < 1e-4 * 3.0 * np.abs(magnetization).max() * 64
    assert np.abs(curl).max() < 1e-4 * np.abs(field).max() * 64


def test_flux_density_is_permeability_times_field_plus_magnetization(grid):
    magnetization = _magnetization(grid)
    solution = Magnetostatics(grid, permeability=3.0).solve(magnetization)
    np.testing.assert_allclose(
        np.asarray(solution.flux_density),
        3.0 * (np.asarray(solution.magnetic_field) + magnetization), atol=1e-5,
    )


def test_is_electrostatics_with_unit_permittivity_and_polarization_as_magnetization(grid):
    magnetization = _magnetization(grid)
    magnetic = Magnetostatics(grid).solve(magnetization)
    electric = Electrostatics(grid, 1.0).solve(polarization=magnetization)
    np.testing.assert_allclose(
        np.asarray(magnetic.magnetic_field), np.asarray(electric.electric_field), atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(magnetic.flux_density), np.asarray(electric.displacement), atol=1e-6
    )


def test_a_uniform_magnetization_makes_no_field_and_a_far_field_is_the_mean(grid):
    uniform = np.zeros((3,) + grid.shape, dtype=np.float32)
    uniform[0] = 0.7
    solution = Magnetostatics(grid).solve(uniform)
    np.testing.assert_allclose(np.asarray(solution.magnetic_field), 0.0, atol=1e-6)
    applied = Magnetostatics(grid).solve(_magnetization(grid), far_field=[0.3, 0.0, 0.0])
    np.testing.assert_allclose(
        np.asarray(applied.magnetic_field).mean(axis=(1, 2, 3)), [0.3, 0.0, 0.0], atol=1e-6
    )


def test_rejects_a_nonpositive_permeability(grid):
    with pytest.raises(ValueError, match="positive"):
        Magnetostatics(grid, permeability=0.0)
