import numpy as np
import pytest

from crystallite.grid import Grid
from crystallite.magnetostatics import Magnetostatics
from crystallite.verification.magnetostatics import magnetized_disk_fields, smoothed_disk

RADIUS = 0.1


def test_magnetized_cylinder_agrees_with_the_solver_away_from_the_surface():
    grid = Grid(shape=(256, 256, 1))
    magnetization = np.zeros((3,) + grid.shape, dtype=np.float32)
    indicator = smoothed_disk(grid, RADIUS)
    magnetization[0], magnetization[1] = 1.5 * indicator, 0.4 * indicator
    solution = Magnetostatics(grid, permeability=2.0).solve(magnetization)
    field, flux = magnetized_disk_fields(grid, RADIUS, [1.5, 0.4, 0.0], permeability=2.0)
    radius = np.hypot(np.asarray(grid.x[0]) - 0.5, np.asarray(grid.x[1]) - 0.5)[:, :, 0]
    away = np.abs(radius - RADIUS) > 4.0 / 256
    for numeric, reference in ((solution.magnetic_field, field), (solution.flux_density, flux)):
        error = (np.asarray(numeric) - reference)[:2, :, :, 0][:, away]
        assert np.sqrt((error**2).mean()) < 1e-3 * np.abs(reference).max()


def test_interior_field_is_the_demagnetizing_field_and_flux_density_follows():
    grid = Grid(shape=(128, 128, 1))
    field, flux = magnetized_disk_fields(grid, RADIUS, [1.0, 0.0, 0.0], permeability=2.0)
    radius = np.hypot(np.asarray(grid.x[0]) - 0.5, np.asarray(grid.x[1]) - 0.5)[:, :, 0]
    inside = radius < 0.5 * RADIUS
    demagnetizing = -0.5 * (1.0 - np.pi * RADIUS**2)
    np.testing.assert_allclose(field[0][:, :, 0][inside], demagnetizing, atol=1e-12)
    np.testing.assert_allclose(flux[0][:, :, 0][inside], 2.0 * (demagnetizing + 1.0), atol=1e-12)
    np.testing.assert_allclose(field.mean(axis=(1, 2, 3)), 0.0, atol=1e-12)


def test_axial_magnetization_makes_no_field():
    grid = Grid(shape=(64, 64, 1))
    field, flux = magnetized_disk_fields(grid, RADIUS, [0.0, 0.0, 0.6])
    np.testing.assert_allclose(field, 0.0, atol=1e-12)
    assert flux[2].max() == pytest.approx(0.6)
