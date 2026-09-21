import numpy as np
import pytest

from crystallite.electrostatics import Electrostatics
from crystallite.grid import Grid
from crystallite.verification.electrostatics import (
    charged_disk_field,
    polarized_disk_fields,
    smoothed_disk,
)

RADIUS = 0.1


def _radius(grid):
    return np.hypot(np.asarray(grid.x[0]) - 0.5, np.asarray(grid.x[1]) - 0.5)[:, :, 0]


@pytest.mark.parametrize("permittivity", [1.0, 2.5])
def test_charged_cylinder_agrees_with_the_solver(permittivity):
    grid = Grid(shape=(256, 256, 1))
    charge = 1.7 * smoothed_disk(grid, RADIUS)
    numeric = np.asarray(Electrostatics(grid, permittivity).solve(charge_density=charge).electric_field)
    reference = charged_disk_field(grid, RADIUS, 1.7, permittivity)
    assert np.sqrt(((numeric - reference) ** 2).mean()) < 1e-3 * np.abs(reference).max()


def test_isolated_charged_cylinder_has_the_textbook_profile():
    grid = Grid(shape=(128, 128, 1))
    a, permittivity = 0.02, 2.0
    field = charged_disk_field(grid, a, 1.0, permittivity)
    x = np.asarray(grid.x[0])[:, 0, 0] - 0.5
    row = np.argmin(np.abs(np.asarray(grid.x[1])[0, :, 0] - 0.5))
    inside, outside = np.argmin(np.abs(x - 0.5 * a)), np.argmin(np.abs(x - 3.0 * a))
    assert field[0][inside, row, 0] == pytest.approx(x[inside] / (2 * permittivity), rel=0.05)
    assert field[0][outside, row, 0] == pytest.approx(a**2 / (2 * permittivity * x[outside]), rel=0.05)


@pytest.mark.parametrize("permittivity", [1.0, 2.5])
def test_polarized_cylinder_agrees_with_the_solver_away_from_the_surface(permittivity):
    grid = Grid(shape=(256, 256, 1))
    polarization = np.zeros((3,) + grid.shape, dtype=np.float32)
    indicator = smoothed_disk(grid, RADIUS)
    polarization[0], polarization[1] = indicator, 0.4 * indicator
    solution = Electrostatics(grid, permittivity).solve(polarization=polarization)
    field, displacement = polarized_disk_fields(grid, RADIUS, [1.0, 0.4, 0.0], permittivity)
    away = np.abs(_radius(grid) - RADIUS) > 4.0 / 256
    for numeric, reference in ((solution.electric_field, field), (solution.displacement, displacement)):
        error = (np.asarray(numeric) - reference)[:2, :, :, 0][:, away]
        assert np.sqrt((error**2).mean()) < 1e-3 * np.abs(reference).max()


def test_polarized_interior_is_the_exact_periodic_value_not_the_isolated_one():
    grid = Grid(shape=(128, 128, 1))
    field, _ = polarized_disk_fields(grid, RADIUS, [1.0, 0.0, 0.0])
    inside = _radius(grid) < 0.5 * RADIUS
    interior = field[0][:, :, 0][inside]
    np.testing.assert_allclose(interior, interior[0], atol=1e-12)
    # circle in a square cell: S = (1 - f) I / 2 (isotropy by symmetry, trace by Parseval)
    assert interior[0] == pytest.approx(-0.5 * (1.0 - np.pi * RADIUS**2), abs=1e-12)
    assert abs(interior[0] + 0.5) > 0.01


def test_polarized_reference_needs_a_square_cell():
    with pytest.raises(ValueError, match="square"):
        polarized_disk_fields(Grid(shape=(32, 32, 1), lengths=(1.0, 2.0, 1.0)), RADIUS, [1.0, 0.0, 0.0])


def test_polarized_reference_has_zero_mean_field_and_scales_with_permittivity():
    grid = Grid(shape=(128, 128, 1))
    weak, weak_d = polarized_disk_fields(grid, RADIUS, [1.0, 0.4, 0.0], 1.0)
    strong, strong_d = polarized_disk_fields(grid, RADIUS, [1.0, 0.4, 0.0], 4.0)
    np.testing.assert_allclose(weak.mean(axis=(1, 2, 3)), 0.0, atol=1e-12)
    np.testing.assert_allclose(strong * 4.0, weak, atol=1e-12)
    inside = _radius(grid) < RADIUS
    # D = eps E + P in the disk, eps E outside: normal D is continuous, so the outside D equals eps E
    np.testing.assert_allclose(strong_d[:, ~inside[:, :, None]], 4.0 * strong[:, ~inside[:, :, None]], atol=1e-12)


def test_normal_displacement_and_tangential_field_are_continuous_at_the_surface():
    grid = Grid(shape=(1024, 1024, 1))
    field, displacement = polarized_disk_fields(grid, RADIUS, [1.0, 0.0, 0.0])
    x = np.asarray(grid.x[0])[:, 0, 0] - 0.5
    y = np.asarray(grid.x[1])[0, :, 0] - 0.5
    row, column = np.argmin(np.abs(y)), np.argmin(np.abs(x))
    just_in, just_out = np.argmin(np.abs(x - 0.99 * RADIUS)), np.argmin(np.abs(x - 1.01 * RADIUS))
    # along P: normal component of D continuous, of E it jumps by P/eps
    assert displacement[0][just_in, row, 0] == pytest.approx(displacement[0][just_out, row, 0], abs=0.03)
    assert field[0][just_out, row, 0] - field[0][just_in, row, 0] == pytest.approx(1.0, abs=0.05)
    # across P: tangential component of E continuous
    just_in_y, just_out_y = np.argmin(np.abs(y - 0.99 * RADIUS)), np.argmin(np.abs(y - 1.01 * RADIUS))
    assert field[0][column, just_in_y, 0] == pytest.approx(field[0][column, just_out_y, 0], abs=0.03)


def test_axial_polarization_makes_no_field():
    grid = Grid(shape=(64, 64, 1))
    field, displacement = polarized_disk_fields(grid, RADIUS, [0.0, 0.0, 0.6])
    np.testing.assert_allclose(field, 0.0, atol=1e-12)
    inside = _radius(grid) < RADIUS
    np.testing.assert_allclose(displacement[2][:, :, 0][inside], 0.6)
    np.testing.assert_allclose(displacement[2][:, :, 0][~inside], 0.0)
