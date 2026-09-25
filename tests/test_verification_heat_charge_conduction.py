import numpy as np
import pytest

from crystallite.grid import Grid
from crystallite.verification.charge_conduction import ChargeInclusionCase
from crystallite.verification.conduction import ConductionInclusionCase, disk_source_flux
from crystallite.verification.heat_conduction import HeatInclusionCase, heat_generation_flux

FIELD = [1.0, 0.3, 0.0]


@pytest.fixture
def grid():
    return Grid(shape=(64, 64, 1))


@pytest.mark.parametrize("contrast", [0.25, 4.0])
def test_domain_names_are_the_shared_core_under_heat_and_charge_vocabulary(grid, contrast):
    core = ConductionInclusionCase(grid, 1.5, 0.1, 0.1, contrast=contrast)
    heat = HeatInclusionCase(grid, 1.5, 0.1, 0.1, contrast=contrast)
    charge = ChargeInclusionCase(grid, 1.5, 0.1, 0.1, contrast=contrast)
    np.testing.assert_array_equal(
        heat.periodic_interior_thermal_driving_force(FIELD), core.periodic_interior_driving_force(FIELD)
    )
    np.testing.assert_array_equal(
        charge.periodic_interior_electric_field(FIELD), core.periodic_interior_driving_force(FIELD)
    )
    np.testing.assert_array_equal(
        heat.periodic_interior_heat_flux(FIELD), charge.periodic_interior_current_density(FIELD)
    )
    np.testing.assert_array_equal(heat.periodic_heat_flux(FIELD), core.periodic_flux_field(FIELD))
    np.testing.assert_array_equal(
        charge.periodic_current_density(FIELD), core.periodic_flux_field(FIELD)
    )
    np.testing.assert_array_equal(
        heat.periodic_thermal_driving_force(FIELD), charge.periodic_electric_field(FIELD)
    )


def test_flux_field_is_conductivity_times_field_and_the_interior_is_uniform(grid):
    case = HeatInclusionCase(grid, 2.0, 0.1, 0.1, contrast=0.25)
    driving = case.periodic_thermal_driving_force(FIELD)
    flux = case.periodic_heat_flux(FIELD)
    inside = np.asarray(case.elliptical_radius) < 1.0
    np.testing.assert_allclose(flux[:, ~inside[None].repeat(3, 0)[0]], 2.0 * driving[:, ~inside[None].repeat(3, 0)[0]])
    np.testing.assert_allclose(
        flux[0][inside], 0.25 * 2.0 * case.periodic_interior_thermal_driving_force(FIELD)[0]
    )
    np.testing.assert_allclose(
        case.periodic_interior_heat_flux(FIELD), 0.25 * 2.0 * case.periodic_interior_thermal_driving_force(FIELD)
    )


def test_a_perfect_conductor_has_zero_field_and_a_finite_uniform_interior_current(grid):
    case = ChargeInclusionCase(grid, 1.0, 0.1, 0.1, contrast=float("inf"))
    inside = np.asarray(case.elliptical_radius) < 1.0
    current = case.periodic_current_density(FIELD)
    interior = case.periodic_interior_current_density(FIELD)
    assert np.isfinite(current).all()
    for i in range(3):
        np.testing.assert_allclose(current[i][inside], interior[i])
    np.testing.assert_array_equal(case.periodic_interior_electric_field(FIELD), 0.0)
    assert interior[0] == pytest.approx(2.0 * FIELD[0] / (1.0 - np.pi * 0.1**2), rel=1e-4)


def test_heat_generation_names_the_shared_source_reference(grid):
    np.testing.assert_array_equal(
        heat_generation_flux(grid, 0.1, 1.0, thermal_conductivity=1.7),
        disk_source_flux(grid, 0.1, 1.0, conductivity=1.7),
    )
