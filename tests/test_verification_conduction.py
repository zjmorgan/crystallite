import numpy as np
import pytest

from crystallite.conduction import SteadyConduction
from crystallite.grid import Grid
from crystallite.verification.conduction import (
    ConductionInclusionCase,
    circular_inhomogeneity_driving_force,
    disk_source_flux,
    elliptical_perturbation,
    interior_driving_force,
    isolated_depolarization_tensor,
    periodic_depolarization_tensor,
)

ANISOTROPIC = np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("contrast", [0.0, 0.3, 4.0, float("inf")])
def test_isolated_circle_satisfies_both_interface_conditions(contrast):
    a, field = 0.1, (1.0, 0.4)
    theta = np.linspace(0.1, 6.2, 9)
    inner, outer = 0.999999 * a, 1.000001 * a
    x_in, y_in = inner * np.cos(theta), inner * np.sin(theta)
    x_out, y_out = outer * np.cos(theta), outer * np.sin(theta)
    fin = np.array(circular_inhomogeneity_driving_force(x_in, y_in, a, contrast, field))
    fout = np.array(circular_inhomogeneity_driving_force(x_out, y_out, a, contrast, field))
    normal = np.array([np.cos(theta), np.sin(theta)])
    tangent = np.array([-np.sin(theta), np.cos(theta)])
    # potential continuity <-> tangential X continuous
    np.testing.assert_allclose(np.sum(fin * tangent, 0), np.sum(fout * tangent, 0), atol=1e-5)
    if contrast != float("inf"):
        # normal flux continuity: kappa_1 X_n(in) = kappa_0 X_n(out)
        np.testing.assert_allclose(
            contrast * np.sum(fin * normal, 0), np.sum(fout * normal, 0), atol=1e-5
        )
    else:
        np.testing.assert_allclose(fin, 0.0, atol=1e-12)


@pytest.mark.parametrize("contrast", [0.0, 0.3, 4.0])
def test_isolated_interior_matches_the_closed_form_for_a_circle(contrast):
    interior = interior_driving_force(
        contrast, isolated_depolarization_tensor(1.0, 0.1, 0.1), [1.0, 0.4, 0.0]
    )
    np.testing.assert_allclose(interior, np.array([1.0, 0.4, 0.0]) * 2.0 / (1.0 + contrast))


def test_isolated_isotropic_ellipse_is_the_classical_depolarization():
    s = isolated_depolarization_tensor(3.0, 0.06, 0.12)
    np.testing.assert_allclose(np.diag(s)[:2], [0.12 / 0.18, 0.06 / 0.18], atol=1e-12)
    assert abs(s[0, 1]) < 1e-12


def test_isolated_anisotropic_matches_the_coordinate_stretch_oracle():
    # y' = y sqrt(k1/k2) makes the matrix isotropic and the circle an ellipse
    k1, k2, contrast = 2.0, 0.5, 0.2
    s = np.sqrt(k1 / k2)
    stretched = np.diag([s / (1 + s), 1 / (1 + s)])
    mean = np.array([1.0, 0.7])
    prime = np.linalg.solve(np.eye(2) - (1 - contrast) * stretched, mean * [1.0, 1.0 / s])
    expected = prime * [1.0, s]
    interior = interior_driving_force(
        contrast, isolated_depolarization_tensor(np.diag([k1, k2, 1.0]), 0.1, 0.1), [1.0, 0.7, 0.0]
    )
    np.testing.assert_allclose(interior[:2], expected, atol=1e-10)


@pytest.mark.parametrize("conductivity", [1.0, ANISOTROPIC])
@pytest.mark.parametrize("a, b", [(0.1, 0.1), (0.06, 0.12)])
def test_depolarization_traces_are_exact(conductivity, a, b):
    # tr S = 1 isolated; for the periodic array Parseval gives tr S = 1 - area fraction
    isolated = isolated_depolarization_tensor(conductivity, a, b)
    periodic = periodic_depolarization_tensor(conductivity, a, b, 1.0, 1.0, n_modes=256)
    assert np.trace(isolated[:2, :2]) == pytest.approx(1.0, abs=1e-12)
    assert np.trace(periodic[:2, :2]) == pytest.approx(1.0 - np.pi * a * b, abs=2e-4)


@pytest.mark.parametrize("conductivity", [1.0, ANISOTROPIC])
def test_periodic_tensor_approaches_the_isolated_one_for_a_small_inclusion(conductivity):
    a = 0.01
    np.testing.assert_allclose(
        periodic_depolarization_tensor(conductivity, a, a, 1.0, 1.0, n_modes=1024),
        isolated_depolarization_tensor(conductivity, a, a), atol=2e-3,
    )


def test_a_perfect_conductor_has_no_interior_field_and_is_the_large_contrast_limit():
    s = periodic_depolarization_tensor(1.0, 0.1, 0.1, n_modes=256)
    np.testing.assert_array_equal(interior_driving_force(float("inf"), s, [1.0, 0.2, 0.0]), 0.0)
    np.testing.assert_allclose(
        interior_driving_force(1e8, s, [1.0, 0.2, 0.0]), 0.0, atol=1e-7
    )


def test_conductor_flux_is_undefined_and_the_case_requires_a_2d_grid():
    grid = Grid(shape=(16, 16, 1))
    case = ConductionInclusionCase(grid, 1.0, 0.1, 0.1, contrast=float("inf"))
    with pytest.raises(ValueError, match="undefined"):
        case.periodic_interior_flux([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="2D"):
        ConductionInclusionCase(Grid(shape=(8, 8, 8)), 1.0, 0.1, 0.1)
    with pytest.raises(ValueError, match="isotropic"):
        ConductionInclusionCase(grid, ANISOTROPIC, 0.1, 0.1).periodic_exterior_field([1.0, 0.0, 0.0])


@pytest.mark.parametrize("contrast, tolerance", [(0.1, 0.04), (4.0, 0.04)])
def test_periodic_interior_and_exterior_agree_with_the_numeric_solve(contrast, tolerance):
    grid = Grid(shape=(256, 256, 1))
    case = ConductionInclusionCase(grid, 1.0, 0.1, 0.1, contrast=contrast)
    field = [1.0, 0.3, 0.0]
    solution = case.solver().solve(field, tol=1e-6, max_iterations=3000)
    assert solution.converged
    driving = np.asarray(solution.driving_force)
    rho = np.asarray(case.elliptical_radius)
    interior = case.periodic_interior_driving_force(field)
    for i in range(2):
        assert float(driving[i][rho < 0.5].mean()) == pytest.approx(interior[i], rel=tolerance)
    exterior = case.periodic_exterior_field(field)
    error = (driving - exterior)[:2][:, rho > 1.4]
    assert np.sqrt((error**2).mean()) < 0.01


def test_anisotropic_ellipse_agrees_with_the_numeric_tensor_solve():
    grid = Grid(shape=(256, 256, 1))
    case = ConductionInclusionCase(grid, ANISOTROPIC, 0.12, 0.07, contrast=0.3)
    field = [1.0, -0.4, 0.0]
    solution = case.solver().solve(field, tol=1e-6, max_iterations=3000)
    assert solution.converged
    driving = np.asarray(solution.driving_force)
    inside = np.asarray(case.elliptical_radius) < 0.5
    interior = case.periodic_interior_driving_force(field)
    for i in range(2):
        assert float(driving[i][inside].mean()) == pytest.approx(interior[i], rel=0.01)
    # the isolated value is measurably different: the periodic correction is real
    assert abs(case.isolated_interior_driving_force(field)[0] - interior[0]) > 0.01


def test_heat_generating_cylinder_agrees_with_the_spectral_reference():
    grid = Grid(shape=(256, 256, 1))
    x, y = np.asarray(grid.x[0]), np.asarray(grid.x[1])
    radius = 0.1
    indicator = np.where(np.hypot(x - 0.5, y - 0.5) < radius, 1.0, 0.0).astype(np.float32)
    indicator = np.real(np.asarray(grid.ifft(grid.fft(indicator) * grid.lanczos_filter)))
    solution = SteadyConduction(grid, 1.7).solve([0.0, 0.0, 0.0], source=indicator)
    assert solution.converged
    reference = disk_source_flux(grid, radius, 1.0, conductivity=1.7)
    error = np.asarray(solution.flux) - reference
    assert np.sqrt((error**2).mean()) < 1e-3 * np.abs(reference).max()


def test_isolated_heat_flux_is_radial_with_the_textbook_profile():
    # a small cylinder in a large cell approaches phi r/2 inside, phi a^2/(2r) outside
    grid = Grid(shape=(128, 128, 1))
    a = 0.02
    flux = disk_source_flux(grid, a, 1.0)
    x = np.asarray(grid.x[0])[:, 0, 0] - 0.5
    row = np.argmin(np.abs(np.asarray(grid.x[1])[0, :, 0] - 0.5))
    inside = np.argmin(np.abs(x - 0.5 * a))
    outside = np.argmin(np.abs(x - 3.0 * a))
    assert flux[0][inside, row, 0] == pytest.approx(0.5 * x[inside], rel=0.05)
    assert flux[0][outside, row, 0] == pytest.approx(a**2 / (2.0 * x[outside]), rel=0.05)
    assert abs(flux[1][outside, row, 0]) < 1e-6


@pytest.mark.parametrize("contrast", [0.0, 0.3, 5.0, float("inf")])
def test_elliptical_perturbation_reduces_to_the_circle_dipole(contrast):
    a, field = 0.1, np.array([1.0, 0.4])
    x, y = np.array([0.15, -0.12, 0.0, 0.05]), np.array([0.02, 0.09, -0.13, 0.2])
    c = -1.0 if contrast == float("inf") else (1.0 - contrast) / (1.0 + contrast)
    dipole = c * a**2 * field
    r2, dot = x**2 + y**2, dipole[0] * x + dipole[1] * y
    circle = np.array([dipole[0] / r2 - 2 * dot * x / r2**2, dipole[1] / r2 - 2 * dot * y / r2**2])
    ellipse = elliptical_perturbation(x, y, a, a * (1 - 1e-7), contrast, field)
    np.testing.assert_allclose(np.array(ellipse), circle, atol=1e-6)


@pytest.mark.parametrize("contrast", [0.0, 0.3, 5.0])
@pytest.mark.parametrize("a, b", [(0.12, 0.05), (0.05, 0.12)])
def test_elliptical_perturbation_satisfies_both_interface_conditions(contrast, a, b):
    field = np.array([1.0, 0.5])
    theta = np.linspace(0.2, 6.0, 7)
    x, y = a * np.cos(theta) * (1 + 1e-7), b * np.sin(theta) * (1 + 1e-7)
    dx, dy = elliptical_perturbation(x, y, a, b, contrast, field)
    outside = field[:, None] + np.array([dx, dy])
    interior = interior_driving_force(
        contrast, isolated_depolarization_tensor(1.0, a, b), [*field, 0.0]
    )[:2]
    normal = np.array([x / a**2, y / b**2])
    normal /= np.linalg.norm(normal, axis=0)
    tangent = np.array([-normal[1], normal[0]])
    np.testing.assert_allclose(
        np.sum(outside * tangent, 0), np.sum(interior[:, None] * tangent, 0), atol=1e-5
    )
    np.testing.assert_allclose(
        np.sum(outside * normal, 0), contrast * np.sum(interior[:, None] * normal, 0), atol=1e-5
    )


def test_a_thin_hole_is_a_crack_with_the_classical_tip_singularity():
    # isolated slit of half-length a under a field across it: X_2(x) = X x / sqrt(x^2 - a^2) on the axis
    a, b = 0.1, 1e-4
    x = np.array([0.11, 0.13, 0.2, 0.4])
    dx, dy = elliptical_perturbation(x, 0.0 * x, a, b, 0.0, [0.0, 1.0])
    np.testing.assert_allclose(1.0 + dy, x / np.sqrt(x**2 - a**2), rtol=1e-3)
