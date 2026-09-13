import numpy as np
import pytest

from crystallite.grid import Grid
from crystallite.verification import HoleInPlateCase
from crystallite.verification.elastic_deformation import (
    cartesian_to_polar_stress,
    _tension_polar_stress,
    _inhomogeneity_polar_stress,
    _inhomogeneity_interior_stress,
    _isotropic_compliance_apply,
    _isotropic_stress_apply,
    _equivalent_eigenstrain,
    _disk_fourier_transform,
)


def test_tension_analytic_field_is_traction_free_at_the_hole_boundary():
    a = 0.15
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    sigma_rr, _, sigma_r_theta = _tension_polar_stress(a, theta, a, 2.0)
    np.testing.assert_allclose(sigma_rr, 0.0, atol=1e-10)
    np.testing.assert_allclose(sigma_r_theta, 0.0, atol=1e-10)


def test_tension_analytic_hoop_stress_matches_known_concentration_factors():
    a, magnitude = 0.15, 2.0
    case = HoleInPlateCase(
        Grid(shape=(4, 4, 1)), matrix_lame_lambda=1.0, matrix_lame_mu=0.7,
        hole_radius=a,
    )
    max_hoop = case.hoop_stress_at_hole(np.pi / 2.0, "tension", magnitude)
    min_hoop = case.hoop_stress_at_hole(0.0, "tension", magnitude)
    assert max_hoop == pytest.approx(3.0 * magnitude)
    assert min_hoop == pytest.approx(-1.0 * magnitude)


def test_shear_analytic_hoop_stress_matches_known_concentration_factor():
    a, magnitude = 0.15, 2.0
    case = HoleInPlateCase(
        Grid(shape=(4, 4, 1)), matrix_lame_lambda=1.0, matrix_lame_mu=0.7,
        hole_radius=a,
    )
    hoop = case.hoop_stress_at_hole(np.pi / 4.0, "shear", magnitude)
    assert abs(hoop) == pytest.approx(4.0 * magnitude, rel=1e-6)


def test_cartesian_to_polar_stress_round_trips_a_uniform_state():
    theta = np.linspace(-np.pi, np.pi, 37)
    sigma_xx, sigma_yy, sigma_xy = 0.01, 0.0, 0.0
    sigma_rr, sigma_tt, _ = cartesian_to_polar_stress(sigma_xx, sigma_yy, sigma_xy, theta)
    np.testing.assert_allclose(sigma_rr, sigma_xx * np.cos(theta) ** 2, atol=1e-12)
    np.testing.assert_allclose(sigma_tt, sigma_xx * np.sin(theta) ** 2, atol=1e-12)


def test_solve_matches_kirsch_stress_concentration_for_tension_compression_and_shear():
    grid = Grid(shape=(256, 256, 1), lengths=(1.0, 1.0, 1.0))
    dx = grid.spacing[0]
    hole_radius = 0.1
    case = HoleInPlateCase(
        grid, matrix_lame_lambda=1.0, matrix_lame_mu=0.7, hole_radius=hole_radius,
        contrast=1.0e-3, center=(0.5, 0.5), smoothing_width=2.0 * dx,
    )
    solver = case.solver()
    magnitude = 0.01
    r_sample = 2.0 * hole_radius

    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]

    # Tension/compression peak at theta=90 degrees (sampled along the
    # vertical line through the hole center); shear's hoop stress has a
    # *node* at theta=90 (sin(2*theta)=0 there) and peaks instead at 45
    # degrees, so it needs a different sample point -- the same distinction
    # the example script makes by plotting sigma_12 (not hoop stress) along
    # the vertical line for shear.
    sample_angle = {"tension": np.pi / 2.0, "compression": np.pi / 2.0, "shear": np.pi / 4.0}

    for load, expected_scf in (("tension", 3.0), ("compression", 3.0), ("shear", 4.0)):
        eps_bar = case.macro_strain(load, magnitude, solver=solver)
        solution = solver.solve(eps_bar, tol=1.0e-6, max_iterations=3000)
        assert solution.converged
        sigma = np.asarray(solution.stress)

        # Sample well outside the diffuse boundary, where its own
        # influence has mostly decayed -- convert both the numeric and
        # analytic fields to polar there and compare the hoop stress
        # directly, rather than trying to hit the idealized peak at
        # r=hole_radius exactly, which a diffuse (not literally void)
        # boundary systematically softens.
        theta_sample = sample_angle[load]
        target_x1 = case.center[0] + r_sample * np.cos(theta_sample)
        target_x2 = case.center[1] + r_sample * np.sin(theta_sample)
        col = np.argmin(np.abs(x1 - target_x1))
        row = np.argmin(np.abs(x2 - target_x2))

        sigma_xx = sigma[0, 0, col, row, 0]
        sigma_yy = sigma[1, 1, col, row, 0]
        sigma_xy = sigma[0, 1, col, row, 0]
        _, numeric_hoop, _ = cartesian_to_polar_stress(
            sigma_xx, sigma_yy, sigma_xy, theta_sample
        )
        analytic_hoop = case.analytic_stress(r_sample, theta_sample, load, magnitude)[1]

        assert numeric_hoop / magnitude == pytest.approx(
            analytic_hoop / magnitude, rel=0.15
        )
        # Sanity: the sampled point should show real concentration, not
        # just agree with a near-zero value.
        assert abs(analytic_hoop / magnitude) > 1.0


# -- Finite-contrast inhomogeneity and its periodic (Eshelby + KS) solution --


def test_inhomogeneity_formula_reduces_to_the_void_formula_at_zero_contrast():
    a, magnitude, nu = 1.0, 2.0, 0.3
    r = np.linspace(1.0, 5.0, 20)
    theta = np.linspace(0.0, 2.0 * np.pi, 20)
    got = _inhomogeneity_polar_stress(
        r[:, None], theta[None, :], a, magnitude, 0.0, nu
    )
    want = _tension_polar_stress(r[:, None], theta[None, :], a, magnitude)
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, atol=1e-10)


def test_inhomogeneity_exterior_field_is_traction_continuous_at_the_boundary():
    # sigma_rr and sigma_r_theta must match the interior's uniform stress
    # converted to polar at r=hole_radius, for any finite contrast -- a
    # real physical requirement (traction continuity across a bonded
    # interface) that a wrong formula would very likely violate, unlike
    # the void limit's traction-*free* condition, which some wrong
    # formulas could still satisfy by accident.
    a, magnitude, nu, contrast = 1.0, 2.0, 0.3, 0.3
    theta = np.linspace(0.0, 2.0 * np.pi, 50)
    rr_ext, _, rt_ext = _inhomogeneity_polar_stress(
        a, theta, a, magnitude, contrast, nu
    )
    sigma_xx, sigma_yy, _ = _inhomogeneity_interior_stress(magnitude, contrast, nu)
    rr_in, _, rt_in = cartesian_to_polar_stress(sigma_xx, sigma_yy, 0.0, theta)
    np.testing.assert_allclose(rr_ext, rr_in, atol=1e-10)
    np.testing.assert_allclose(rt_ext, rt_in, atol=1e-10)


def test_equivalent_eigenstrain_round_trips_the_interior_stress():
    magnitude, contrast, lam0, mu0 = 2.0, 0.3, 1.0, 0.7
    nu0 = lam0 / (2.0 * (lam0 + mu0))
    sigma_xx, sigma_yy, sigma_zz = _inhomogeneity_interior_stress(magnitude, contrast, nu0)
    sigma_in = np.array([[sigma_xx, 0, 0], [0, sigma_yy, 0], [0, 0, sigma_zz]])
    true_strain = _isotropic_compliance_apply(sigma_in, contrast * lam0, contrast * mu0)
    eps_star = _equivalent_eigenstrain(magnitude, contrast, lam0, mu0, "tension")
    assert eps_star[2, 2] == pytest.approx(0.0, abs=1e-12)
    recovered = _isotropic_stress_apply(true_strain - eps_star, lam0, mu0)
    np.testing.assert_allclose(np.asarray(recovered), sigma_in, atol=1e-10)


def test_disk_fourier_transform_matches_a_real_space_fft():
    grid = Grid(shape=(128, 128, 1), lengths=(1.0, 1.0, 1.0))
    a = 0.1
    center = (0.5, 0.5)
    x = np.asarray(grid.x[0]) - center[0]
    y = np.asarray(grid.x[1]) - center[1]
    disk = (np.sqrt(x**2 + y**2) < a).astype(np.float32)
    numeric = np.asarray(grid.fft(disk))
    analytic = np.asarray(
        _disk_fourier_transform(grid, a, center=(center[0], center[1], 0.0))
    )
    # A sharp disk's staircase discretization on a finite grid is itself
    # only an approximation to the continuum circle -- this is checking
    # the Fourier-transform *formula and normalization*, not asking for
    # exact discrete/continuum agreement.
    scale = np.max(np.abs(numeric))
    np.testing.assert_allclose(numeric, analytic, atol=0.15 * scale)


def test_periodic_analytic_solution_mean_strain_equals_macro_strain():
    grid = Grid(shape=(128, 128, 1), lengths=(1.0, 1.0, 1.0))
    case = HoleInPlateCase(
        grid, matrix_lame_lambda=1.0, matrix_lame_mu=0.7, hole_radius=0.1,
        contrast=1.0e-3, center=(0.5, 0.5),
    )
    magnitude = 0.01
    eps_bar = np.asarray(case.macro_strain("tension", magnitude))
    strain, _ = case.periodic_analytic_solution("tension", magnitude)
    strain = np.asarray(strain)
    for i in range(3):
        for j in range(3):
            assert np.mean(strain[i, j]) == pytest.approx(eps_bar[i, j], abs=1e-6)


def test_periodic_analytic_solution_is_centered_on_the_hole_not_the_origin():
    # Regression test for a Fourier shift-theorem omission: an early
    # version of _disk_fourier_transform assumed the disk sat at the grid
    # origin, so the resulting field was centered on the domain corner
    # instead of `center` -- caught by inspecting the field's spatial
    # pattern directly. The far corner (diagonally opposite the hole, and
    # by periodicity roughly equidistant from the four neighboring
    # images) should sit close to the remote stress; a grid point just
    # outside the hole should not.
    grid = Grid(shape=(128, 128, 1), lengths=(1.0, 1.0, 1.0))
    hole_radius = 0.1
    case = HoleInPlateCase(
        grid, matrix_lame_lambda=1.0, matrix_lame_mu=0.7, hole_radius=hole_radius,
        contrast=1.0e-3, center=(0.5, 0.5),
    )
    magnitude = 0.01
    _, stress = case.periodic_analytic_solution("tension", magnitude)
    stress = np.asarray(stress)
    x1 = np.asarray(grid.x[0])[:, 0, 0]
    far_index = np.argmin(np.abs(x1 - 0.02))
    near_index = np.argmin(np.abs(x1 - (0.5 + 1.4 * hole_radius)))
    center_index = np.argmin(np.abs(x1 - 0.5))
    far_sigma_xx = stress[0, 0, far_index, far_index, 0]
    near_sigma_xx = stress[0, 0, center_index, near_index, 0]
    assert far_sigma_xx == pytest.approx(magnitude, rel=0.1)
    assert near_sigma_xx > 1.3 * magnitude


def test_periodic_analytic_solution_matches_numeric_solver():
    grid = Grid(shape=(256, 256, 1), lengths=(1.0, 1.0, 1.0))
    dx = grid.spacing[0]
    hole_radius = 0.1
    case = HoleInPlateCase(
        grid, matrix_lame_lambda=1.0, matrix_lame_mu=0.7, hole_radius=hole_radius,
        contrast=1.0e-3, center=(0.5, 0.5), smoothing_width=2.0 * dx,
    )
    solver = case.solver()
    magnitude = 0.01

    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    col = np.argmin(np.abs(x1 - case.center[0]))
    row = np.argmin(np.abs((x2 - case.center[1]) - 2.0 * hole_radius))

    for load in ("tension", "compression", "shear"):
        eps_bar = case.macro_strain(load, magnitude, solver=solver)
        numeric = solver.solve(eps_bar, tol=1.0e-6, max_iterations=3000)
        assert numeric.converged
        _, periodic_stress = case.periodic_analytic_solution(load, magnitude)

        numeric_sigma = np.asarray(numeric.stress)[:, :, col, row, 0]
        periodic_sigma = np.asarray(periodic_stress)[:, :, col, row, 0]
        scale = max(np.max(np.abs(periodic_sigma)), magnitude)
        np.testing.assert_allclose(numeric_sigma, periodic_sigma, atol=0.1 * scale)
