import numpy as np
import pytest

from crystallite.grid import Grid
from crystallite.verification import HoleInPlateCase, EllipticalHoleInPlateCase
from crystallite.verification.elastic_deformation import (
    cartesian_to_polar_stress,
    polar_to_cartesian_stress,
    polar_to_cartesian_antiplane_stress,
    _tension_polar_stress,
    _antiplane_polar_stress,
    _antiplane_inhomogeneity_polar_stress,
    _antiplane_inhomogeneity_interior_stress,
    _antiplane_periodic_equivalent_eigenstrain,
    _ellipse_void_cartesian_stress,
    _ellipse_void_antiplane_cartesian_stress,
    _ellipse_equivalent_eigenstrain,
    _ellipse_inhomogeneity_exterior_stress,
    _ellipse_inhomogeneity_interior_stress,
    _ellipse_inhomogeneity_antiplane_exterior_stress,
    _ellipse_inhomogeneity_antiplane_interior_stress,
    _inhomogeneity_polar_stress,
    _inhomogeneity_interior_stress,
    _inhomogeneity_interior_cartesian,
    _isotropic_compliance_apply,
    _isotropic_stress_apply,
    _equivalent_eigenstrain,
    _disk_fourier_transform,
    _gradient_void_hole_correction_cartesian,
    _ellipse_gradient_inhomogeneity_solution,
    _ellipse_gradient_inhomogeneity_correction_cartesian,
    _lattice_eshelby_tensor,
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


# -- Antiplane (Mode III) hole -- circular hole under remote antiplane shear --


def test_antiplane_analytic_field_is_traction_free_at_the_hole_boundary():
    a = 0.15
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    sigma_r3, _ = _antiplane_polar_stress(a, theta, a, 2.0)
    np.testing.assert_allclose(sigma_r3, 0.0, atol=1e-10)


def test_antiplane_analytic_hoop_stress_matches_known_concentration_factor():
    # The classical antiplane result: stress concentration factor of
    # exactly 2 at the hole boundary (vs. 3 for the in-plane tension
    # case) -- a well-known, independent sanity check on the closed form.
    # theta=pi/2 (perpendicular to the loaded axis, sigma_13) is this
    # formula's peak, the same relative geometry as
    # test_tension_analytic_hoop_stress_matches_known_concentration_factors'
    # own peak at theta=pi/2 relative to its own loaded axis.
    a, magnitude = 0.15, 2.0
    _, sigma_theta3 = _antiplane_polar_stress(a, np.pi / 2.0, a, magnitude)
    assert abs(sigma_theta3) == pytest.approx(2.0 * magnitude)


def test_antiplane_analytic_field_reduces_to_remote_stress_far_away():
    a, magnitude = 0.15, 2.0
    r_far = 1.0e4 * a
    theta = np.linspace(0.0, 2.0 * np.pi, 37)
    sigma_r3, sigma_theta3 = _antiplane_polar_stress(r_far, theta, a, magnitude)
    sigma_13, sigma_23 = polar_to_cartesian_antiplane_stress(sigma_r3, sigma_theta3, theta)
    # sigma_13 (not sigma_23) is the loaded component -- matches
    # remote_antiplane_stress's own convention (see that method's
    # docstring for why: loading the same axis this project's in-plane
    # cases load, matching the reference material's own illustration).
    np.testing.assert_allclose(sigma_13, magnitude, rtol=1e-6)
    np.testing.assert_allclose(sigma_23, 0.0, atol=1e-6 * magnitude)


def test_solve_matches_antiplane_stress_concentration():
    grid = Grid(shape=(256, 256, 1), lengths=(1.0, 1.0, 1.0))
    hole_radius = 0.1
    case = HoleInPlateCase(
        grid, matrix_lame_lambda=1.0, matrix_lame_mu=0.7, hole_radius=hole_radius,
        contrast=1.0e-3, center=(0.5, 0.5), dealias=True,
    )
    solver = case.solver()
    magnitude = 0.01
    r_sample = 2.0 * hole_radius
    theta_sample = np.pi / 2.0  # concentration peak for antiplane shear

    eps_bar = case.macro_antiplane_strain(magnitude, solver=solver)
    solution = solver.solve(eps_bar, tol=1.0e-6, max_iterations=3000)
    assert solution.converged
    sigma = np.asarray(solution.stress)

    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    target_x1 = case.center[0] + r_sample * np.cos(theta_sample)
    target_x2 = case.center[1] + r_sample * np.sin(theta_sample)
    col = np.argmin(np.abs(x1 - target_x1))
    row = np.argmin(np.abs(x2 - target_x2))

    numeric_sigma_13 = sigma[0, 2, col, row, 0]
    sigma_r3, sigma_theta3 = case.antiplane_analytic_stress(r_sample, theta_sample, magnitude)
    analytic_sigma_13, _ = polar_to_cartesian_antiplane_stress(sigma_r3, sigma_theta3, theta_sample)

    assert numeric_sigma_13 / magnitude == pytest.approx(
        analytic_sigma_13 / magnitude, rel=0.15
    )
    assert abs(analytic_sigma_13 / magnitude) > 1.0


def test_antiplane_inhomogeneity_formula_reduces_to_the_void_formula_at_zero_contrast():
    a, magnitude = 1.0, 2.0
    r = np.linspace(1.0, 5.0, 20)
    theta = np.linspace(0.0, 2.0 * np.pi, 20)
    got = _antiplane_inhomogeneity_polar_stress(r[:, None], theta[None, :], a, magnitude, 0.0)
    want = _antiplane_polar_stress(r[:, None], theta[None, :], a, magnitude)
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, atol=1e-10)


def test_antiplane_inhomogeneity_interior_stress_vanishes_at_zero_contrast():
    sigma_13, sigma_23 = _antiplane_inhomogeneity_interior_stress(2.0, 0.0)
    assert sigma_13 == pytest.approx(0.0, abs=1e-12)
    assert sigma_23 == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("contrast", [0.5, 1.5])  # soft, hard
def test_antiplane_inhomogeneity_exterior_field_is_traction_continuous_at_the_boundary(contrast):
    # Antiplane counterpart of
    # test_inhomogeneity_exterior_field_is_traction_continuous_at_the_boundary:
    # sigma_r3 must match the interior's uniform sigma_13 rotated into
    # polar form at r=hole_radius, for any finite contrast -- checked for
    # both a soft (contrast=1/2) and a hard (contrast=3/2) inhomogeneity.
    a, magnitude = 1.0, 2.0
    theta = np.linspace(0.0, 2.0 * np.pi, 50)
    sigma_r3_ext, _ = _antiplane_inhomogeneity_polar_stress(a, theta, a, magnitude, contrast)
    sigma_13_in, sigma_23_in = _antiplane_inhomogeneity_interior_stress(magnitude, contrast)
    sigma_r3_in, _ = polar_to_cartesian_antiplane_stress(sigma_13_in, sigma_23_in, -theta)
    np.testing.assert_allclose(sigma_r3_ext, sigma_r3_in, atol=1e-10)


def test_antiplane_periodic_equivalent_eigenstrain_matches_the_isolated_closed_form_at_s1313_one_quarter():
    # s1313=1/4 is the isolated (dilute-limit) antiplane Eshelby-tensor
    # self-interaction -- derived directly from the equivalent-inclusion
    # condition eps_in_true = eps_bar + s_eff*eps_star (same relation
    # _periodic_equivalent_eigenstrain's shear_eigenstrain branch uses),
    # matching _antiplane_inhomogeneity_polar_stress's own interior/exterior
    # closed forms: solving that relation with the true interior strain
    # sigma_13_in/(2*contrast*mu0) and the remote eps_bar_13=magnitude/(2*mu0)
    # gives s_eff=1/2 independent of contrast, i.e. s1313=1/4.
    magnitude, mu0 = 2.0, 0.7
    for contrast in (0.0, 0.5, 1.0, 1.5, 1.0e6):  # void, soft, matched, hard, rigid-ish
        eps_star = _antiplane_periodic_equivalent_eigenstrain(magnitude, contrast, mu0, 0.25)
        want = magnitude * (1.0 - contrast) / ((1.0 + contrast) * mu0)
        assert eps_star[0, 2] == pytest.approx(want, rel=1e-8)
        assert eps_star[2, 0] == pytest.approx(want, rel=1e-8)


# -- Sharp crack limit: the elliptical hole formula vs. the classical LEFM
# near-tip asymptotic (a purely analytic check, no grid/solver involved:
# a genuine mathematical singularity cannot be resolved on any finite
# grid, so this is checked independently of the numeric solver entirely) --


def test_ellipse_void_tension_reduces_to_the_classical_crack_tip_asymptotic():
    r"""As an elliptical void's minor semi-axis shrinks (`semi_axis_a` ->
    0, `semi_axis_b`=`a_crack` fixed as the crack half-length, tension
    applied perpendicular to the major axis to open it), the exact
    closed form (:func:`_ellipse_void_cartesian_stress`) should approach
    the classical linear-elastic-fracture-mechanics Mode I near-tip
    asymptotic :math:`\sigma_{22}(r')=K_I/\sqrt{2\pi r'}`,
    :math:`K_I=\sigma_\infty\sqrt{\pi a_{\rm crack}}`, :math:`r'` the
    distance ahead of the tip along the crack's own axis.

    Not exact at any finite `semi_axis_a`: a real (rounded) tip has a
    finite curvature radius :math:`\rho_{\rm tip}=\mathtt{semi\_axis\_a}^2
    /a_{\rm crack}`, which caps the true peak stress at the classical
    finite Inglis value (:math:`1+2a_{\rm crack}/\mathtt{semi\_axis\_a}`)
    rather than the ideal formula's divergence as :math:`r'\to 0` --
    checked directly (not just assumed) that evaluating exactly at
    :math:`r'=0` is numerically degenerate (a removable 0/0 in the
    conformal-map formula at the ellipse's own corner), which is why this
    test samples a genuinely small but nonzero `r_prime`, chosen (as
    `100*rho_tip`) to sit inside the "K-dominant zone"
    (:math:`\rho_{\rm tip}\ll r'\ll a_{\rm crack}`) where the asymptotic
    actually applies -- confirmed to approach 1.0 as `semi_axis_a` shrinks
    further, not just accidentally close for one value. `semi_axis_a`
    values below ``1e-6`` are avoided: `r_prime` (already down at
    ``1e-16``-ish scale relative to `a_crack` by then) starts hitting
    float64's own precision floor in the conformal-map formula, an
    unrelated numerical artifact (checked directly -- confirmed by the
    formula returning outright ``nan``/huge-garbage values there, not a
    slowly-growing error), not a property of the physics being tested.
    """
    a_crack, magnitude = 0.125, 1.0
    ratios = []
    for semi_axis_a in (1.0e-2, 1.0e-4, 1.0e-6):
        rho_tip = semi_axis_a**2 / a_crack
        r_prime = 100.0 * rho_tip
        x, y = 0.0, a_crack + r_prime
        _, sigma_yy, _ = _ellipse_void_cartesian_stress(
            x, y, semi_axis_a, a_crack, "tension", magnitude
        )
        k_i_asymptotic = np.sqrt(a_crack / (2.0 * r_prime))
        ratios.append(np.real(sigma_yy) / k_i_asymptotic)

    # Should climb toward 1 as the ellipse sharpens -- not just be close
    # for a single cherry-picked case.
    assert ratios[-1] == pytest.approx(1.0, abs=0.02)
    assert ratios == sorted(ratios)


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


# -- periodic_inhomogeneity_stress / periodic_void_stress (real-space image sum) --


def _dilute_hole_case(contrast=1.0e-3, dealias=True):
    grid = Grid(shape=(64, 64, 1), lengths=(1.0, 1.0, 1.0))
    return HoleInPlateCase(
        grid, matrix_lame_lambda=1.0, matrix_lame_mu=0.7, hole_radius=0.1,
        contrast=contrast, center=(0.5, 0.5), dealias=dealias,
    )


def test_inhomogeneity_analytic_stress_reduces_to_analytic_stress_at_zero_contrast():
    # Method-level counterpart of test_inhomogeneity_formula_reduces_to_the_
    # void_formula_at_zero_contrast, exercised through the full load
    # dispatch (_dispatch_tension_polar_stress), not just the bare formula.
    case = _dilute_hole_case(contrast=0.0)
    r = np.linspace(0.15, 0.5, 10)
    theta = np.linspace(0.0, 2.0 * np.pi, 10)
    for load, magnitude in (
        ("tension", 2.0), ("compression", 2.0), ("shear", 2.0), ("biaxial", 2.0),
    ):
        got = case.inhomogeneity_analytic_stress(r[:, None], theta[None, :], load, magnitude)
        want = case.analytic_stress(r[:, None], theta[None, :], load, magnitude)
        for g, w in zip(got, want):
            np.testing.assert_allclose(np.asarray(g), np.asarray(w), atol=1e-10)


@pytest.mark.parametrize("n_images", [0, 2])
def test_periodic_inhomogeneity_stress_interior_is_uniform_regardless_of_n_images(n_images):
    # Regression test for a real bug: an earlier version left interior
    # points as the raw (uncorrected) image sum, which -- unlike
    # eshelby.cpp's own recipe -- lets neighboring images' exterior field
    # leak into the home hole's interior once n_images > 0, breaking the
    # exact uniformity Eshelby's theorem guarantees for an isolated
    # inhomogeneity (observed directly: the interior stress came out
    # non-uniform and even sign-flipped relative to the true closed-form
    # value at n_images=1, before the fix).
    case = _dilute_hole_case(contrast=0.3)
    magnitude = 0.01
    stress = np.asarray(
        case.periodic_inhomogeneity_stress("tension", magnitude, n_images=n_images)
    )
    inside = np.asarray(case.radius) < case.hole_radius
    interior_xx = stress[0, 0][inside]

    nu = case.matrix_lame_lambda / (2.0 * (case.matrix_lame_lambda + case.matrix_lame_mu))
    expected = _inhomogeneity_interior_cartesian(magnitude, case.contrast, nu, "tension")

    # exact periodic interior (lattice-sum Eshelby tensor), uniform and
    # independent of n_images; it differs from the isolated closed form
    # only at O(area fraction)
    exact = float(np.asarray(case.periodic_interior_stress("tension", magnitude))[0, 0])
    np.testing.assert_allclose(interior_xx, exact, atol=1e-12)
    assert exact == pytest.approx(float(expected[0, 0]), rel=2.0 * np.pi * case.hole_radius**2)


@pytest.mark.parametrize("n_images", [0, 2])
def test_periodic_void_stress_interior_is_exactly_zero_regardless_of_n_images(n_images):
    case = _dilute_hole_case(contrast=1.0e-3)
    magnitude = 0.01
    stress = np.asarray(case.periodic_void_stress("tension", magnitude, n_images=n_images))
    inside = np.asarray(case.radius) < case.hole_radius
    np.testing.assert_allclose(stress[0, 0][inside], 0.0, atol=1e-12)
    np.testing.assert_allclose(stress[1, 1][inside], 0.0, atol=1e-12)
    np.testing.assert_allclose(stress[0, 1][inside], 0.0, atol=1e-12)


def test_correction_functions_match_at_zero_contrast():
    # _void_correction_cartesian and _inhomogeneity_correction_cartesian
    # should agree exactly at contrast=0 -- called directly through
    # _periodic_image_sum (both with the *same* default recenter_target),
    # not through the public periodic_void_stress/periodic_inhomogeneity_
    # stress wrappers, which intentionally use different recentering
    # targets now (see test_periodic_inhomogeneity_stress_far_field_
    # matches_periodic_analytic_solution_domain_mean below) -- comparing
    # those directly would conflate that deliberate difference with this
    # test's actual question (do the two correction formulas themselves
    # still agree at the void limit).
    case = _dilute_hole_case(contrast=0.0)
    magnitude = 0.01
    for load in ("tension", "compression", "shear", "biaxial"):
        void = np.asarray(
            case._periodic_image_sum(case._void_correction_cartesian, load, magnitude, 1)
        )
        inhom = np.asarray(
            case._periodic_image_sum(case._inhomogeneity_correction_cartesian, load, magnitude, 1)
        )
        np.testing.assert_allclose(void, inhom, atol=1e-12)


def test_periodic_image_sum_honors_a_custom_recenter_target():
    # Isolates the recentering-target mechanism itself, with a trivial,
    # by-hand-verifiable correction function -- independent of both the
    # physical inhomogeneity formulas and the interior-overwrite behavior
    # (a spatially constant correction is unaffected by which points
    # count as "inside").
    case = _dilute_hole_case(contrast=1.0e-3)
    magnitude = 0.01

    def constant_correction(dx, dy, load, mag):
        return 0.5 * mag, 0.0, 0.0

    outside = np.asarray(case.radius) >= case.hole_radius

    default = np.asarray(
        case._periodic_image_sum(constant_correction, "tension", magnitude, n_images=0)
    )
    assert np.mean(default[0, 0][outside]) == pytest.approx(magnitude, rel=1e-6)

    custom_target = np.zeros((3, 3))
    custom_target[0, 0] = 3.0 * magnitude
    targeted = np.asarray(
        case._periodic_image_sum(
            constant_correction, "tension", magnitude, n_images=0,
            recenter_target=custom_target,
        )
    )
    assert np.mean(targeted[0, 0][outside]) == pytest.approx(3.0 * magnitude, rel=1e-6)


def test_periodic_inhomogeneity_stress_far_field_matches_periodic_analytic_solution_domain_mean():
    # The recentering target fix: periodic_inhomogeneity_stress recenters
    # its *whole-domain* mean to periodic_analytic_solution's own
    # whole-domain mean (the correct target for this solver's strain-
    # controlled boundary condition), not the naive nominal `magnitude`
    # periodic_void_stress still uses (eshelby.cpp's own target, correct
    # only for a stress-controlled boundary condition). Deliberately a
    # whole-domain mean on both sides, not an outside-only one: the
    # interior region (near-zero stress here) pulls a whole-domain mean
    # down relative to an outside-only mean, so comparing an outside-only
    # mean against a whole-domain target would be comparing two different
    # quantities, not testing the recentering guarantee itself.
    case = _dilute_hole_case(contrast=1.0e-3)
    magnitude = 0.01
    stress = np.asarray(case.periodic_inhomogeneity_stress("tension", magnitude, n_images=1))
    domain_mean = np.mean(stress[0, 0])

    _, stress_fft = case.periodic_analytic_solution("tension", magnitude)
    expected_mean = np.mean(np.asarray(stress_fft)[0, 0])

    # Not exact: overwriting the interior with the uncontaminated
    # home-only value (see the interior-uniformity tests above) shifts
    # the whole-domain mean slightly relative to the raw sum the
    # recentering step itself was computed from -- small here since the
    # hole is a small area fraction of the domain, but not exactly zero.
    assert domain_mean == pytest.approx(expected_mean, rel=0.02)
    # And *not* close to the naive nominal target, confirming this isn't
    # a vacuous check (the two targets genuinely differ here).
    assert domain_mean != pytest.approx(magnitude, rel=0.02)


# -- periodic_inhomogeneity_pressurized_stress / periodic_void_pressurized_stress
# (the "pressure" special case, left on the void-only method until now) --


@pytest.mark.parametrize("contrast", [0.5, 1.5])  # soft, hard
def test_periodic_inhomogeneity_pressurized_stress_interior_is_uniform(contrast):
    # Pressurized counterpart of
    # test_periodic_inhomogeneity_stress_interior_is_uniform_regardless_of_n_images:
    # the "biaxial minus uniform background" construction should not
    # break periodic_inhomogeneity_stress's own exact interior uniformity,
    # for either a soft (contrast=1/2) or a hard (contrast=3/2)
    # inhomogeneity.
    case = _dilute_hole_case(contrast=contrast)
    magnitude = 0.01
    stress = np.asarray(case.periodic_inhomogeneity_pressurized_stress(magnitude, n_images=1))
    inside = np.asarray(case.radius) < case.hole_radius

    nu = case.matrix_lame_lambda / (2.0 * (case.matrix_lame_lambda + case.matrix_lame_mu))
    expected = np.asarray(_inhomogeneity_interior_cartesian(magnitude, contrast, nu, "biaxial"))
    expected = expected - magnitude * np.eye(3)

    # Only the in-plane (0,0)/(1,1)/(0,1) components are actually
    # populated by periodic_inhomogeneity_stress (and so by this method,
    # built from it) -- (2,2) is left exactly zero, same plane-strain-zz
    # omission periodic_void_stress/periodic_inhomogeneity_stress already
    # have (their own Returns docstrings promise a (3,3)-shaped array, not
    # that every component is physically populated).
    exact = np.asarray(case.periodic_interior_stress("biaxial", magnitude)) - magnitude * np.eye(3)
    for i, j in ((0, 0), (1, 1), (0, 1)):
        np.testing.assert_allclose(stress[i, j][inside], exact[i, j], atol=1e-10)
        # the isolated closed form is the O(area fraction) limit of the exact value
        assert exact[i, j] == pytest.approx(
            expected[i, j], abs=3.0 * np.pi * case.hole_radius**2 * magnitude
        )


def test_periodic_inhomogeneity_pressurized_stress_matches_periodic_pressurized_stress_domain_mean():
    # Same recentering-target confirmation as
    # test_periodic_inhomogeneity_stress_far_field_matches_periodic_
    # analytic_solution_domain_mean, for the pressurized special case:
    # periodic_inhomogeneity_pressurized_stress inherits its recentering
    # from periodic_inhomogeneity_stress, so its domain mean should match
    # periodic_pressurized_stress's (the exact FFT construction, "biaxial
    # minus uniform background" the same way), not the naive `magnitude`.
    case = _dilute_hole_case(contrast=1.0e-3)
    magnitude = 0.01
    stress = np.asarray(case.periodic_inhomogeneity_pressurized_stress(magnitude, n_images=1))
    domain_mean = np.mean(stress[0, 0])

    expected_stress = np.asarray(case.periodic_pressurized_stress(magnitude))
    expected_mean = np.mean(expected_stress[0, 0])

    # By construction the field's domain mean is the exact lattice-sum
    # target for "biaxial", minus the uniform background.
    exact_mean = float(np.asarray(case.periodic_mean_stress("biaxial", magnitude))[0, 0]) - magnitude
    assert domain_mean == pytest.approx(exact_mean, abs=1e-6 * magnitude)

    # Cross-check against the FFT construction. Its own grid error (this is
    # a 64^2 grid, ~0.7% of the load) is the limit here, and the pressurized
    # signal is mostly cancelled background, so compare on the load scale.
    assert domain_mean == pytest.approx(expected_mean, abs=1.5e-2 * magnitude)


# -- periodic_antiplane_inhomogeneity_stress / periodic_antiplane_void_stress
# (antiplane counterparts of the block above) --


@pytest.mark.parametrize("n_images", [0, 2])
@pytest.mark.parametrize("contrast", [0.5, 1.5])  # soft, hard
def test_periodic_antiplane_inhomogeneity_stress_interior_is_uniform_regardless_of_n_images(
    contrast, n_images,
):
    # Antiplane counterpart of
    # test_periodic_inhomogeneity_stress_interior_is_uniform_regardless_of_n_images:
    # the same eshelby.cpp two-pass recipe (raw image sum outside,
    # uncontaminated home-image-only value inside) must hold here too, for
    # both a soft (contrast=1/2) and a hard (contrast=3/2) inhomogeneity.
    case = _dilute_hole_case(contrast=contrast)
    magnitude = 0.01
    stress = np.asarray(
        case.periodic_antiplane_inhomogeneity_stress(magnitude, n_images=n_images)
    )
    inside = np.asarray(case.radius) < case.hole_radius
    interior_13 = stress[0, 2][inside]

    expected_13, _ = _antiplane_inhomogeneity_interior_stress(magnitude, case.contrast)
    exact = float(np.asarray(case.periodic_antiplane_interior_stress(magnitude))[0, 2])
    np.testing.assert_allclose(interior_13, exact, atol=1e-12)
    assert exact == pytest.approx(expected_13, rel=2.0 * np.pi * case.hole_radius**2)


@pytest.mark.parametrize("n_images", [0, 2])
def test_periodic_antiplane_void_stress_interior_is_exactly_zero_regardless_of_n_images(n_images):
    case = _dilute_hole_case(contrast=1.0e-3)
    magnitude = 0.01
    stress = np.asarray(case.periodic_antiplane_void_stress(magnitude, n_images=n_images))
    inside = np.asarray(case.radius) < case.hole_radius
    np.testing.assert_allclose(stress[0, 2][inside], 0.0, atol=1e-12)
    np.testing.assert_allclose(stress[1, 2][inside], 0.0, atol=1e-12)


def test_antiplane_correction_functions_match_at_zero_contrast():
    # Antiplane counterpart of test_correction_functions_match_at_zero_contrast.
    case = _dilute_hole_case(contrast=0.0)
    magnitude = 0.01
    void_13, void_23 = case._antiplane_void_correction_cartesian(
        case.grid.x[0] - case.center[0], case.grid.x[1] - case.center[1], magnitude
    )
    inhom_13, inhom_23 = case._antiplane_inhomogeneity_correction_cartesian(
        case.grid.x[0] - case.center[0], case.grid.x[1] - case.center[1], magnitude
    )
    np.testing.assert_allclose(np.asarray(void_13), np.asarray(inhom_13), atol=1e-12)
    np.testing.assert_allclose(np.asarray(void_23), np.asarray(inhom_23), atol=1e-12)


# -- periodic_antiplane_analytic_solution (antiplane Eshelby-eigenstrain
# FFT construction, antiplane counterpart of periodic_analytic_solution) --


def test_periodic_antiplane_analytic_solution_decouples_from_in_plane_deformation():
    # A pure eps*_13 eigenstrain's body-force source lands entirely in the
    # antiplane (i=3) equilibrium equation for an isotropic reference
    # medium with no x3-variation (see _periodic_eshelby_antiplane_tensor's
    # docstring) -- so the in-plane strain response must come back out
    # exactly zero, not just small. This is what makes reusing the fully
    # general ElasticDeformation.reference_green/DifferentialOperators.div
    # machinery (built for -- and, until now, only exercised by -- the
    # four in-plane load cases) valid for antiplane loading too.
    case = _dilute_hole_case(contrast=1.0e-3)
    magnitude = 0.01
    strain, _ = case.periodic_antiplane_analytic_solution(magnitude)
    strain = np.asarray(strain)
    np.testing.assert_allclose(strain[0, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(strain[1, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(strain[0, 1], 0.0, atol=1e-12)


def test_periodic_antiplane_analytic_solution_mean_strain_equals_macro_antiplane_strain():
    # Antiplane counterpart of test_periodic_analytic_solution_mean_strain_
    # equals_macro_strain.
    case = _dilute_hole_case(contrast=1.0e-3)
    magnitude = 0.01
    eps_bar = np.asarray(case.macro_antiplane_strain(magnitude))
    strain, _ = case.periodic_antiplane_analytic_solution(magnitude)
    strain = np.asarray(strain)
    assert np.mean(strain[0, 2]) == pytest.approx(eps_bar[0, 2], abs=1e-6)
    assert np.mean(strain[1, 2]) == pytest.approx(eps_bar[1, 2], abs=1e-6)


def test_periodic_eshelby_antiplane_tensor_is_close_to_the_isolated_dilute_limit():
    # This module's own dilute geometry (HOLE_RADIUS=0.1 in a unit cell)
    # should sit close to, but not necessarily exactly at, the isolated
    # closed-form self-interaction s1313=1/4 -- the same "already close in
    # the dilute limit, but genuinely lattice-corrected" relationship
    # periodic_eshelby_tensor has to the isolated in-plane S-tensor.
    case = _dilute_hole_case(contrast=1.0e-3)
    s1313 = case.periodic_eshelby_antiplane_tensor()
    assert s1313 == pytest.approx(0.25, rel=0.1)


def test_periodic_antiplane_inhomogeneity_stress_far_field_matches_periodic_antiplane_analytic_solution_domain_mean():
    # Antiplane counterpart of test_periodic_inhomogeneity_stress_far_
    # field_matches_periodic_analytic_solution_domain_mean: the recentering
    # target fix applies here too, now that periodic_antiplane_analytic_
    # solution exists to calibrate against.
    case = _dilute_hole_case(contrast=1.0e-3)
    magnitude = 0.01
    stress = np.asarray(case.periodic_antiplane_inhomogeneity_stress(magnitude, n_images=1))
    domain_mean = np.mean(stress[0, 2])

    _, stress_fft = case.periodic_antiplane_analytic_solution(magnitude)
    expected_mean = np.mean(np.asarray(stress_fft)[0, 2])

    assert domain_mean == pytest.approx(expected_mean, rel=0.02)
    assert domain_mean != pytest.approx(magnitude, rel=0.02)


# -- Elliptical inhomogeneity H/T-tensor (eshelby.cpp's f()/g(), ported) and
# EllipticalHoleInPlateCase.periodic_inhomogeneity_stress --


def _dilute_ellipse_case(contrast=1.0e-3, semi_axis_a=0.0625, semi_axis_b=0.125):
    grid = Grid(shape=(64, 64, 1), lengths=(1.0, 1.0, 1.0))
    return EllipticalHoleInPlateCase(
        grid, matrix_lame_lambda=1.0, matrix_lame_mu=0.7,
        semi_axis_a=semi_axis_a, semi_axis_b=semi_axis_b,
        contrast=contrast, center=(0.5, 0.5), dealias=True,
    )


@pytest.mark.parametrize("load", ["tension", "biaxial"])
def test_ellipse_inhomogeneity_reduces_to_the_void_formula_at_zero_contrast(load):
    a, b, magnitude = 0.0625, 0.125, 2.0
    lam0, mu0 = 1.0, 0.7
    nu = lam0 / (2.0 * (lam0 + mu0))
    background = {"tension": (magnitude, 0.0), "biaxial": (magnitude, magnitude)}[load]

    r = 3.0
    theta = np.linspace(0.0, 2.0 * np.pi, 20)
    x, y = r * np.cos(theta), r * np.sin(theta)

    eps_star = _ellipse_equivalent_eigenstrain(magnitude, 0.0, lam0, mu0, load, a, b)
    sxx, syy, sxy = _ellipse_inhomogeneity_exterior_stress(x, y, eps_star, a, b, mu0, nu)
    sxx, syy = sxx + background[0], syy + background[1]

    sxx_v, syy_v, sxy_v = _ellipse_void_cartesian_stress(x, y, a, b, load, magnitude)
    np.testing.assert_allclose(sxx, np.real(sxx_v), atol=1e-10)
    np.testing.assert_allclose(syy, np.real(syy_v), atol=1e-10)
    np.testing.assert_allclose(sxy, np.real(sxy_v), atol=1e-10)


@pytest.mark.parametrize("load", ["tension", "biaxial"])
def test_ellipse_inhomogeneity_matches_circular_inhomogeneity_at_equal_semi_axes(load):
    # Method-level cross-check against the independently-derived (not
    # eigenstrain/H-tensor based) circular finite-contrast closed form.
    a = b = 1.0
    magnitude, contrast = 2.0, 0.3
    lam0, mu0 = 1.0, 0.7
    nu = lam0 / (2.0 * (lam0 + mu0))
    background = {"tension": (magnitude, 0.0), "biaxial": (magnitude, magnitude)}[load]

    r = 3.0
    theta = np.linspace(0.0, 2.0 * np.pi, 20)
    x, y = r * np.cos(theta), r * np.sin(theta)

    eps_star = _ellipse_equivalent_eigenstrain(magnitude, contrast, lam0, mu0, load, a, b)
    sxx, syy, sxy = _ellipse_inhomogeneity_exterior_stress(x, y, eps_star, a, b, mu0, nu)
    sxx, syy = sxx + background[0], syy + background[1]

    def formula(r, theta, hole_radius, magnitude):
        return _inhomogeneity_polar_stress(r, theta, hole_radius, magnitude, contrast, nu)

    from crystallite.verification.elastic_deformation import _dispatch_tension_polar_stress
    srr, stt, srt = _dispatch_tension_polar_stress(formula, r, theta, a, magnitude, load)
    sxx_c, syy_c, sxy_c = polar_to_cartesian_stress(srr, stt, srt, theta)

    np.testing.assert_allclose(sxx, sxx_c, atol=1e-10)
    np.testing.assert_allclose(syy, syy_c, atol=1e-10)
    np.testing.assert_allclose(sxy, sxy_c, atol=1e-10)


@pytest.mark.parametrize("load", ["tension", "biaxial", "shear"])
def test_ellipse_inhomogeneity_exterior_field_is_traction_continuous_at_the_boundary(load):
    # Elliptical counterpart of
    # test_inhomogeneity_exterior_field_is_traction_continuous_at_the_boundary,
    # for a genuinely non-circular ellipse (b/a=2) -- a real physical
    # requirement an H-tensor contraction bug would very likely violate.
    # "shear" used to be excluded here (see the now-removed
    # test_ellipse_inhomogeneity_shear_h_tensor_has_a_known_traction_bug,
    # a regression marker for exactly this failure): the h0100/h0111/h0101
    # terms in _ellipse_inhomogeneity_exterior_stress contract against the
    # off-diagonal (0,1)/(1,0) eigenstrain index pair, and were missing the
    # factor of 2 that pairing requires (eps_star[0, 1] alone stands for
    # both eps_star[0, 1] and the equal eps_star[1, 0], unlike a diagonal
    # entry) -- confirmed a clean, exact factor of 2 (not a fuzzy ~20%) by
    # comparison against an independent reference built by rotating the
    # already-correct diagonal/tension case 45 degrees, then fixed in that
    # function directly. "shear" now passes this same continuity check to
    # the same tolerance as "tension"/"biaxial".
    a, b, magnitude, contrast = 0.0625, 0.125, 2.0, 0.3
    lam0, mu0 = 1.0, 0.7
    nu = lam0 / (2.0 * (lam0 + mu0))
    eps_star = _ellipse_equivalent_eigenstrain(magnitude, contrast, lam0, mu0, load, a, b)

    theta = np.linspace(0.001, 2.0 * np.pi - 0.001, 13)
    x, y = a * np.cos(theta), b * np.sin(theta)
    nx, ny = x / a**2, y / b**2
    norm = np.hypot(nx, ny)
    nx, ny = nx / norm, ny / norm

    sxx_e, syy_e, sxy_e = _ellipse_inhomogeneity_exterior_stress(x, y, eps_star, a, b, mu0, nu)
    sxx_i, syy_i, sxy_i = _ellipse_inhomogeneity_interior_stress(eps_star, a, b, mu0, nu)

    tx_e, ty_e = sxx_e * nx + sxy_e * ny, sxy_e * nx + syy_e * ny
    tx_i, ty_i = sxx_i * nx + sxy_i * ny, sxy_i * nx + syy_i * ny
    np.testing.assert_allclose(tx_e, tx_i, atol=1e-10)
    np.testing.assert_allclose(ty_e, ty_i, atol=1e-10)


def test_ellipse_inhomogeneity_exterior_shear_matches_the_rotated_diagonal_reference():
    # Independent check of the shear-contraction fix above, stronger than
    # traction continuity alone: at the circular limit (semi_axis_a ==
    # semi_axis_b), the isolated problem is exactly equivariant under
    # rotation, so a pure eps*_12=gamma eigenstrain must give the same
    # field as eps*_11=gamma, eps*_22=-gamma (already independently
    # trustworthy -- it only exercises the diagonal e0/e1/e2 terms, never
    # the buggy e5 ones) rotated by 45 degrees. Checked directly, not
    # assumed: this is exactly how the factor-of-2 bug was originally
    # diagnosed (a clean 2x ratio at every probed point, not ~20% noise).
    a = b = 0.1
    lam0, mu0 = 1.0, 0.7
    nu = lam0 / (2.0 * (lam0 + mu0))
    gamma = 0.01

    def rotation_matrix(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    def exterior_tensor(x, y, eps_2x2):
        eps_star = np.zeros((3, 3))
        eps_star[:2, :2] = eps_2x2
        sxx, syy, sxy = _ellipse_inhomogeneity_exterior_stress(x, y, eps_star, a, b, mu0, nu)
        return np.array([[sxx, sxy], [sxy, syy]])

    eps_shear = np.array([[0.0, gamma], [gamma, 0.0]])
    eps_diag = np.array([[gamma, 0.0], [0.0, -gamma]])
    rot = rotation_matrix(np.deg2rad(45.0))

    for x0, y0 in [(0.0, 0.2), (0.15, 0.15), (0.3, 0.05), (0.2, -0.1), (-0.05, 0.25)]:
        direct = exterior_tensor(x0, y0, eps_shear)
        x_prime, y_prime = rot.T @ np.array([x0, y0])
        expected = rot @ exterior_tensor(x_prime, y_prime, eps_diag) @ rot.T
        np.testing.assert_allclose(direct, expected, atol=1e-12)


@pytest.mark.parametrize("contrast", [0.5, 1.5])  # soft, hard
def test_ellipse_periodic_inhomogeneity_stress_interior_is_uniform(contrast):
    case = _dilute_ellipse_case(contrast=contrast)
    magnitude = 0.01
    stress = np.asarray(case.periodic_inhomogeneity_stress("tension", magnitude, n_images=1))
    inside = np.asarray(case.elliptical_radius) < 1.0

    nu = case.matrix_lame_lambda / (2.0 * (case.matrix_lame_lambda + case.matrix_lame_mu))
    eps_star = _ellipse_equivalent_eigenstrain(
        magnitude, contrast, case.matrix_lame_lambda, case.matrix_lame_mu, "tension",
        case.semi_axis_a, case.semi_axis_b,
    )
    expected_xx, expected_yy, _ = _ellipse_inhomogeneity_interior_stress(
        eps_star, case.semi_axis_a, case.semi_axis_b, case.matrix_lame_mu, nu
    )
    background = case.remote_stress("tension", magnitude)
    isolated_xx = float(expected_xx + background[0, 0])
    exact_xx = float(np.asarray(case.periodic_interior_stress("tension", magnitude))[0, 0])
    np.testing.assert_allclose(stress[0, 0][inside], exact_xx, atol=1e-10)
    area_fraction = np.pi * case.semi_axis_a * case.semi_axis_b
    assert exact_xx == pytest.approx(isolated_xx, rel=2.0 * area_fraction)
    exact_yy = float(np.asarray(case.periodic_interior_stress("tension", magnitude))[1, 1])
    np.testing.assert_allclose(stress[1, 1][inside], exact_yy, atol=1e-10)
    # sigma_yy is small (transverse to the load): bound against the load, not itself
    assert exact_yy == pytest.approx(
        float(expected_yy + background[1, 1]), abs=3.0 * area_fraction * magnitude
    )


@pytest.mark.parametrize("contrast", [0.5, 1.5])  # soft, hard
def test_ellipse_periodic_inhomogeneity_pressurized_stress_interior_is_uniform(contrast):
    case = _dilute_ellipse_case(contrast=contrast)
    magnitude = 0.01
    stress = np.asarray(case.periodic_inhomogeneity_pressurized_stress(magnitude, n_images=1))
    inside = np.asarray(case.elliptical_radius) < 1.0

    nu = case.matrix_lame_lambda / (2.0 * (case.matrix_lame_lambda + case.matrix_lame_mu))
    eps_star = _ellipse_equivalent_eigenstrain(
        magnitude, contrast, case.matrix_lame_lambda, case.matrix_lame_mu, "biaxial",
        case.semi_axis_a, case.semi_axis_b,
    )
    expected_xx, expected_yy, _ = _ellipse_inhomogeneity_interior_stress(
        eps_star, case.semi_axis_a, case.semi_axis_b, case.matrix_lame_mu, nu
    )
    background = case.remote_stress("biaxial", magnitude)
    exact = np.asarray(case.periodic_interior_stress("biaxial", magnitude)) - magnitude * np.eye(3)
    np.testing.assert_allclose(stress[0, 0][inside], exact[0, 0], atol=1e-10)
    np.testing.assert_allclose(stress[1, 1][inside], exact[1, 1], atol=1e-10)
    # the isolated closed form is the O(area fraction) limit of the exact value
    tol = 3.0 * np.pi * case.semi_axis_a * case.semi_axis_b * magnitude
    assert exact[0, 0] == pytest.approx(float(expected_xx + background[0, 0] - magnitude), abs=tol)
    assert exact[1, 1] == pytest.approx(float(expected_yy + background[1, 1] - magnitude), abs=tol)


def test_ellipse_periodic_inhomogeneity_stress_far_field_matches_periodic_analytic_stress_domain_mean():
    case = _dilute_ellipse_case(contrast=1.0e-3)
    magnitude = 0.01
    stress = np.asarray(case.periodic_inhomogeneity_stress("tension", magnitude, n_images=1))
    domain_mean = np.mean(stress[0, 0])

    stress_fft = np.asarray(case.periodic_analytic_stress("tension", magnitude))
    expected_mean = np.mean(stress_fft[0, 0])

    assert domain_mean == pytest.approx(expected_mean, rel=0.05)
    assert domain_mean != pytest.approx(magnitude, rel=0.05)


@pytest.mark.parametrize("n_images", [0, 2])
def test_ellipse_periodic_void_stress_interior_is_exactly_zero_regardless_of_n_images(n_images):
    # Regression test for the interior-leakage fix: an earlier version of
    # this method (unlike HoleInPlateCase.periodic_void_stress, already
    # fixed) never overwrote interior points with the uncontaminated
    # home-image-only value, letting neighboring images' exterior field
    # leak into the true ellipse's own interior once n_images>0.
    case = _dilute_ellipse_case(contrast=1.0e-3)
    magnitude = 0.01
    stress = np.asarray(case.periodic_void_stress("tension", magnitude, n_images=n_images))
    inside = np.asarray(case.elliptical_radius) < 1.0
    np.testing.assert_allclose(stress[0, 0][inside], 0.0, atol=1e-12)
    np.testing.assert_allclose(stress[1, 1][inside], 0.0, atol=1e-12)


# -- _ellipse_void_antiplane_cartesian_stress / periodic_antiplane_void_stress
# (isolated closed form + periodic image sum, elliptical counterpart of
# _antiplane_polar_stress / HoleInPlateCase.periodic_antiplane_void_stress,
# added for the thin elliptical crack's Mode III case in thin_crack.py) --


def test_ellipse_void_antiplane_reduces_to_the_circular_concentration_factor_at_equal_semi_axes():
    # At semi_axis_a == semi_axis_b (m=0), the ellipse degenerates to a
    # circle, so the closed form should reproduce the classical circular
    # antiplane result at theta=pi/2: hoop stress concentration factor of
    # exactly 2 (see _antiplane_polar_stress's own docstring) -- checked
    # by evaluating this formula directly, not assumed from the analogy.
    # sigma_theta3 = -sigma_13 at theta=pi/2, and comes out negative here
    # (-2*magnitude) for this axis/loading convention -- verified against
    # _antiplane_polar_stress directly (see the off-axis test below), not
    # assumed from the factor-of-2 magnitude alone.
    a = b = 0.1
    magnitude = 2.0
    x, y = 0.0, a  # boundary point at theta = pi/2
    sigma_13, sigma_23 = _ellipse_void_antiplane_cartesian_stress(x, y, a, b, magnitude)
    assert sigma_13 == pytest.approx(2.0 * magnitude, rel=1e-10)
    assert sigma_23 == pytest.approx(0.0, abs=1e-10)


def test_ellipse_void_antiplane_matches_the_circular_closed_form_off_axis():
    # Broader cross-check against _antiplane_polar_stress (independently
    # implemented) at equal semi-axes, off the special theta=pi/2 point.
    a = b = 0.1
    magnitude = 2.0
    r, theta = 0.3, 0.7
    x, y = r * np.cos(theta), r * np.sin(theta)
    sigma_13, sigma_23 = _ellipse_void_antiplane_cartesian_stress(x, y, a, b, magnitude)

    sigma_r3, sigma_theta3 = _antiplane_polar_stress(r, theta, a, magnitude)
    sigma_13_c, sigma_23_c = polar_to_cartesian_antiplane_stress(sigma_r3, sigma_theta3, theta)
    assert sigma_13 == pytest.approx(sigma_13_c, rel=1e-10)
    assert sigma_23 == pytest.approx(sigma_23_c, rel=1e-10)


@pytest.mark.parametrize("n_images", [0, 2])
def test_ellipse_periodic_antiplane_void_stress_interior_is_exactly_zero_regardless_of_n_images(n_images):
    # Antiplane counterpart of test_ellipse_periodic_void_stress_interior_
    # is_exactly_zero_regardless_of_n_images: the home-image-only interior
    # value must stay exactly zero, not accumulate neighbor contamination.
    case = _dilute_ellipse_case(contrast=1.0e-3)
    magnitude = 0.01
    stress = np.asarray(case.periodic_antiplane_void_stress(magnitude, n_images=n_images))
    inside = np.asarray(case.elliptical_radius) < 1.0
    np.testing.assert_allclose(stress[0, 2][inside], 0.0, atol=1e-12)
    np.testing.assert_allclose(stress[1, 2][inside], 0.0, atol=1e-12)


def test_ellipse_periodic_antiplane_void_stress_matches_circular_case_at_equal_semi_axes():
    # Cross-check against the independently-implemented circular method
    # (HoleInPlateCase.periodic_antiplane_void_stress): a circle is just
    # an ellipse with semi_axis_a == semi_axis_b, so the two periodic
    # image sums should agree exactly for the same domain/hole-size ratio.
    hole_case = _dilute_hole_case(contrast=1.0e-3)
    ellipse_case = _dilute_ellipse_case(contrast=1.0e-3, semi_axis_a=0.1, semi_axis_b=0.1)
    magnitude = 0.01
    hole_stress = np.asarray(hole_case.periodic_antiplane_void_stress(magnitude, n_images=1))
    ellipse_stress = np.asarray(ellipse_case.periodic_antiplane_void_stress(magnitude, n_images=1))
    # atol/rtol loose enough to absorb floating-point accumulation from the
    # two independent (circular-polar vs. elliptical-conformal-map) formulas
    # each being summed/recentered over the full grid -- not a physics gap.
    np.testing.assert_allclose(ellipse_stress, hole_stress, atol=1e-8, rtol=1e-6)


# -- EllipticalHoleInPlateCase.periodic_prescribed_eigenstrain_void_stress
# (closed-form H/T-tensor image sum for a *prescribed* eigenstrain in a
# homogeneous matrix -- the eshelby.cpp-recipe counterpart of the FFT-based
# periodic_prescribed_eigenstrain_solution, added for the dislocation/point-
# defect cases where that FFT construction's shape aliasing is unreliable) --


@pytest.mark.parametrize("n_images", [0, 2])
def test_ellipse_periodic_prescribed_eigenstrain_void_stress_interior_is_uniform_regardless_of_n_images(n_images):
    # Regression-style check mirroring test_ellipse_periodic_void_stress_
    # interior_is_exactly_zero_regardless_of_n_images: the home-image-only
    # interior value (here generally nonzero, not zero, since this is an
    # eigenstrain-carrying region, not a void) must not depend on n_images
    # -- neighboring images' exterior field must never leak in.
    case = _dilute_ellipse_case(contrast=1.0)
    e = np.zeros((3, 3))
    e[0, 0] = 0.01
    stress_a = np.asarray(case.periodic_prescribed_eigenstrain_void_stress(e, n_images=0))
    stress_b = np.asarray(case.periodic_prescribed_eigenstrain_void_stress(e, n_images=2))
    inside = np.asarray(case.elliptical_radius) < 1.0
    np.testing.assert_allclose(stress_a[0, 0][inside], stress_b[0, 0][inside], atol=1e-12)
    np.testing.assert_allclose(stress_a[1, 1][inside], stress_b[1, 1][inside], atol=1e-12)


@pytest.mark.parametrize("eta", ["eta0", "eta5", "eta4"])
def test_ellipse_periodic_prescribed_eigenstrain_void_stress_matches_the_fft_reference(eta):
    # Cross-check against the independent FFT construction
    # (periodic_prescribed_eigenstrain_solution) at a well-resolved (not
    # pixel-thin) ellipse, where both constructions should be reasonably
    # accurate: domain mean (including this method's own nonzero-target
    # recentering, not just 0) and near-field pointwise values should
    # agree to a few percent -- the same kind of residual gap this
    # project's other closed-form-vs-FFT cross-checks show, not floating
    # noise, but not a large discrepancy either. "eta4" (antiplane eps_13)
    # exercises _ellipse_inhomogeneity_antiplane_exterior_stress/
    # _interior_stress, the from-scratch closed form that replaced
    # eshelby.cpp's own (found-wrong) H1212/H2012/H2020/T1212/T2020.
    case = _dilute_ellipse_case(contrast=1.0)
    e = np.zeros((3, 3))
    if eta == "eta0":
        e[0, 0] = 0.01
        i, j = 0, 0
    elif eta == "eta4":
        e[0, 2] = e[2, 0] = 0.01
        i, j = 0, 2
    else:
        e[0, 1] = e[1, 0] = 0.01
        i, j = 0, 1

    stress_img = np.asarray(case.periodic_prescribed_eigenstrain_void_stress(e, n_images=1))
    _, stress_fft = case.periodic_prescribed_eigenstrain_solution(e)
    stress_fft = np.asarray(stress_fft)

    mean_img = np.mean(stress_img[i, j])
    mean_fft = np.mean(stress_fft[i, j])
    assert mean_img == pytest.approx(mean_fft, rel=0.05)

    grid = case.grid
    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    col = np.argmin(np.abs(x1 - case.center[0]))
    outside = np.asarray(case.elliptical_radius)[col, :, 0] > 1.3
    np.testing.assert_allclose(
        stress_img[i, j, col, outside, 0], stress_fft[i, j, col, outside, 0],
        atol=2e-3, rtol=0.1,
    )


def test_ellipse_inhomogeneity_antiplane_interior_stress_matches_the_classical_circular_self_interaction():
    # At the circular limit (semi_axis_a == semi_axis_b), the antiplane
    # Eshelby self-interaction S1313=1/4 is a classical, Poisson-ratio-
    # independent result -- already relied on elsewhere in this project
    # (_antiplane_periodic_equivalent_eigenstrain's own docstring). Via
    # eps_in_13=2*S1313*eps*_13 and sigma_13=2*mu*(eps_in_13-eps*_13), the
    # interior stress this closed form should give at a=b is exactly
    # -mu*eps*_13, not eshelby.cpp's own (wrong) -0.75*mu*eps*_13.
    r0 = 0.1
    mu0 = 0.7
    eps13 = 0.02
    eigenstrain = np.zeros((3, 3))
    eigenstrain[0, 2] = eigenstrain[2, 0] = eps13
    sigma_13, sigma_23 = _ellipse_inhomogeneity_antiplane_interior_stress(eigenstrain, r0, r0, mu0)
    assert sigma_13 == pytest.approx(-mu0 * eps13, rel=1e-12)
    assert sigma_23 == pytest.approx(0.0, abs=1e-12)


def test_ellipse_inhomogeneity_antiplane_exterior_stress_has_no_cross_coupling_at_the_circular_limit():
    # Pure eps_13 should leave sigma_23 exactly zero directly ahead
    # (theta=0, along the loaded axis) at the circular limit, and pure
    # eps_23 should leave sigma_13 exactly zero there -- the antiplane
    # analog of checking there is no spurious cross term, independent of
    # the interior formula's own (already covered) lack of one.
    r0 = 0.1
    mu0 = 0.7
    magnitude = 0.02
    x, y = 0.2, 0.0

    eps13_only = np.zeros((3, 3))
    eps13_only[0, 2] = eps13_only[2, 0] = magnitude
    sigma_13, sigma_23 = _ellipse_inhomogeneity_antiplane_exterior_stress(x, y, eps13_only, r0, r0, mu0)
    assert sigma_23 == pytest.approx(0.0, abs=1e-12)
    assert abs(sigma_13) > 1e-6

    eps23_only = np.zeros((3, 3))
    eps23_only[1, 2] = eps23_only[2, 1] = magnitude
    sigma_13, sigma_23 = _ellipse_inhomogeneity_antiplane_exterior_stress(x, y, eps23_only, r0, r0, mu0)
    assert sigma_13 == pytest.approx(0.0, abs=1e-12)
    assert abs(sigma_23) > 1e-6


# -- HoleInPlateCase.periodic_body_force_solution/body_force_field (a
# genuinely applied force, not a kinematic eigenstrain -- added for
# cylindrical_body_force.py, alongside ElasticDeformation.solve's new
# body_force parameter) --


def test_hole_periodic_body_force_solution_matches_the_numeric_solve():
    # For this homogeneous-matrix (contrast=1) case, both sides are exact
    # in principle (no image summation, no equivalent-inclusion
    # approximation): the analytic side is a direct Green's-function
    # evaluation, and the numeric side is ElasticDeformation.solve's own
    # body_force path. Loose tolerance, not tight: this specific
    # right-hand side (nothing forcing the out-of-plane sector) is exactly
    # the CG stagnation case solve()'s best-iterate guard exists for (see
    # test_elastic_deformation.py's own regression test), so the numeric
    # side stops at its best found residual rather than machine precision
    # -- checked directly at this module's 64x64 grid: residual ~3.5% at
    # iteration 3, consistent with the gap this tolerance allows for.
    case = _dilute_hole_case(contrast=1.0)
    solver = case.solver()
    force_vector = np.array([0.0, -0.01, 0.0])

    force_field = case.body_force_field(force_vector)
    sol = solver.solve(
        np.zeros((3, 3)), body_force=force_field, tol=1e-6, max_iterations=3000
    )
    sigma_num = np.asarray(sol.stress)

    _, sigma_an = case.periodic_body_force_solution(force_vector)
    sigma_an = np.asarray(sigma_an)

    np.testing.assert_allclose(sigma_num, sigma_an, atol=1e-4, rtol=0.1)


def test_hole_periodic_body_force_solution_has_zero_domain_mean():
    # A genuinely applied, spatially localized force has no net effect on
    # the domain-average stress for a periodic, homogeneous-matrix problem
    # with no other loading -- the k=0 Fourier mode of the force response
    # is exactly zero (GreenOperator.field's own k=0 branch), unlike a
    # prescribed *eigenstrain*'s domain mean (see
    # EllipticalHoleInPlateCase.periodic_prescribed_eigenstrain_void_stress's
    # own docstring for why that case is different: a strain-controlled,
    # not force-controlled, boundary condition).
    case = _dilute_hole_case(contrast=1.0)
    force_vector = np.array([0.0, -0.01, 0.0])
    _, sigma_an = case.periodic_body_force_solution(force_vector)
    sigma_an = np.asarray(sigma_an)
    np.testing.assert_allclose(np.mean(sigma_an[0, 0]), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.mean(sigma_an[1, 1]), 0.0, atol=1e-10)


def test_hole_body_force_field_is_zero_outside_the_hole_radius():
    case = _dilute_hole_case(contrast=1.0)
    force_vector = np.array([0.0, -0.01, 0.0])
    field = np.asarray(case.body_force_field(force_vector))
    outside = np.asarray(case.radius) >= case.hole_radius
    np.testing.assert_allclose(field[:, outside], 0.0, atol=0.0)
    inside = np.asarray(case.radius) < case.hole_radius
    np.testing.assert_allclose(field[1][inside], -0.01, atol=0.0)


# -- HoleInPlateCase.surface_force_field/periodic_surface_force_solution
# (radial boundary line force on a same-material disk) --


def test_hole_surface_force_solution_matches_the_numeric_solve():
    case = _dilute_hole_case(contrast=1.0)
    solver = case.solver()
    traction = 0.01
    sol = solver.solve(
        np.zeros((3, 3)), body_force=case.surface_force_field(traction),
        tol=1e-6, max_iterations=3000,
    )
    _, sigma_an = case.periodic_surface_force_solution(traction)
    np.testing.assert_allclose(np.asarray(sol.stress), np.asarray(sigma_an), atol=1e-4, rtol=0.1)


def test_hole_surface_force_interior_is_uniformly_compressed_by_the_isolated_amount():
    # Isolated plane-strain limit: interior sigma = -p I with
    # p = q (lambda + mu) / (lambda + 2 mu); the periodic solve differs
    # only by an area-fraction correction.
    case = _dilute_hole_case(contrast=1.0)
    traction = 0.01
    lam, mu = case.matrix_lame_lambda, case.matrix_lame_mu
    pressure = traction * (lam + mu) / (lam + 2.0 * mu)
    _, sigma_an = case.periodic_surface_force_solution(traction)
    sigma_an = np.asarray(sigma_an)
    center = tuple(s // 2 for s in sigma_an.shape[2:4])
    np.testing.assert_allclose(sigma_an[0, 0, center[0], center[1], 0], -pressure, rtol=0.15)
    np.testing.assert_allclose(sigma_an[1, 1, center[0], center[1], 0], -pressure, rtol=0.15)


def test_hole_surface_force_field_has_zero_mean():
    case = _dilute_hole_case(contrast=1.0)
    field = np.asarray(case.surface_force_field(0.01))
    np.testing.assert_allclose(field.mean(axis=(1, 2, 3)), 0.0, atol=1e-8)


# -- EllipticalHoleInPlateCase.periodic_antiplane_inhomogeneity_stress --


@pytest.mark.parametrize("n_images", [0, 2])
@pytest.mark.parametrize("contrast", [0.5, 1.5])  # soft, hard
def test_ellipse_periodic_antiplane_inhomogeneity_stress_interior_matches_closed_form(
    contrast, n_images,
):
    # Independent closed form for the interior antiplane stress of an
    # elliptical inhomogeneity (semi-axis a along the loaded x1) under remote
    # sigma_13=tau: tau*beta*(a+b)/(a+beta*b). Limits: 2*beta/(1+beta) for a
    # circle, tau for a->0 (traction continuity), beta*tau for b->0 (strain
    # continuity), 0 for a void.
    case = _dilute_ellipse_case(contrast=contrast)
    magnitude = 0.01
    stress = np.asarray(case.periodic_antiplane_inhomogeneity_stress(magnitude, n_images=n_images))
    inside = np.asarray(case.elliptical_radius) < 1.0

    a, b = case.semi_axis_a, case.semi_axis_b
    isolated = magnitude * contrast * (a + b) / (a + contrast * b)
    exact = float(np.asarray(case.periodic_antiplane_interior_stress(magnitude))[0, 2])
    np.testing.assert_allclose(stress[0, 2][inside], exact, atol=1e-12)
    np.testing.assert_allclose(stress[1, 2][inside], 0.0, atol=1e-12)
    # the periodic value differs from the isolated closed form only at O(area fraction)
    assert exact == pytest.approx(isolated, rel=2.0 * np.pi * a * b)


def test_ellipse_periodic_antiplane_inhomogeneity_stress_far_field_matches_periodic_antiplane_analytic_stress_domain_mean():
    case = _dilute_ellipse_case(contrast=1.0e-3)
    magnitude = 0.01
    stress = np.asarray(case.periodic_antiplane_inhomogeneity_stress(magnitude, n_images=1))
    domain_mean = np.mean(stress[0, 2])

    expected_mean = np.mean(np.asarray(case.periodic_antiplane_analytic_stress(magnitude))[0, 2])

    assert domain_mean == pytest.approx(expected_mean, rel=0.05)
    assert domain_mean != pytest.approx(magnitude, rel=0.05)


def test_ellipse_periodic_antiplane_inhomogeneity_stress_matches_circular_case_at_equal_semi_axes():
    circle = _dilute_hole_case(contrast=0.5)
    radius = circle.hole_radius
    ellipse = _dilute_ellipse_case(contrast=0.5, semi_axis_a=radius, semi_axis_b=radius)
    magnitude = 0.01
    np.testing.assert_allclose(
        np.asarray(ellipse.periodic_antiplane_inhomogeneity_stress(magnitude, n_images=1))[0, 2],
        np.asarray(circle.periodic_antiplane_inhomogeneity_stress(magnitude, n_images=1))[0, 2],
        atol=2.0e-4 * magnitude,
    )


# -- Isolated elliptical inhomogeneity under a remote stress gradient ("moment") --

_GRADIENT_NU = 0.2


def _gradient_total_stress(solution, x, y):
    sxx, syy, sxy = _ellipse_gradient_inhomogeneity_correction_cartesian(x, y, solution)
    return sxx + solution["magnitude"] * y, syy, sxy


def test_ellipse_gradient_inhomogeneity_correction_vanishes_at_unit_contrast():
    solution = _ellipse_gradient_inhomogeneity_solution(0.03, 0.06, 1.0, _GRADIENT_NU, 1.0)
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-0.2, 0.2, 200), rng.uniform(-0.2, 0.2, 200)
    for component in _ellipse_gradient_inhomogeneity_correction_cartesian(x, y, solution):
        np.testing.assert_allclose(component, 0.0, atol=1e-12)


def test_ellipse_gradient_inhomogeneity_reduces_to_the_circular_void_closed_form():
    radius = 0.05
    solution = _ellipse_gradient_inhomogeneity_solution(radius, radius, 1e-10, _GRADIENT_NU, 1.0)
    rng = np.random.default_rng(1)
    r = rng.uniform(1.05 * radius, 0.3, 200)
    theta = rng.uniform(0.0, 2.0 * np.pi, 200)
    x, y = r * np.cos(theta), r * np.sin(theta)
    got = _ellipse_gradient_inhomogeneity_correction_cartesian(x, y, solution)
    expected = _gradient_void_hole_correction_cartesian(x, y, radius, 1.0)
    for g, e in zip(got, expected):
        np.testing.assert_allclose(g, np.asarray(e), atol=1e-9)


@pytest.mark.parametrize("a, b, contrast", [(0.03, 0.06, 0.3), (0.06, 0.03, 4.0), (0.05, 0.05, 2.0)])
def test_ellipse_gradient_inhomogeneity_traction_is_continuous_across_the_interface(a, b, contrast):
    solution = _ellipse_gradient_inhomogeneity_solution(a, b, contrast, _GRADIENT_NU, 1.0)
    t = np.linspace(0.0, 2.0 * np.pi, 97)
    nx, ny = np.cos(t) / a, np.sin(t) / b
    norm = np.hypot(nx, ny)
    nx, ny = nx / norm, ny / norm
    xb, yb = a * np.cos(t), b * np.sin(t)
    eps = 1e-8
    out = _gradient_total_stress(solution, xb + eps * nx, yb + eps * ny)
    inn = _gradient_total_stress(solution, xb - eps * nx, yb - eps * ny)
    for (sxx1, syy1, sxy1), (sxx2, syy2, sxy2) in [(out, inn)]:
        np.testing.assert_allclose(sxx1 * nx + sxy1 * ny, sxx2 * nx + sxy2 * ny, atol=1e-6)
        np.testing.assert_allclose(sxy1 * nx + syy1 * ny, sxy2 * nx + syy2 * ny, atol=1e-6)


def test_ellipse_gradient_inhomogeneity_field_is_in_equilibrium():
    solution = _ellipse_gradient_inhomogeneity_solution(0.03, 0.06, 0.3, _GRADIENT_NU, 1.0)
    h = 1e-6
    for px, py in [(0.05, 0.02), (-0.04, 0.09), (0.01, -0.05), (0.005, 0.01)]:
        def s(dx, dy):
            return _gradient_total_stress(solution, np.array([px + dx]), np.array([py + dy]))
        div_x = (s(h, 0)[0] - s(-h, 0)[0] + s(0, h)[2] - s(0, -h)[2]) / (2 * h)
        div_y = (s(h, 0)[2] - s(-h, 0)[2] + s(0, h)[1] - s(0, -h)[1]) / (2 * h)
        np.testing.assert_allclose(div_x, 0.0, atol=1e-8)
        np.testing.assert_allclose(div_y, 0.0, atol=1e-8)


# -- Lattice-sum periodic Eshelby tensor and the exact periodic mean stress
# (replaces the FFT domain mean the image-sum recentering used to need) --


def _isolated_eshelby(a, b, nu):
    d = 2.0 * (1.0 - nu)
    return {
        "s1111": ((b**2 + 2 * a * b) / (a + b) ** 2 + (1 - 2 * nu) * b / (a + b)) / d,
        "s2222": ((a**2 + 2 * a * b) / (a + b) ** 2 + (1 - 2 * nu) * a / (a + b)) / d,
        "s1122": (b**2 / (a + b) ** 2 - (1 - 2 * nu) * b / (a + b)) / d,
        "s2211": (a**2 / (a + b) ** 2 - (1 - 2 * nu) * a / (a + b)) / d,
        "s1212": ((a**2 + b**2) / (2 * (a + b) ** 2) + (1 - 2 * nu) / 2) / d,
    }


@pytest.mark.parametrize("a, b", [(0.004, 0.004), (0.003, 0.006), (0.006, 0.003)])
def test_lattice_eshelby_tensor_approaches_the_isolated_closed_form_as_the_area_fraction_vanishes(a, b):
    lam, mu = 1.0, 0.7
    nu = lam / (2.0 * (lam + mu))
    lattice = _lattice_eshelby_tensor(a, b, 1.0, 1.0, lam, mu)
    isolated = _isolated_eshelby(a, b, nu)
    # The periodic correction is O(area fraction) ~ 1e-4 here; the residual
    # is the lattice sum's own truncation error, which for ellipses this
    # small (transition mode ~ L/(2 pi a) ~ 50) is a few 1e-4 at 1024 modes.
    for key, value in isolated.items():
        assert lattice[key] == pytest.approx(value, abs=6e-4)
    # antiplane: S1313 = b / (2 (a + b))
    assert lattice["s1313"] == pytest.approx(b / (2.0 * (a + b)), abs=6e-4)


def test_lattice_eshelby_tensor_is_converged_in_the_number_of_modes():
    lam, mu = 1.0, 0.7
    coarse = _lattice_eshelby_tensor(0.07, 0.14, 1.0, 1.0, lam, mu, n_modes=512)
    fine = _lattice_eshelby_tensor(0.07, 0.14, 1.0, 1.0, lam, mu, n_modes=1024)
    for key in coarse:
        assert coarse[key] == pytest.approx(fine[key], abs=2e-4)


def test_lattice_eshelby_tensor_of_a_circle_in_a_square_cell_has_the_cubic_symmetry():
    lattice = _lattice_eshelby_tensor(0.1, 0.1, 1.0, 1.0, 1.0, 0.7)
    assert lattice["s1111"] == pytest.approx(lattice["s2222"], rel=1e-10)
    assert lattice["s1122"] == pytest.approx(lattice["s2211"], rel=1e-10)


@pytest.mark.parametrize("load", ["tension", "biaxial", "shear"])
def test_periodic_mean_stress_is_the_remote_stress_at_unit_contrast(load):
    case = _dilute_ellipse_case(contrast=1.0)
    magnitude = 0.01
    np.testing.assert_allclose(
        np.asarray(case.periodic_mean_stress(load, magnitude))[:2, :2],
        np.asarray(case.remote_stress(load, magnitude))[:2, :2],
        atol=1e-12,
    )


def test_periodic_antiplane_mean_stress_is_the_remote_stress_at_unit_contrast():
    case = _dilute_ellipse_case(contrast=1.0)
    assert float(np.asarray(case.periodic_antiplane_mean_stress(0.01))[0, 2]) == pytest.approx(0.01, rel=1e-10)


def _fine_ellipse_case(contrast, semi_axis_a=0.14, semi_axis_b=0.28):
    # 128^2: fine enough that the FFT reference is accurate to ~0.5% of the load
    grid = Grid(shape=(128, 128, 1), lengths=(1.0, 1.0, 1.0))
    return EllipticalHoleInPlateCase(
        grid, matrix_lame_lambda=1.0, matrix_lame_mu=0.7,
        semi_axis_a=semi_axis_a, semi_axis_b=semi_axis_b,
        contrast=contrast, center=(0.5, 0.5), dealias=True,
    )


@pytest.mark.parametrize("contrast", [1.0e-3, 3.0])
@pytest.mark.parametrize("load", ["tension", "shear"])
def test_periodic_mean_stress_agrees_with_the_fft_domain_mean(load, contrast):
    # the FFT mean is itself only grid-accurate, so this cross-checks the
    # lattice sum (which matched a converged CG solve to ~0.2% of the load)
    case = _fine_ellipse_case(contrast)
    magnitude = 0.01
    comp = (0, 0) if load == "tension" else (0, 1)
    lattice = float(np.asarray(case.periodic_mean_stress(load, magnitude))[comp])
    fft_mean = float(np.mean(np.asarray(case.periodic_analytic_stress(load, magnitude))[comp]))
    assert lattice == pytest.approx(fft_mean, abs=1e-2 * magnitude)


def test_periodic_antiplane_mean_stress_agrees_with_the_fft_domain_mean():
    case = _fine_ellipse_case(contrast=0.5)
    magnitude = 0.01
    lattice = float(np.asarray(case.periodic_antiplane_mean_stress(magnitude))[0, 2])
    fft_mean = float(np.mean(np.asarray(case.periodic_antiplane_analytic_stress(magnitude))[0, 2]))
    assert lattice == pytest.approx(fft_mean, abs=1e-2 * magnitude)


def test_circular_and_equal_axis_elliptical_periodic_mean_stress_agree():
    circle = _dilute_hole_case(contrast=0.5)
    ellipse = _dilute_ellipse_case(
        contrast=0.5, semi_axis_a=circle.hole_radius, semi_axis_b=circle.hole_radius
    )
    np.testing.assert_allclose(
        np.asarray(circle.periodic_mean_stress("tension", 0.01)),
        np.asarray(ellipse.periodic_mean_stress("tension", 0.01)),
        atol=1e-12,
    )


@pytest.mark.parametrize("load", ["tension", "biaxial", "shear"])
def test_periodic_interior_stress_is_the_remote_stress_at_unit_contrast(load):
    case = _dilute_ellipse_case(contrast=1.0)
    magnitude = 0.01
    np.testing.assert_allclose(
        np.asarray(case.periodic_interior_stress(load, magnitude))[:2, :2],
        np.asarray(case.remote_stress(load, magnitude))[:2, :2],
        atol=1e-12,
    )


@pytest.mark.parametrize("load", ["tension", "biaxial", "shear"])
def test_periodic_interior_stress_of_a_void_is_zero(load):
    case = _dilute_ellipse_case(contrast=0.0)
    np.testing.assert_allclose(
        np.asarray(case.periodic_interior_stress(load, 0.01))[:2, :2], 0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(case.periodic_antiplane_interior_stress(0.01)), 0.0, atol=1e-12
    )


@pytest.mark.parametrize("contrast", [0.3, 3.0])
def test_periodic_interior_stress_approaches_the_isolated_value_for_a_tiny_ellipse(contrast):
    # area fraction ~ 1e-4: the periodic correction is negligible, so the
    # lattice-sum interior must reproduce the classical isolated closed form
    a, b = 0.005, 0.01
    grid = Grid(shape=(64, 64, 1), lengths=(1.0, 1.0, 1.0))
    case = EllipticalHoleInPlateCase(
        grid, matrix_lame_lambda=1.0, matrix_lame_mu=0.7, semi_axis_a=a, semi_axis_b=b,
        contrast=contrast, center=(0.5, 0.5),
    )
    magnitude = 0.01
    nu = 1.0 / (2.0 * (1.0 + 0.7))
    eps_star = _ellipse_equivalent_eigenstrain(magnitude, contrast, 1.0, 0.7, "tension", a, b)
    isolated_xx = float(
        _ellipse_inhomogeneity_interior_stress(eps_star, a, b, 0.7, nu)[0]
        + case.remote_stress("tension", magnitude)[0, 0]
    )
    exact_xx = float(np.asarray(case.periodic_interior_stress("tension", magnitude))[0, 0])
    assert exact_xx == pytest.approx(isolated_xx, rel=5e-3)
    # antiplane: sigma_13 = tau*beta*(a+b)/(a+beta*b)
    antiplane = float(np.asarray(case.periodic_antiplane_interior_stress(magnitude))[0, 2])
    assert antiplane == pytest.approx(magnitude * contrast * (a + b) / (a + contrast * b), rel=5e-3)


# -- Radial line force (the "pressurized hole" of cylindrical_pressurized_hole.py)
# as a dilatation eigenstrain plus a uniform interior shift --


def _line_force_case(a, b, grid_n=128):
    grid = Grid(shape=(grid_n, grid_n, 1), lengths=(1.0, 1.0, 1.0))
    return EllipticalHoleInPlateCase(
        grid, matrix_lame_lambda=1.0, matrix_lame_mu=0.7, semi_axis_a=a, semi_axis_b=b,
        contrast=1.0, center=(0.5, 0.5),
    )


def test_line_force_interior_of_a_circle_is_the_uniform_hydrostatic_minus_p():
    case = _line_force_case(0.1, 0.1)
    q = 0.02
    lam, mu = 1.0, 0.7
    p = q * (lam + mu) / (lam + 2.0 * mu)
    stress = np.asarray(case.periodic_line_force_stress(q, n_images=2))
    inside = np.asarray(case.elliptical_radius) < 1.0
    np.testing.assert_allclose(stress[0, 0][inside], -p, rtol=1e-9)
    np.testing.assert_allclose(stress[1, 1][inside], -p, rtol=1e-9)
    np.testing.assert_allclose(stress[0, 1][inside], 0.0, atol=1e-12)


def test_line_force_stress_is_the_eigenstrain_stress_outside_and_shifted_by_q_inside():
    case = _line_force_case(0.09, 0.13)
    q = 0.02
    lam, mu = 1.0, 0.7
    eigenstrain = np.zeros((3, 3))
    eigenstrain[0, 0] = eigenstrain[1, 1] = -q / (2.0 * (lam + mu))
    eigen = np.asarray(case.periodic_prescribed_eigenstrain_void_stress(eigenstrain, n_images=1))
    force = np.asarray(case.periodic_line_force_stress(q, n_images=1))
    inside = np.asarray(case.elliptical_radius) < 1.0
    for i in (0, 1):
        np.testing.assert_allclose(force[i, i][~inside], eigen[i, i][~inside], atol=1e-14)
        np.testing.assert_allclose(force[i, i][inside], eigen[i, i][inside] - q, atol=1e-14)
    np.testing.assert_allclose(force[0, 1], eigen[0, 1], atol=1e-14)


def test_line_force_domain_mean_is_zero_up_to_the_grid_sampling_of_the_disk():
    case = _line_force_case(0.1, 0.1)
    q = 0.02
    stress = np.asarray(case.periodic_line_force_stress(q, n_images=2))
    assert abs(np.mean(stress[0, 0])) < 5e-3 * q
    assert abs(np.mean(stress[1, 1])) < 5e-3 * q


def test_line_force_exterior_approaches_the_isolated_lame_field_for_a_small_disk():
    # A small disk (area fraction ~2e-4): sigma_rr = -sigma_thth = A (R/r)^2
    a = 0.008
    case = _line_force_case(a, a)
    q = 0.02
    lam, mu = 1.0, 0.7
    amplitude = q * mu / (lam + 2.0 * mu)
    stress = np.asarray(case.periodic_line_force_stress(q, n_images=2))
    x1 = np.asarray(case.grid.x[0])[:, 0, 0]
    x2 = np.asarray(case.grid.x[1])[0, :, 0]
    col = int(np.argmin(np.abs(x1 - 0.5)))
    r = np.abs(x2 - 0.5)
    mask = (r > 3.0 * a) & (r < 0.2)
    # along the x2 axis: sigma_yy is the radial component, sigma_xx the hoop one
    np.testing.assert_allclose(
        stress[1, 1, col, :, 0][mask], amplitude * (a / r[mask]) ** 2, atol=0.03 * amplitude
    )
    np.testing.assert_allclose(
        stress[0, 0, col, :, 0][mask], -amplitude * (a / r[mask]) ** 2, atol=0.03 * amplitude
    )


# -- Body force over a disk: no compact closed form (a net force cannot be
# balanced by a periodic operator, so the exact solution needs lattice sums
# with a compensating background), so the reference is the spectral solve
# with the analytic disk transform; verify it independently of any solver --


def _body_force_case(grid_n):
    grid = Grid(shape=(grid_n, grid_n, 1), lengths=(1.0, 1.0, 1.0))
    return HoleInPlateCase(
        grid, matrix_lame_lambda=1.0, matrix_lame_mu=0.7, hole_radius=0.1,
        contrast=1.0, center=(0.5, 0.5),
    )


def _body_force_equilibrium_residual(grid_n, force):
    case = _body_force_case(grid_n)
    _, stress = case.periodic_body_force_solution(force)
    stress = np.asarray(stress)[:, :, :, :, 0]
    dx = 1.0 / grid_n

    def derivative(a, axis):
        return (np.roll(a, -1, axis=axis) - np.roll(a, 1, axis=axis)) / (2.0 * dx)

    radius = np.asarray(case.radius)[:, :, 0]
    chi = (radius < case.hole_radius).astype(float)
    source = chi - np.pi * case.hole_radius**2  # force density is force * (chi - mean)
    residual_x = derivative(stress[0, 0], 0) + derivative(stress[0, 1], 1) + force[0] * source
    residual_y = derivative(stress[1, 0], 0) + derivative(stress[1, 1], 1) + force[1] * source
    away = (np.abs(radius - case.hole_radius) > 0.25 * case.hole_radius) & (radius < 0.4)
    return max(np.abs(residual_x[away]).max(), np.abs(residual_y[away]).max()), stress


def test_body_force_solution_has_zero_domain_mean_stress():
    _, stress = _body_force_equilibrium_residual(64, np.array([0.0, -0.01, 0.0]))
    # zero net force, zero macroscopic strain
    np.testing.assert_allclose(np.mean(stress[0, 0]), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.mean(stress[1, 1]), 0.0, atol=1e-15)


def test_body_force_solution_satisfies_equilibrium_and_converges_with_the_grid():
    force = np.array([0.0, -0.01, 0.0])
    coarse, _ = _body_force_equilibrium_residual(64, force)
    fine, _ = _body_force_equilibrium_residual(128, force)
    # div(sigma) + f = 0 away from the boundary layer, to a small fraction of
    # the force (the remainder is the finite-difference check's own error)
    assert fine < 2.0e-2 * abs(force[1])
    assert fine < 0.6 * coarse
