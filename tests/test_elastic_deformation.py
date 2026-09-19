import numpy as np
import pytest

from crystallite.elastic_deformation import ElasticDeformation
from crystallite.grid import Grid
from crystallite.material.properties import isotropic_stiffness
from crystallite.spectral.long_range import GreenOperator


def test_homogeneous_material_gives_zero_displacement_and_exact_strain():
    grid = Grid(shape=(16, 16, 16), lengths=(1.0, 1.0, 1.0))
    lam0, mu0 = 1.0, 0.7
    solver = ElasticDeformation(
        grid, lame_lambda=lam0, lame_mu=mu0,
        reference_lame_lambda=lam0, reference_lame_mu=mu0,
    )
    macro_strain = np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    solution = solver.solve(macro_strain)

    assert solution.converged
    assert solution.iterations == 0
    np.testing.assert_allclose(np.asarray(solution.displacement), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(solution.strain)[0, 0], 0.01, atol=1e-6)


def test_matvec_is_self_adjoint_for_a_heterogeneous_material():
    grid = Grid(shape=(16, 16, 16), lengths=(1.0, 1.0, 1.0))
    rng = np.random.default_rng(0)
    lam_field = (1.0 + 0.3 * np.sin(
        2 * np.pi * np.asarray(grid.x[0])
    )).astype(np.float32)
    mu_field = (0.7 + 0.1 * np.cos(
        2 * np.pi * np.asarray(grid.x[1])
    )).astype(np.float32)
    lam_field = np.broadcast_to(lam_field, grid.shape)
    mu_field = np.broadcast_to(mu_field, grid.shape)
    solver = ElasticDeformation(
        grid, lame_lambda=lam_field, lame_mu=mu_field,
        reference_lame_lambda=1.0, reference_lame_mu=0.7,
    )

    u = rng.standard_normal((3,) + grid.shape).astype(np.float32)
    v = rng.standard_normal((3,) + grid.shape).astype(np.float32)
    a_u = np.asarray(solver._matvec(u))
    a_v = np.asarray(solver._matvec(v))

    lhs = np.sum(a_u * v)
    rhs = np.sum(u * a_v)
    assert lhs == pytest.approx(rhs, rel=1e-4)


def test_matvec_matches_a_finite_difference_reference():
    # Independent cross-check of the matrix-free spectral operator against
    # a plain central-difference calculation, for a smooth heterogeneous
    # material -- this is what actually caught (and rules out) bugs in the
    # operator itself, as opposed to the sign convention connecting it to
    # the applied load (see test_solve_reproduces_uniform_stress_state_*
    # below for that).
    n = 128
    grid = Grid(shape=(n, n, 1), lengths=(1.0, 1.0, 1.0))
    x = np.asarray(grid.x[0])[:, 0, 0]
    y = np.asarray(grid.x[1])[0, :, 0]
    xx, yy = np.meshgrid(x, y, indexing="ij")

    lam_field = (1.0 + 0.3 * np.sin(2 * np.pi * xx)).astype(np.float32).reshape(n, n, 1)
    mu_field = (0.7 + 0.1 * np.cos(2 * np.pi * yy)).astype(np.float32).reshape(n, n, 1)
    solver = ElasticDeformation(
        grid, lame_lambda=lam_field, lame_mu=mu_field,
        reference_lame_lambda=1.0, reference_lame_mu=0.7,
    )

    ux = np.sin(2 * np.pi * xx).astype(np.float32)
    uy = np.cos(2 * np.pi * yy).astype(np.float32)
    uz = np.zeros((n, n), dtype=np.float32)
    u = np.stack([ux, uy, uz], axis=0).reshape(3, n, n, 1)

    a_u_spectral = np.asarray(solver._matvec(u))[:, :, :, 0]

    dx = grid.spacing[0]
    dy = grid.spacing[1]

    def d_dx(f):
        return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)

    def d_dy(f):
        return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dy)

    u3 = u[:, :, :, 0]
    grad = np.zeros((3, 3, n, n))
    grad[0, 0], grad[0, 1] = d_dx(u3[0]), d_dy(u3[0])
    grad[1, 0], grad[1, 1] = d_dx(u3[1]), d_dy(u3[1])
    grad[2, 0], grad[2, 1] = d_dx(u3[2]), d_dy(u3[2])
    strain = 0.5 * (grad + np.swapaxes(grad, 0, 1))

    lam2, mu2 = lam_field[:, :, 0], mu_field[:, :, 0]
    trace = strain[0, 0] + strain[1, 1] + strain[2, 2]
    sigma = np.zeros((3, 3, n, n))
    for i in range(3):
        for j in range(3):
            sigma[i, j] = lam2 * trace * (1.0 if i == j else 0.0) + 2 * mu2 * strain[i, j]

    divergence = np.zeros((3, n, n))
    for i in range(3):
        divergence[i] = d_dx(sigma[i, 0]) + d_dy(sigma[i, 1])
    a_u_finite_difference = -divergence

    # A plain 2nd-order finite difference has real truncation error on
    # these fields (not negligible relative to values near a zero
    # crossing, hence the generous atol); the spectral operator itself is
    # effectively exact for this band-limited input.
    np.testing.assert_allclose(a_u_spectral, a_u_finite_difference, atol=0.15, rtol=0.05)


def test_solve_converges_for_a_moderate_contrast_inclusion():
    grid = Grid(shape=(32, 32, 32), lengths=(1.0, 1.0, 1.0))
    lam0, mu0 = 1.0, 0.7
    x, y, z = (np.asarray(grid.x[i]) for i in range(3))
    r = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2)
    inside = r < 0.15
    lam_field = np.broadcast_to(
        np.where(inside, lam0 * 0.01, lam0).astype(np.float32), grid.shape
    )
    mu_field = np.broadcast_to(
        np.where(inside, mu0 * 0.01, mu0).astype(np.float32), grid.shape
    )
    solver = ElasticDeformation(
        grid, lame_lambda=lam_field, lame_mu=mu_field,
        reference_lame_lambda=lam0, reference_lame_mu=mu0,
    )
    macro_strain = np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    solution = solver.solve(macro_strain, tol=1e-6, max_iterations=500)

    assert solution.converged
    np.testing.assert_allclose(
        np.mean(np.asarray(solution.strain)[0, 0]), 0.01, atol=1e-5
    )


def test_solve_reproduces_uniform_stress_state_near_a_soft_inclusion():
    # Regression test for a sign error in the body force -- A(u*) must
    # equal +div(C:eps_bar), not -div(C:eps_bar). Both signs give the
    # right far-field stress (u* -> 0 there regardless, since the body
    # force is localized near the inclusion), so this only shows up close
    # to the inclusion: with the wrong sign, the correction there points
    # the wrong way, turning a stress increase into a decrease. Checked
    # here against the sign of the *change* in sigma_yy near the top of a
    # soft circular inclusion under uniaxial x-tension, which must be
    # positive (a stress concentration), not negative.
    grid = Grid(shape=(128, 128, 1), lengths=(1.0, 1.0, 1.0))
    lam0, mu0 = 1.0, 0.7
    a = 0.1
    x = np.asarray(grid.x[0]) - 0.5
    y = np.asarray(grid.x[1]) - 0.5
    r = np.sqrt(x**2 + y**2)
    dx = grid.spacing[0]
    width = 2.0 * dx
    blend = 0.5 * (1.0 + np.tanh((r - a) / width))
    contrast = 1.0e-3
    lam_field = np.broadcast_to(
        ((contrast + (1.0 - contrast) * blend) * lam0).astype(np.float32), grid.shape
    )
    mu_field = np.broadcast_to(
        ((contrast + (1.0 - contrast) * blend) * mu0).astype(np.float32), grid.shape
    )
    solver = ElasticDeformation(
        grid, lame_lambda=lam_field, lame_mu=mu_field,
        reference_lame_lambda=lam0, reference_lame_mu=mu0,
    )
    magnitude = 0.01
    eps_bar = solver.reference_green.apply_compliance(
        np.array([[magnitude, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    )
    solution = solver.solve(eps_bar, tol=1e-6, max_iterations=3000)
    assert solution.converged

    sigma = np.asarray(solution.stress)
    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    col = np.argmin(np.abs(x1 - 0.5))
    row_near_hole = np.argmin(np.abs((x2 - 0.5) - 1.4 * a))

    # sigma_xx (the hoop stress on the line perpendicular to the load,
    # through the top of the hole) must exceed the remote value there --
    # a real stress concentration, not a relief.
    sigma_xx_near_hole = sigma[0, 0, col, row_near_hole, 0]
    assert sigma_xx_near_hole > 1.3 * magnitude


# -- preconditioner_green (decoupled from reference_green/reference_lame_*) --


def test_preconditioner_green_defaults_to_reference_green():
    grid = Grid(shape=(8, 8, 8), lengths=(1.0, 1.0, 1.0))
    solver = ElasticDeformation(
        grid, lame_lambda=1.0, lame_mu=0.7, reference_lame_lambda=1.0, reference_lame_mu=0.7,
    )
    assert solver.preconditioner_green is solver.reference_green


def test_custom_preconditioner_green_does_not_change_the_converged_solution():
    # CG's correctness never depends on which SPD preconditioner is used,
    # only how fast it gets there -- a genuinely different (not just
    # rescaled) preconditioner should still converge to the same answer.
    grid = Grid(shape=(32, 32, 32), lengths=(1.0, 1.0, 1.0))
    lam0, mu0 = 1.0, 0.7
    x, y, z = (np.asarray(grid.x[i]) for i in range(3))
    r = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2)
    inside = r < 0.15
    lam_field = np.broadcast_to(
        np.where(inside, lam0 * 5.0, lam0).astype(np.float32), grid.shape
    )
    mu_field = np.broadcast_to(
        np.where(inside, mu0 * 5.0, mu0).astype(np.float32), grid.shape
    )
    macro_strain = np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    solver_default = ElasticDeformation(
        grid, lame_lambda=lam_field, lame_mu=mu_field,
        reference_lame_lambda=lam0, reference_lame_mu=mu0,
    )
    sol_default = solver_default.solve(macro_strain, tol=1e-6, max_iterations=500)
    assert sol_default.converged

    # A different-*shape* (not just rescaled) preconditioner: a different
    # Poisson ratio than the matrix's own.
    precond_stiffness = isotropic_stiffness(lam0 * 0.1, mu0)
    solver_custom = ElasticDeformation(
        grid, lame_lambda=lam_field, lame_mu=mu_field,
        reference_lame_lambda=lam0, reference_lame_mu=mu0,
        preconditioner_green=GreenOperator(grid, precond_stiffness),
    )
    sol_custom = solver_custom.solve(macro_strain, tol=1e-6, max_iterations=1000)
    assert sol_custom.converged

    np.testing.assert_allclose(
        np.asarray(sol_custom.displacement), np.asarray(sol_default.displacement), atol=1e-5
    )


# -- solve()'s `body_force` parameter, and the CG stagnation guard building
# it surfaced the need for --


def _localized_disk_force(grid, radius, force_vector, center=(0.5, 0.5)):
    x = np.asarray(grid.x[0]) - center[0]
    y = np.asarray(grid.x[1]) - center[1]
    inside = (x**2 + y**2) < radius**2
    force = np.zeros((3,) + grid.shape, dtype=np.float32)
    for i, f in enumerate(force_vector):
        if f != 0.0:
            force[i] = np.broadcast_to(np.where(inside, f, 0.0), grid.shape)
    return force


def test_body_force_matches_the_exact_reference_green_solution_for_a_homogeneous_material():
    # For a homogeneous material (no stiffness contrast), reference_green
    # is already the exact Green's function for the true medium, not just
    # a preconditioner -- so a direct (non-iterative) evaluation of it
    # against the same force is the exact answer to compare the iterative
    # solve() path against, independent of solve()'s own CG machinery.
    grid = Grid(shape=(256, 256, 1), lengths=(1.0, 1.0, 1.0))
    lam0, mu0 = 1.0, 0.7
    solver = ElasticDeformation(
        grid, lame_lambda=lam0, lame_mu=mu0,
        reference_lame_lambda=lam0, reference_lame_mu=mu0,
    )
    force_field = _localized_disk_force(grid, 0.1, (0.0, -0.01, 0.0))

    solution = solver.solve(
        np.zeros((3, 3)), body_force=force_field, tol=1e-6, max_iterations=3000
    )
    sigma_num = np.asarray(solution.stress)

    force_hat = grid.fft(force_field)
    eps_ref = np.asarray(grid.ifft(solver.reference_green.field(force_hat)))
    sigma_ref = np.asarray(solver.stress(eps_ref))

    np.testing.assert_allclose(sigma_num, sigma_ref, atol=1e-6, rtol=1e-3)


def test_body_force_with_nonzero_mean_has_its_mean_dropped_not_left_to_diverge():
    # A uniform (nonzero-mean) body force has no periodic solution (a
    # periodic operator cannot balance a net force) -- solve() must drop
    # the mean itself (matching GreenOperator.field's own k=0 handling)
    # rather than handing CG an inconsistent right-hand side, which would
    # have no solution to converge to at all.
    grid = Grid(shape=(32, 32, 32), lengths=(1.0, 1.0, 1.0))
    lam0, mu0 = 1.0, 0.7
    solver = ElasticDeformation(
        grid, lame_lambda=lam0, lame_mu=mu0,
        reference_lame_lambda=lam0, reference_lame_mu=mu0,
    )
    uniform_force = np.zeros((3,) + grid.shape, dtype=np.float32)
    uniform_force[1] = 0.01

    solution = solver.solve(
        np.zeros((3, 3)), body_force=uniform_force, tol=1e-6, max_iterations=200
    )
    assert solution.converged
    np.testing.assert_allclose(np.asarray(solution.displacement), 0.0, atol=1e-5)


def test_solve_does_not_diverge_for_a_body_force_that_does_not_excite_the_out_of_plane_sector():
    # Regression test for a real, solve()-wide CG fragility this
    # `body_force` feature surfaced (not specific to body forces): on a
    # pseudo-2D grid (grid.shape[2] == 1), a right-hand side that forces
    # only the in-plane directions leaves the out-of-plane sector's own
    # near-zero-eigenvalue (low in-plane wavenumber) modes completely
    # unforced; round-off alone is enough to excite them, and once excited
    # `p^T A p` collapses toward zero, sending `alpha` (and the iterate)
    # to increasingly wild values within a handful of iterations if CG is
    # simply left to run. Checked directly: before the stagnation guard in
    # solve() (tracking the best iterate, stopping once the residual grows
    # past 2x its best value), this diverged to a residual > 1e15 by
    # iteration ~500 on exactly this setup; now it stays bounded and close
    # to the true (exact, for this homogeneous case) answer regardless of
    # how many iterations are requested.
    grid = Grid(shape=(256, 256, 1), lengths=(1.0, 1.0, 1.0))
    lam0, mu0 = 1.0, 0.7
    solver = ElasticDeformation(
        grid, lame_lambda=lam0, lame_mu=mu0,
        reference_lame_lambda=lam0, reference_lame_mu=mu0,
    )
    force_field = _localized_disk_force(grid, 0.1, (0.0, -0.01, 0.0))

    solution = solver.solve(
        np.zeros((3, 3)), body_force=force_field, tol=1e-12, max_iterations=3000
    )

    assert solution.residual_norm < 0.1
    assert np.all(np.isfinite(np.asarray(solution.stress)))
    assert np.max(np.abs(np.asarray(solution.stress))) < 1.0
