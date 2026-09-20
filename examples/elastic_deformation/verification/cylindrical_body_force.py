"""Analytical vs. numerical stress around a circular region carrying a
genuinely applied body force (e.g. a region whose *density*, not
stiffness, differs from the matrix, so gravity pulls on it by a different
amount than it does the matrix material it displaces) in an otherwise
homogeneous, force-free matrix.

This project's original (pre-crystallite) reference material's own
version of this case (``dislocations.cpp``'s sibling ``force.cpp``,
``hooke.tex``'s ``cylindrical_body_force``) instead modeled a circular
region of density :math:`\\rho_1` inside an *infinite* matrix of density
:math:`\\rho_0`, both under the same ambient gravity :math:`g` --
matching the classical "gravitating inclusion" problem, but at the cost
of the ambient (matrix-wide) part of that load having no net-zero force
over any finite/periodic cell (a uniform body force cannot be balanced
by a periodic operator at all), which is why that reference's own
periodic image sum needed an extra ad hoc linear correction fit to
specific boundary-point values after the fact, to patch up the resulting
mismatch.

This module drops the ambient-gravity framing entirely and keeps only
the genuinely *localized* part: a compact excess/deficit force confined
to the disk, zero everywhere else -- the same self-contained,
localized-perturbation structure every other case in this project
already has (a hole, a crack, a dislocation, a point defect). Physically
this is "how much *more* (or less) this region weighs than the matrix
material it replaces," which is exactly the part of the classical problem
that actually varies with position; the matrix's own uniform response to
gravity does not depend on where you evaluate it and contributes no
information about the inhomogeneity.

The analytic reference is
:meth:`crystallite.verification.HoleInPlateCase.periodic_body_force_solution`
-- a spectral solve, the one reference in this family that is *not* a
closed-form image sum. That is deliberate: a uniform force over the disk has
a nonzero net force, which no periodic cell can balance, so an exact
real-space solution needs lattice sums (Ewald or Weierstrass potentials)
with a compensating uniform background -- machinery with no accuracy payoff
here, because the stress is *continuous* across the disk boundary (the force
loads the stress's derivative, not the stress), so the spectral sum, with the
analytic Bessel transform of the disk (no pixelated mask), converges quickly:
against a 1024^2 solution the 256^2 line is within 2e-4 of the force times
the radius away from the boundary and 1.2e-3 within 0.4 radii of it (peak
stress 0.65). It also satisfies ``div(sigma) + f = 0`` (see the tests) and has
exactly zero domain mean. It is exact, not an approximation or an image sum,
and not even iterative:
since only the *force* differs between the disk and the matrix (not the
elastic constants), the reference-medium Green's function is already the
exact solution, the same way
:meth:`~crystallite.verification.HoleInPlateCase.periodic_prescribed_eigenstrain_solution`
is exact for a prescribed eigenstrain in a homogeneous matrix, one
differentiation order down. The numeric side uses
:class:`crystallite.elastic_deformation.ElasticDeformation`'s new
`body_force` support (added alongside this module): a genuinely applied
force entering equilibrium as :math:`\\nabla\\cdot\\sigma(x)+f(x)=0`,
unlike `eigenstrain`'s kinematic-misfit route through Hooke's law.

Building this surfaced a real, independent bug in the CG solver: a
compact force with nothing forcing the out-of-plane direction excites a
near-null sector of the elastic operator on this project's pseudo-2D
grids (``grid.shape[2] == 1``, giving that direction a whole continuum of
near-zero eigenvalues at low in-plane wavenumber) and diverges within a
handful of iterations once round-off leaks in there -- now fixed
directly in :meth:`ElasticDeformation.solve` (tracks the best iterate and
stops once the residual starts growing, rather than chasing the
divergence), not worked around here.
"""

from __future__ import annotations

import numpy as np

from _plotting import analytic_numeric_curve, component_legend, plt, save_all
from crystallite.grid import Grid
from crystallite.verification import HoleInPlateCase

MATRIX_LAME_LAMBDA = 1.0
MATRIX_LAME_MU = 0.7
HOLE_RADIUS = 0.1
GRID_SHAPE = (256, 256, 1)
BODY_FORCE_MAGNITUDE = 0.01  # force density magnitude, pointing down (-x2)


def _force_vector():
    force = np.zeros(3)
    force[1] = -BODY_FORCE_MAGNITUDE
    return force


def _build_case():
    grid = Grid(shape=GRID_SHAPE, lengths=(1.0, 1.0, 1.0))
    return HoleInPlateCase(
        grid, matrix_lame_lambda=MATRIX_LAME_LAMBDA, matrix_lame_mu=MATRIX_LAME_MU,
        hole_radius=HOLE_RADIUS, contrast=1.0, center=(0.5, 0.5),
    )


def _line_through_hole_numeric(case, solver, force_vector):
    """Return (xi, sigma_11, sigma_22)/(BODY_FORCE_MAGNITUDE * HOLE_RADIUS)
    along the line x1=center[0], x2 varying."""
    grid = case.grid
    force_field = case.body_force_field(force_vector)
    sol = solver.solve(
        np.zeros((3, 3)), body_force=force_field, tol=1.0e-6, max_iterations=3000
    )
    print(f"  numerical: converged={sol.converged}, iterations={sol.iterations}, "
          f"residual={sol.residual_norm:.3g}")
    sigma = np.asarray(sol.stress)

    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    col = np.argmin(np.abs(x1 - case.center[0]))

    scale = BODY_FORCE_MAGNITUDE * HOLE_RADIUS
    xi = (x2 - case.center[1]) / HOLE_RADIUS
    sigma_xx = sigma[0, 0, col, :, 0] / scale
    sigma_yy = sigma[1, 1, col, :, 0] / scale
    return xi, sigma_xx, sigma_yy


def _line_through_hole_analytic(case, force_vector):
    """Return (xi, sigma_11, sigma_22)/(BODY_FORCE_MAGNITUDE * HOLE_RADIUS)
    along the same line, from
    :meth:`HoleInPlateCase.periodic_body_force_solution`."""
    grid = case.grid
    _, stress = case.periodic_body_force_solution(force_vector)
    stress = np.asarray(stress)

    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    col = np.argmin(np.abs(x1 - case.center[0]))

    scale = BODY_FORCE_MAGNITUDE * HOLE_RADIUS
    xi = (x2 - case.center[1]) / HOLE_RADIUS
    sigma_xx = stress[0, 0, col, :, 0] / scale
    sigma_yy = stress[1, 1, col, :, 0] / scale
    return xi, sigma_xx, sigma_yy


def plot_body_force():
    print("body force:")
    case = _build_case()
    solver = case.solver()
    force_vector = _force_vector()

    xi, sxx, syy = _line_through_hole_numeric(case, solver, force_vector)
    _, pxx, pyy = _line_through_hole_analytic(case, force_vector)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.set_xlim(-4, 4)
    analytic_numeric_curve(ax, xi, pxx, sxx, r"$\sigma_{11}$", "C0")
    analytic_numeric_curve(ax, xi, pyy, syy, r"$\sigma_{22}$", "C1")
    ax.set_xlabel(r"$x_2 / r_0$")
    ax.set_ylabel(r"$\sigma / (F_0 r_0)$")
    component_legend(ax)
    save_all(fig, "elastic_deformation.cylindrical_body_force")
    plt.close(fig)


if __name__ == "__main__":
    plot_body_force()
