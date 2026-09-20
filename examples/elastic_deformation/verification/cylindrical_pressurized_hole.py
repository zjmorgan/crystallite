"""Pressurized circular "hole" modeled as a genuinely applied radial line
force on the boundary of a disk of the *same* material as the matrix.

The force is ``f = q grad(chi)`` (``chi`` the disk indicator): a radial line
force of magnitude ``q`` on the boundary, directed inward. With no
stiffness contrast the disk interior is uniformly, hydrostatically
stressed, ``sigma = -p I`` with ``p = q (lambda + mu) / (lambda + 2 mu)``
(plane strain), and the exterior is the Lame field
``sigma_rr = -sigma_thth = A (R/r)^2`` with ``A = p mu / (lambda + mu)``
-- tensile radially at the boundary, so *not* the pressurized-cavity
exterior of ``hole_in_plate.py``'s pressure figure (that would need an
eigenstrain, i.e. the Eshelby route, not a force).

The analytic reference is
:meth:`crystallite.verification.EllipticalHoleInPlateCase.periodic_line_force_stress`
at ``semi_axis_a == semi_axis_b``, with no FFT. A source
``f = -div(sigma*)`` is the equilibrium source of the eigenstress
``sigma* = -q chi I``, i.e. a uniform dilatation eigenstrain
``e = -q / (2 (lambda + mu))`` over the disk, so the force-loaded stress is
the exact closed-form image sum of that eigenstrain
(:meth:`~crystallite.verification.EllipticalHoleInPlateCase.periodic_prescribed_eigenstrain_void_stress`)
minus ``q`` on the diagonal inside the disk -- identical outside, exactly
``-p I`` inside. No new kernel is involved. The numeric solve applies the
same band-limited line force.
"""

from __future__ import annotations

import numpy as np

from _plotting import analytic_numeric_curve, component_legend, plt, save_all
from crystallite.grid import Grid
from crystallite.verification import EllipticalHoleInPlateCase, HoleInPlateCase

MATRIX_LAME_LAMBDA = 1.0
MATRIX_LAME_MU = 0.7
HOLE_RADIUS = 0.1
GRID_SHAPE = (256, 256, 1)
PRESSURE = 0.01  # interior stress is -PRESSURE * I
N_IMAGES = 2  # periodic image cutoff for the analytic reference


def _traction():
    lam, mu = MATRIX_LAME_LAMBDA, MATRIX_LAME_MU
    return PRESSURE * (lam + 2.0 * mu) / (lam + mu)


def _build_case():
    grid = Grid(shape=GRID_SHAPE, lengths=(1.0, 1.0, 1.0))
    return HoleInPlateCase(
        grid, matrix_lame_lambda=MATRIX_LAME_LAMBDA, matrix_lame_mu=MATRIX_LAME_MU,
        hole_radius=HOLE_RADIUS, contrast=1.0, center=(0.5, 0.5),
    )


def _build_analytic_case():
    """The same disk as :func:`_build_case`, as the ``a == b`` ellipse that
    owns the closed-form image-sum reference."""
    grid = Grid(shape=GRID_SHAPE, lengths=(1.0, 1.0, 1.0))
    return EllipticalHoleInPlateCase(
        grid, matrix_lame_lambda=MATRIX_LAME_LAMBDA, matrix_lame_mu=MATRIX_LAME_MU,
        semi_axis_a=HOLE_RADIUS, semi_axis_b=HOLE_RADIUS, contrast=1.0, center=(0.5, 0.5),
    )


def _line(case, stress):
    grid = case.grid
    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    col = np.argmin(np.abs(x1 - case.center[0]))
    xi = (x2 - case.center[1]) / HOLE_RADIUS
    return xi, stress[0, 0, col, :, 0] / PRESSURE, stress[1, 1, col, :, 0] / PRESSURE


def plot_pressurized_hole():
    case = _build_case()
    solver = case.solver()
    traction = _traction()
    sol = solver.solve(
        np.zeros((3, 3)), body_force=case.surface_force_field(traction),
        tol=1.0e-6, max_iterations=5000,
    )
    print(f"numerical: converged={sol.converged}, iterations={sol.iterations}, "
          f"residual={sol.residual_norm:.3g}")
    xi, sxx_num, syy_num = _line(case, np.asarray(sol.stress))
    stress = _build_analytic_case().periodic_line_force_stress(traction, n_images=N_IMAGES)
    _, sxx_an, syy_an = _line(case, np.asarray(stress))
    center = len(xi) // 2
    print(f"interior sigma_11/p: numeric={sxx_num[center]:.3f} periodic={sxx_an[center]:.3f} isolated=-1")

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.set_xlim(-4, 4)
    analytic_numeric_curve(ax, xi, sxx_an, sxx_num, r"$\sigma_{11}$", "C0")
    analytic_numeric_curve(ax, xi, syy_an, syy_num, r"$\sigma_{22}$", "C1")
    ax.set_xlabel(r"$x_2 / r_0$")
    ax.set_ylabel(r"$\sigma / p$")
    component_legend(ax)
    save_all(fig, "elastic_deformation.pressurized_hole")
    plt.close(fig)


if __name__ == "__main__":
    plot_pressurized_hole()
