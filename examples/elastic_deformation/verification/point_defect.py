"""Stress around a point defect (a vacancy or interstitial's misfit
volume), modeled as the single-grid-pixel limit of
``cylindrical_inhomogeneity.py``'s own ``dilation`` case: a circular
region carrying a full hydrostatic eigenstrain
(eps_11=eps_22=eps_33=`EIGENSTRAIN_MAGNITUDE`) in an otherwise homogeneous
matrix, shrunk from that module's ``HOLE_RADIUS`` down to a single grid
pixel's radius (`DEFECT_RADIUS`) -- the point-defect limit of the same
"misfitting inclusion" picture a dislocation is the *line* limit of (see
``dislocation.py``), just isotropic (a circle, not a degenerate ellipse).

Same domain, grid, and normalizing radius as every other case in this
family (``hole_in_plate.py``, ``cylindrical_inhomogeneity.py``,
``thin_crack.py``, ``dislocation.py``): ``xi = x1/HOLE_RADIUS`` in
``[0, 4]``, *not* normalized by the defect's own (much smaller)
`DEFECT_RADIUS` -- an earlier version of this module tried exactly that,
and since ``[0, 4]`` defect-radii spans only a couple of grid spacings, it
left the numeric side with only 2-3 points to plot from, however the
curve was framed or sampled. Normalizing by `HOLE_RADIUS` instead (the
same one every other script here already uses) keeps `DEFECT_RADIUS`
itself genuinely tiny -- a single pixel, so the defect is still
effectively a point -- while the *window* being plotted is the same
physical size as every other figure in this family, which this project's
own standard ``GRID_SHAPE`` already resolves comfortably (`HOLE_RADIUS`
alone spans ~26 grid points), no separate finer grid needed.

Sampled along x1 (x2=center[1] fixed), matching ``dislocation.py``'s own
sampling axis, one-sided (``[0, 4]``, not ``[-4, 4]``) since a genuine
circle has no preferred direction -- the dropped half is exactly
redundant, not a different regime, the same reasoning
``screw_dislocation.py`` used for its own other half before it was folded
into ``dislocation.py``.

Built on :class:`crystallite.verification.EllipticalHoleInPlateCase` with
``semi_axis_a == semi_axis_b`` (a circle is just a degenerate ellipse)
rather than :class:`crystallite.verification.HoleInPlateCase`, since only
the elliptical case has
:meth:`~crystallite.verification.EllipticalHoleInPlateCase.periodic_prescribed_eigenstrain_void_stress`,
the closed-form H/T-tensor image sum -- this project's own
``eshelby.cpp``-recipe generalization to a directly-prescribed eigenstrain
-- used here as the analytic reference in place of the FFT construction
(:meth:`~crystallite.verification.HoleInPlateCase.periodic_prescribed_eigenstrain_solution`)
``cylindrical_inhomogeneity.py`` uses at its own, much larger,
well-resolved ``HOLE_RADIUS``: checked directly (see ``thin_crack.py``'s
own history) that the FFT construction's shape-function aliasing washes
out exactly the near-core concentration a single-pixel region exists to
check, while the closed form (no shape-FFT step to alias) does not. The
dilation eigenstrain is purely diagonal, so it needs none of the
shear/antiplane H-tensor extensions the dislocation cases needed --
usable here unchanged.
"""

from __future__ import annotations

import numpy as np

from _plotting import plt, save_all
from crystallite.grid import Grid
from crystallite.verification import EllipticalHoleInPlateCase

MATRIX_LAME_LAMBDA = 1.0
MATRIX_LAME_MU = 0.7
HOLE_RADIUS = 0.1  # matches hole_in_plate.py/cylindrical_inhomogeneity.py/dislocation.py
GRID_SHAPE = (256, 256, 1)
EIGENSTRAIN_MAGNITUDE = 0.01
N_IMAGES = 1  # periodic image cutoff for the analytic curve (already converged here)


def _eigenstrain_tensor():
    e = np.zeros((3, 3))
    e[0, 0] = e[1, 1] = e[2, 2] = EIGENSTRAIN_MAGNITUDE
    return e


def _build_case():
    grid = Grid(shape=GRID_SHAPE, lengths=(1.0, 1.0, 1.0))
    defect_radius = 0.5 * grid.spacing[0]  # one grid pixel wide, full width
    return EllipticalHoleInPlateCase(
        grid, matrix_lame_lambda=MATRIX_LAME_LAMBDA, matrix_lame_mu=MATRIX_LAME_MU,
        semi_axis_a=defect_radius, semi_axis_b=defect_radius, contrast=1.0, center=(0.5, 0.5),
    )


def _xi_and_mask(case):
    """Grid row at x2=center[1], and the boolean mask/``xi`` (distance
    from the defect's own center, normalized by `HOLE_RADIUS`) for ``xi``
    in ``(0, 4]`` -- excludes the grid point exactly at the defect's own
    center (deep inside it, at this single-pixel size), whose interior
    value dwarfs the exterior decay this plot exists to show."""
    grid = case.grid
    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    row = np.argmin(np.abs(x2 - case.center[1]))
    xi = (x1 - case.center[0]) / HOLE_RADIUS
    mask = (xi > 0.0) & (xi <= 4.0)
    return row, mask, xi[mask]


def _line_through_defect_numeric(case, solver, eigenstrain_tensor):
    """Return (xi, sigma_11, sigma_22)/(mu*EIGENSTRAIN_MAGNITUDE) along
    x2=center[1], x1 in [center, center + 4*HOLE_RADIUS]."""
    eigenstrain_field = case.eigenstrain_field(eigenstrain_tensor)
    sol = solver.solve(
        np.zeros((3, 3)), eigenstrain=eigenstrain_field, tol=1.0e-6, max_iterations=5000
    )
    print(f"  numerical: converged={sol.converged}, iterations={sol.iterations}, "
          f"residual={sol.residual_norm:.3g}")
    sigma = np.asarray(sol.stress)

    row, mask, xi = _xi_and_mask(case)
    scale = MATRIX_LAME_MU * EIGENSTRAIN_MAGNITUDE
    sigma_11 = sigma[0, 0, mask, row, 0] / scale
    sigma_22 = sigma[1, 1, mask, row, 0] / scale
    return xi, sigma_11, sigma_22


def _line_through_defect_analytic(case, eigenstrain_tensor):
    stress = np.asarray(
        case.periodic_prescribed_eigenstrain_void_stress(eigenstrain_tensor, n_images=N_IMAGES)
    )
    row, mask, xi = _xi_and_mask(case)
    scale = MATRIX_LAME_MU * EIGENSTRAIN_MAGNITUDE
    sigma_11 = stress[0, 0, mask, row, 0] / scale
    sigma_22 = stress[1, 1, mask, row, 0] / scale
    return xi, sigma_11, sigma_22


def plot_point_defect():
    eigenstrain_tensor = _eigenstrain_tensor()
    print("point defect:")
    case = _build_case()
    solver = case.solver()

    xi_num, sxx_num, syy_num = _line_through_defect_numeric(case, solver, eigenstrain_tensor)
    xi_an, sxx_an, syy_an = _line_through_defect_analytic(case, eigenstrain_tensor)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.set_xlim(0.0, 4.0)
    ax.plot(xi_an, sxx_an, "-", color="C0", label=r"$\sigma_{11}$ analytical")
    ax.plot(xi_num, sxx_num, "o", color="C0", ms=4, label=r"$\sigma_{11}$ numerical")
    ax.plot(xi_an, syy_an, "-", color="C1", label=r"$\sigma_{22}$ analytical")
    ax.plot(xi_num, syy_num, "o", color="C1", ms=4, label=r"$\sigma_{22}$ numerical")
    ax.set_xlabel(r"$x_1 / r_0$")
    ax.set_ylabel(r"$\sigma / (\mu \varepsilon^0)$")
    ax.legend(fontsize=7, ncol=2)
    save_all(fig, "elastic_deformation.point_defect")
    plt.close(fig)


if __name__ == "__main__":
    plot_point_defect()
