"""Analytical vs. numerical stress across a circular *inclusion* (finite
stiffness contrast, not a near-void hole) under remote tension.

Reuses :class:`crystallite.verification.HoleInPlateCase` unchanged --
despite the name, its equivalent-eigenstrain machinery
(:func:`crystallite.verification.elastic_deformation._equivalent_eigenstrain`)
is already a general finite-contrast Eshelby inhomogeneity construction,
not a void-specific approximation, so "soft hole" and "stiff/hard
inclusion" are just different `contrast` values of the same case. Only the
`smoothing_width` needs adjusting: a higher stiffness contrast needs a
*narrower* diffuse boundary (relative to the hole radius) for the
numerical solve to resolve the sharper near-boundary stress concentration
-- a well known characteristic of diffuse-interface methods, not a bug
(confirmed here by checking that the numerical/analytical mismatch at
fixed contrast shrinks as `smoothing_width` narrows, and grows with
`smoothing_width` held fixed as contrast increases).

Same 1D line-scan-through-the-hole convention as ``hole_in_plate.py``.
"""

from __future__ import annotations

import numpy as np

from _plotting import plt, save_all
from crystallite.grid import Grid
from crystallite.verification import HoleInPlateCase

MATRIX_LAME_LAMBDA = 1.0
MATRIX_LAME_MU = 0.7
HOLE_RADIUS = 0.1
GRID_SHAPE = (256, 256, 1)
MAGNITUDE = 0.01

# (label, contrast, smoothing_width as a multiple of grid spacing)
CASES = {
    "soft": (0.1, 2.0),
    "hard": (10.0, 0.5),
}


def _build_case(contrast, smoothing_factor):
    grid = Grid(shape=GRID_SHAPE, lengths=(1.0, 1.0, 1.0))
    smoothing_width = smoothing_factor * grid.spacing[0]
    return HoleInPlateCase(
        grid, matrix_lame_lambda=MATRIX_LAME_LAMBDA, matrix_lame_mu=MATRIX_LAME_MU,
        hole_radius=HOLE_RADIUS, contrast=contrast, center=(0.5, 0.5),
        smoothing_width=smoothing_width,
    )


def _line_through_hole(case, solver, load):
    """Return (xi, sigma_xx, sigma_yy)/MAGNITUDE along the line
    x1=center[0], x2 varying."""
    grid = case.grid
    eps_bar = case.macro_strain(load, MAGNITUDE, solver=solver)
    sol = solver.solve(eps_bar, tol=1.0e-6, max_iterations=5000)
    print(f"  numerical: converged={sol.converged}, iterations={sol.iterations}, "
          f"residual={sol.residual_norm:.3g}")
    sigma = np.asarray(sol.stress)

    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    col = np.argmin(np.abs(x1 - case.center[0]))

    xi = (x2 - case.center[1]) / HOLE_RADIUS
    sigma_xx = sigma[0, 0, col, :, 0] / MAGNITUDE
    sigma_yy = sigma[1, 1, col, :, 0] / MAGNITUDE
    return xi, sigma_xx, sigma_yy


def _periodic_analytic_line(case, load):
    """Return (xi, sigma_xx, sigma_yy)/MAGNITUDE along the same line, from
    :meth:`HoleInPlateCase.periodic_analytic_solution`."""
    grid = case.grid
    _, stress = case.periodic_analytic_solution(load, MAGNITUDE)
    stress = np.asarray(stress)

    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    col = np.argmin(np.abs(x1 - case.center[0]))

    xi = (x2 - case.center[1]) / HOLE_RADIUS
    sigma_xx = stress[0, 0, col, :, 0] / MAGNITUDE
    sigma_yy = stress[1, 1, col, :, 0] / MAGNITUDE
    return xi, sigma_xx, sigma_yy


def plot_inclusion(label):
    contrast, smoothing_factor = CASES[label]
    print(f"{label} inclusion (contrast={contrast}):")
    case = _build_case(contrast, smoothing_factor)
    solver = case.solver()
    xi, sxx, syy = _line_through_hole(case, solver, "tension")
    _, pxx, pyy = _periodic_analytic_line(case, "tension")

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.plot(xi, pxx, color="C0", label=r"$\sigma_{11}$ analytical")
    ax.plot(xi[::4], sxx[::4], "o", color="C0", markersize=3, label=r"$\sigma_{11}$ numerical")
    ax.plot(xi, pyy, color="C1", label=r"$\sigma_{22}$ analytical")
    ax.plot(xi[::4], syy[::4], "o", color="C1", markersize=3, label=r"$\sigma_{22}$ numerical")
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"$x_2 / r_0$")
    ax.set_ylabel(r"$\sigma / \sigma_\infty$")
    ax.set_title(f"{label} inclusion (contrast={contrast:g})")
    ax.legend(fontsize=7, ncol=2)
    save_all(fig, f"elastic_deformation.inclusion_{label}")
    plt.close(fig)


if __name__ == "__main__":
    plot_inclusion("soft")
    plot_inclusion("hard")
