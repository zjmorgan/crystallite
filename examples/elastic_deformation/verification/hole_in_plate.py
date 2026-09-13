"""Analytical vs. numerical stress across a circular hole, under remote
tension, compression, pure shear, equal biaxial tension, uniform internal
pressure (zero remote stress), and a remote stress gradient ("moment" /
pure-bending loading).

Mirrors the comparison-figure convention used in the original solver's own
reference presentation (``.../modeling/presentations/hooke/hooke.tex``,
figures ``cylindrical.hole.pgf``/``cylindrical.hole.shear.pgf``): a 1D line
scan straight through the hole (position normalized by the hole radius,
stress normalized by the remote stress), not a full 2D field map or an
angular sweep at fixed radius -- a much more direct way to see the
concentration-then-decay profile and to place the numerical points right
next to the analytic curve.

The analytic curve shown against the numerical points is the exact solution
to the actual periodic problem
(:meth:`crystallite.verification.HoleInPlateCase.periodic_analytic_solution`),
via an Eshelby equivalent eigenstrain and a closed-form
Khachaturyan-Shatalov Fourier solve -- the numerical points should track it
closely.

The hole is a soft circular inclusion
(:class:`crystallite.verification.HoleInPlateCase`) with a diffuse (tanh)
boundary, evolved to equilibrium by
:class:`crystallite.elastic_deformation.ElasticDeformation`.
"""

from __future__ import annotations

import numpy as np

from _plotting import plt, save_all
from crystallite.grid import Grid
from crystallite.verification import HoleInPlateCase

MATRIX_LAME_LAMBDA = 1.0
MATRIX_LAME_MU = 0.7
HOLE_RADIUS = 0.1
CONTRAST = 1.0e-3
GRID_SHAPE = (256, 256, 1)
MAGNITUDE = 0.01
GRADIENT_MAGNITUDE = MAGNITUDE / HOLE_RADIUS  # so magnitude * HOLE_RADIUS == MAGNITUDE


def _build_case():
    grid = Grid(shape=GRID_SHAPE, lengths=(1.0, 1.0, 1.0))
    smoothing_width = 2.0 * grid.spacing[0]
    return HoleInPlateCase(
        grid, matrix_lame_lambda=MATRIX_LAME_LAMBDA, matrix_lame_mu=MATRIX_LAME_MU,
        hole_radius=HOLE_RADIUS, contrast=CONTRAST, center=(0.5, 0.5),
        smoothing_width=smoothing_width,
    )


def _line_through_hole(case, solver, load):
    """Return (xi, sigma_xx, sigma_yy, sigma_xy) along the line x1=center[0],
    x2 varying -- normalized position xi=(x2-center[1])/hole_radius and
    stress by `MAGNITUDE`."""
    grid = case.grid
    eps_bar = case.macro_strain(load, MAGNITUDE, solver=solver)
    sol = solver.solve(eps_bar, tol=1.0e-6, max_iterations=3000)
    print(f"{load}: converged={sol.converged}, iterations={sol.iterations}, "
          f"residual={sol.residual_norm:.3g}")
    sigma = np.asarray(sol.stress)

    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    col = np.argmin(np.abs(x1 - case.center[0]))

    xi = (x2 - case.center[1]) / HOLE_RADIUS
    sigma_xx = sigma[0, 0, col, :, 0] / MAGNITUDE
    sigma_yy = sigma[1, 1, col, :, 0] / MAGNITUDE
    sigma_xy = sigma[0, 1, col, :, 0] / MAGNITUDE
    return xi, sigma_xx, sigma_yy, sigma_xy


def _periodic_analytic_line(case, load):
    """Return (xi, sigma_xx, sigma_yy, sigma_xy)/MAGNITUDE along the same
    line, from the exact periodic (Eshelby equivalent-eigenstrain +
    Khachaturyan-Shatalov Fourier) solution -- the solution to the actual
    periodic problem this grid represents.
    """
    grid = case.grid
    _, stress = case.periodic_analytic_solution(load, MAGNITUDE)
    stress = np.asarray(stress)

    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    col = np.argmin(np.abs(x1 - case.center[0]))

    xi = (x2 - case.center[1]) / HOLE_RADIUS
    sigma_xx = stress[0, 0, col, :, 0] / MAGNITUDE
    sigma_yy = stress[1, 1, col, :, 0] / MAGNITUDE
    sigma_xy = stress[0, 1, col, :, 0] / MAGNITUDE
    return xi, sigma_xx, sigma_yy, sigma_xy


def _pressurized_line(case, solver):
    """Return (xi, sigma_xx, sigma_yy)/MAGNITUDE along the same line for a
    hole under uniform internal pressure `MAGNITUDE` (zero remote stress) --
    built from the already-computed "biaxial" numerical solve by
    subtracting the uniform background stress ``MAGNITUDE * I``, which is
    valid regardless of the heterogeneous stiffness field since a spatially
    uniform stress trivially has zero divergence (see
    :meth:`crystallite.verification.HoleInPlateCase.periodic_pressurized_stress`).
    """
    xi, sxx, syy, _ = _line_through_hole(case, solver, "biaxial")
    return xi, sxx - 1.0, syy - 1.0


def _periodic_pressurized_line(case):
    """Return (xi, sigma_xx, sigma_yy)/MAGNITUDE along the same line, from
    :meth:`crystallite.verification.HoleInPlateCase.periodic_pressurized_stress`.
    """
    grid = case.grid
    stress = np.asarray(case.periodic_pressurized_stress(MAGNITUDE))

    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    col = np.argmin(np.abs(x1 - case.center[0]))

    xi = (x2 - case.center[1]) / HOLE_RADIUS
    sigma_xx = stress[0, 0, col, :, 0] / MAGNITUDE
    sigma_yy = stress[1, 1, col, :, 0] / MAGNITUDE
    return xi, sigma_xx, sigma_yy


def _moment_line(case, solver):
    """Return (xi, sigma_xx, sigma_yy)/(GRADIENT_MAGNITUDE * HOLE_RADIUS)
    along the same line, under the "moment" remote stress gradient
    ``d(sigma_11)/dx2 = GRADIENT_MAGNITUDE`` (zero macro strain, only the
    gradient background) -- solved via
    :meth:`crystallite.elastic_deformation.ElasticDeformation.solve`'s
    `macro_strain_gradient` support, centered at the hole via
    `gradient_origin`.
    """
    grid = case.grid
    eps_gradient = case.macro_strain_gradient("moment", GRADIENT_MAGNITUDE, solver=solver)
    origin = (case.center[0], case.center[1], 0.0)
    sol = solver.solve(
        np.zeros((3, 3)), tol=1.0e-6, max_iterations=3000,
        macro_strain_gradient=eps_gradient, gradient_origin=origin,
    )
    print(f"moment: converged={sol.converged}, iterations={sol.iterations}, "
          f"residual={sol.residual_norm:.3g}")
    sigma = np.asarray(sol.stress)

    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    col = np.argmin(np.abs(x1 - case.center[0]))

    xi = (x2 - case.center[1]) / HOLE_RADIUS
    scale = GRADIENT_MAGNITUDE * HOLE_RADIUS
    sigma_xx = sigma[0, 0, col, :, 0] / scale
    sigma_yy = sigma[1, 1, col, :, 0] / scale
    return xi, sigma_xx, sigma_yy


def _periodic_moment_line(case):
    """Return (xi, sigma_xx, sigma_yy)/(GRADIENT_MAGNITUDE * HOLE_RADIUS)
    along the same line, from
    :meth:`crystallite.verification.HoleInPlateCase.periodic_gradient_stress`
    -- an image sum of the exact traction-free-void closed form, not an
    Eshelby equivalent-eigenstrain construction (see that method).
    """
    grid = case.grid
    stress = np.asarray(case.periodic_gradient_stress(GRADIENT_MAGNITUDE))

    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    col = np.argmin(np.abs(x1 - case.center[0]))

    xi = (x2 - case.center[1]) / HOLE_RADIUS
    scale = GRADIENT_MAGNITUDE * HOLE_RADIUS
    sigma_xx = stress[0, 0, col, :, 0] / scale
    sigma_yy = stress[1, 1, col, :, 0] / scale
    return xi, sigma_xx, sigma_yy


def plot_moment():
    case = _build_case()
    solver = case.solver()
    xi, sxx, syy = _moment_line(case, solver)
    _, pxx, pyy = _periodic_moment_line(case)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.plot(xi, pxx, color="C0", label=r"$\sigma_{11}$ analytical")
    ax.plot(xi[::4], sxx[::4], "o", color="C0", markersize=3, label=r"$\sigma_{11}$ numerical")
    ax.plot(xi, pyy, color="C1", label=r"$\sigma_{22}$ analytical")
    ax.plot(xi[::4], syy[::4], "o", color="C1", markersize=3, label=r"$\sigma_{22}$ numerical")
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"$x_2 / r_0$")
    ax.set_ylabel(r"$\sigma / (k r_0)$")
    ax.set_title("moment")
    ax.legend(fontsize=7, ncol=2)
    save_all(fig, "elastic_deformation.hole_moment")
    plt.close(fig)


def plot_pressure():
    case = _build_case()
    solver = case.solver()
    xi, sxx, syy = _pressurized_line(case, solver)
    _, pxx, pyy = _periodic_pressurized_line(case)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.plot(xi, pxx, color="C0", label=r"$\sigma_{11}$ analytical")
    ax.plot(xi[::4], sxx[::4], "o", color="C0", markersize=3, label=r"$\sigma_{11}$ numerical")
    ax.plot(xi, pyy, color="C1", label=r"$\sigma_{22}$ analytical")
    ax.plot(xi[::4], syy[::4], "o", color="C1", markersize=3, label=r"$\sigma_{22}$ numerical")
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"$x_2 / r_0$")
    ax.set_ylabel(r"$\sigma / p_\infty$")
    ax.set_title("pressure")
    ax.legend(fontsize=7, ncol=2)
    save_all(fig, "elastic_deformation.hole_pressure")
    plt.close(fig)


def plot_tension_or_compression(load):
    case = _build_case()
    solver = case.solver()
    xi, sxx, syy, sxy = _line_through_hole(case, solver, load)
    _, pxx, pyy, _ = _periodic_analytic_line(case, load)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.plot(xi, pxx, color="C0", label=r"$\sigma_{11}$ analytical")
    ax.plot(xi[::4], sxx[::4], "o", color="C0", markersize=3, label=r"$\sigma_{11}$ numerical")
    ax.plot(xi, pyy, color="C1", label=r"$\sigma_{22}$ analytical")
    ax.plot(xi[::4], syy[::4], "o", color="C1", markersize=3, label=r"$\sigma_{22}$ numerical")
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"$x_2 / r_0$")
    ax.set_ylabel(r"$\sigma / \sigma_\infty$")
    ax.set_title(load)
    ax.legend(fontsize=7, ncol=2)
    save_all(fig, f"elastic_deformation.hole_{load}")
    plt.close(fig)


def plot_biaxial():
    case = _build_case()
    solver = case.solver()
    xi, sxx, syy, sxy = _line_through_hole(case, solver, "biaxial")
    _, pxx, pyy, _ = _periodic_analytic_line(case, "biaxial")

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.plot(xi, pxx, color="C0", label=r"$\sigma_{11}$ analytical")
    ax.plot(xi[::4], sxx[::4], "o", color="C0", markersize=3, label=r"$\sigma_{11}$ numerical")
    ax.plot(xi, pyy, color="C1", label=r"$\sigma_{22}$ analytical")
    ax.plot(xi[::4], syy[::4], "o", color="C1", markersize=3, label=r"$\sigma_{22}$ numerical")
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"$x_2 / r_0$")
    ax.set_ylabel(r"$\sigma / \sigma_\infty$")
    ax.set_title("biaxial")
    ax.legend(fontsize=7, ncol=2)
    save_all(fig, "elastic_deformation.hole_biaxial")
    plt.close(fig)


def plot_shear():
    case = _build_case()
    solver = case.solver()
    xi, sxx, syy, sxy = _line_through_hole(case, solver, "shear")
    _, _, _, pxy = _periodic_analytic_line(case, "shear")

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.plot(xi, pxy, color="C0", label=r"$\sigma_{12}$ analytical")
    ax.plot(xi[::4], sxy[::4], "o", color="C0", markersize=3, label=r"$\sigma_{12}$ numerical")
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"$x_2 / r_0$")
    ax.set_ylabel(r"$\sigma / \sigma_\infty$")
    ax.set_title("shear")
    ax.legend(fontsize=8)
    save_all(fig, "elastic_deformation.hole_shear")
    plt.close(fig)


if __name__ == "__main__":
    plot_tension_or_compression("tension")
    plot_tension_or_compression("compression")
    plot_biaxial()
    plot_pressure()
    plot_moment()
    plot_shear()
