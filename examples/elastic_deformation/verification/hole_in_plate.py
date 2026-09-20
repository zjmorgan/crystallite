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

The analytic curve shown against the numerical points is built by summing
the isolated closed-form, finite-contrast inhomogeneity solution
(:meth:`crystallite.verification.HoleInPlateCase.periodic_inhomogeneity_stress`)
over periodic images of the hole, up to an `N_IMAGES` cutoff, in place of
:meth:`crystallite.verification.HoleInPlateCase.periodic_analytic_solution`'s
Eshelby-eigenstrain FFT construction (still used by
``cylindrical_inclusion.py`` for genuine finite-contrast inhomogeneities).
This is this project's own generalization of the original (pre-crystallite)
``eshelby.cpp`` reference implementation's own recipe -- that file's
``e()``/``f()``/``g()`` functions already solve for and image-sum the
*finite-contrast* Eshelby field in general; its own worked ``main()``
example just happens to set the inhomogeneity's modulus to zero (a literal
void). :meth:`~crystallite.verification.HoleInPlateCase.periodic_void_stress`
(the void-only predecessor of `periodic_inhomogeneity_stress` used here
until this module was generalized to match `CONTRAST`) remains available
for a literal void.

The tradeoff is deliberate: the image sum never touches the grid's discrete
Fourier machinery, so it is completely free of that construction's
finite-grid Gibbs ringing -- but for the uniform loads here (``tension``,
``compression``, ``biaxial``, ``shear``, the ``pressure`` case built from
``biaxial``) it is only a dilute-limit *approximation*, not an exact
periodic solution: it still differs from the FFT reference/numeric solver
by roughly 5-8% in this module's own geometry (``HOLE_RADIUS=0.1`` in a
unit cell, ~3% hole area fraction), because it cannot capture the periodic
array's hole-to-hole self-interaction -- see
:meth:`~crystallite.verification.HoleInPlateCase._periodic_image_sum`'s
docstring for why. Using the finite-contrast field here instead of the
void-only one only fixes the (here, negligible, since `CONTRAST` is
already near-void) void-vs-actual-`contrast` mismatch -- it does *not*
reduce this periodicity bias, which dominates the residual gap; verified
directly by comparing both curves against this module's own numeric
solve, which come out within floating noise of each other. The ``moment``
case
(:meth:`crystallite.verification.HoleInPlateCase.periodic_gradient_stress`)
has no such bias and is unaffected: its background has zero leading-order
Eshelby response by symmetry, so there is no self-interaction for images to
miss. In short: these curves favor a smooth, ringing-free view of the
concentration/decay *shape* over tight quantitative agreement with the
numerical points for the uniform-load cases; expect the numerical markers
to sit visibly, not just noisily, off the analytic curve there.

The hole is a soft circular inclusion
(:class:`crystallite.verification.HoleInPlateCase`) with a diffuse (tanh)
boundary, evolved to equilibrium by
:class:`crystallite.elastic_deformation.ElasticDeformation`.
"""

from __future__ import annotations

import numpy as np

from _plotting import analytic_numeric_curve, component_legend, plt, save_all
from crystallite.grid import Grid
from crystallite.verification import HoleInPlateCase

MATRIX_LAME_LAMBDA = 1.0
MATRIX_LAME_MU = 0.7
HOLE_RADIUS = 0.1
CONTRAST = 1.0e-3
GRID_SHAPE = (256, 256, 1)
MAGNITUDE = 0.01
GRADIENT_MAGNITUDE = MAGNITUDE / HOLE_RADIUS  # so magnitude * HOLE_RADIUS == MAGNITUDE
N_IMAGES = 3  # periodic image cutoff for the analytic curves (already converged here)


def _build_case():
    grid = Grid(shape=GRID_SHAPE, lengths=(1.0, 1.0, 1.0))
    return HoleInPlateCase(
        grid, matrix_lame_lambda=MATRIX_LAME_LAMBDA, matrix_lame_mu=MATRIX_LAME_MU,
        hole_radius=HOLE_RADIUS, contrast=CONTRAST, center=(0.5, 0.5),
        dealias=True,
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
    line, from image-summing the isolated, finite-contrast closed form
    (:meth:`~crystallite.verification.HoleInPlateCase.periodic_inhomogeneity_stress`,
    this project's own generalization of the ``eshelby.cpp`` reference's
    image-sum recipe to a real `CONTRAST`, not just its void-limit worked
    example) up to `N_IMAGES` -- a smooth, ringing-free dilute-limit
    *approximation* of the actual periodic problem this grid represents
    (see this module's own docstring for the residual bias this carries
    for uniform loads).
    """
    grid = case.grid
    stress = case.periodic_inhomogeneity_stress(load, MAGNITUDE, n_images=N_IMAGES)
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
    :meth:`crystallite.verification.HoleInPlateCase.periodic_inhomogeneity_pressurized_stress`
    -- the finite-`contrast` generalization of
    :meth:`~crystallite.verification.HoleInPlateCase.periodic_void_pressurized_stress`,
    matching the other load cases in this module (see
    :func:`_periodic_analytic_line`): this case's numeric solve uses the
    same soft (``CONTRAST=1.0e-3``), not literally void, hole those cases
    do, so the void-only closed form left the same void-vs-actual-contrast
    mismatch here.
    """
    grid = case.grid
    stress = np.asarray(
        case.periodic_inhomogeneity_pressurized_stress(MAGNITUDE, n_images=N_IMAGES)
    )

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
    ax.set_xlim(-4, 4)
    analytic_numeric_curve(ax, xi, pxx, sxx, r"$\sigma_{11}$", "C0")
    analytic_numeric_curve(ax, xi, pyy, syy, r"$\sigma_{22}$", "C1")
    ax.set_xlabel(r"$x_2 / r_0$")
    ax.set_ylabel(r"$\sigma / (k r_0)$")
    component_legend(ax)
    save_all(fig, "elastic_deformation.hole_moment")
    plt.close(fig)


def plot_pressure():
    case = _build_case()
    solver = case.solver()
    xi, sxx, syy = _pressurized_line(case, solver)
    _, pxx, pyy = _periodic_pressurized_line(case)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.set_xlim(-4, 4)
    analytic_numeric_curve(ax, xi, pxx, sxx, r"$\sigma_{11}$", "C0")
    analytic_numeric_curve(ax, xi, pyy, syy, r"$\sigma_{22}$", "C1")
    ax.set_xlabel(r"$x_2 / r_0$")
    ax.set_ylabel(r"$\sigma / p_\infty$")
    component_legend(ax)
    save_all(fig, "elastic_deformation.hole_pressure")
    plt.close(fig)


def plot_tension_or_compression(load):
    case = _build_case()
    solver = case.solver()
    xi, sxx, syy, sxy = _line_through_hole(case, solver, load)
    _, pxx, pyy, _ = _periodic_analytic_line(case, load)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.set_xlim(-4, 4)
    analytic_numeric_curve(ax, xi, pxx, sxx, r"$\sigma_{11}$", "C0")
    analytic_numeric_curve(ax, xi, pyy, syy, r"$\sigma_{22}$", "C1")
    ax.set_xlabel(r"$x_2 / r_0$")
    ax.set_ylabel(r"$\sigma / \sigma_\infty$")
    component_legend(ax)
    save_all(fig, f"elastic_deformation.hole_{load}")
    plt.close(fig)


def plot_biaxial():
    case = _build_case()
    solver = case.solver()
    xi, sxx, syy, sxy = _line_through_hole(case, solver, "biaxial")
    _, pxx, pyy, _ = _periodic_analytic_line(case, "biaxial")

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.set_xlim(-4, 4)
    analytic_numeric_curve(ax, xi, pxx, sxx, r"$\sigma_{11}$", "C0")
    analytic_numeric_curve(ax, xi, pyy, syy, r"$\sigma_{22}$", "C1")
    ax.set_xlabel(r"$x_2 / r_0$")
    ax.set_ylabel(r"$\sigma / \sigma_\infty$")
    component_legend(ax)
    save_all(fig, "elastic_deformation.hole_biaxial")
    plt.close(fig)


def plot_shear():
    case = _build_case()
    solver = case.solver()
    xi, sxx, syy, sxy = _line_through_hole(case, solver, "shear")
    _, _, _, pxy = _periodic_analytic_line(case, "shear")

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.set_xlim(-4, 4)
    analytic_numeric_curve(ax, xi, pxy, sxy, r"$\sigma_{12}$", "C0")
    ax.set_xlabel(r"$x_2 / r_0$")
    ax.set_ylabel(r"$\sigma / \sigma_\infty$")
    component_legend(ax)
    save_all(fig, "elastic_deformation.hole_shear")
    plt.close(fig)


if __name__ == "__main__":
    plot_tension_or_compression("tension")
    plot_tension_or_compression("compression")
    plot_biaxial()
    plot_pressure()
    plot_moment()
    plot_shear()
