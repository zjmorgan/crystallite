"""Stress near the edge of an elliptical hole, under remote tension,
shear, and internal pressure, as a function of the hole's aspect ratio --
the classical Inglis-type verification (peak/edge stress vs. shape), not
a spatial profile at one fixed shape.

For each aspect ratio ``b/a`` (semi-axis `b` perpendicular to the tension
load over semi-axis `a` parallel to it), the hole is a soft elliptical
inclusion (:class:`crystallite.verification.EllipticalHoleInPlateCase`)
with a diffuse (tanh) boundary, evolved to equilibrium by
:class:`crystallite.elastic_deformation.ElasticDeformation`; the numeric
stress is read off a small fixed physical distance
(`NUMERIC_EDGE_OFFSET`) outside the boundary, along the true outward
normal at a chosen boundary point -- an absolute distance, not one scaled
by `b`, since the stress actually decays over a length set by the *local
radius of curvature* there (varies with position and aspect ratio, e.g.
a**2/b at the b-axis tip), not by `b` itself; a `b`-scaled offset
overshoots that decay length for `b/a` >> 1 and undershoots it for
`b/a` << 1, producing a spurious non-monotonic curve unrelated to the
physics (an earlier version of this script did exactly that). The
analytic reference is
:meth:`EllipticalHoleInPlateCase.boundary_stress`, evaluated directly at
the boundary point (point-wise, not grid-interpolated -- interpolating
the grid field there is unstable, landing right on its "zero inside the
void" mask discontinuity depending on which side neighboring grid points
happen to fall -- an earlier version of this script hit exactly that).
Both close over the exact isolated Kirsch/Inglis closed form
(:func:`crystallite.verification.elastic_deformation._ellipse_void_cartesian_stress`)
summed over periodic images of the hole (this project has no elliptical
Eshelby tensor implemented, so this -- not an eigenstrain construction --
is how periodicity is handled here; see that method's docstring).

That image sum is only as good as the "sum of isolated single-hole
fields" approximation itself, which misses real elastic interaction
between periodic images -- confirmed by a controlled resolution sweep to
be the dominant remaining error (not periodic-image truncation, not
diffuse-boundary width, not the finite hole/matrix stiffness contrast):
holding grid resolution and boundary sharpness fixed and varying only the
hole-to-domain size ratio, numeric/analytic agreement went from ~94% at a
semi-axis 15% of the domain to ~99% at 5%. `CHARACTERISTIC_SIZE` below is
chosen accordingly, small enough that this stays a minor effect across
the whole aspect-ratio sweep.

Tension and pressure are sampled at the tip of the b semi-axis
(theta=pi/2), where their concentration peaks by symmetry. Shear has no
such peak there -- at theta=0 or pi/2 its boundary stress is exactly
zero, by the same symmetry that makes tension's vanish at theta=0 -- so
it is sampled at theta=pi/4 instead (the peak for a circular hole; not
exactly the peak once the hole is eccentric, but a fixed, consistent
reference point across the sweep, in the same spirit as tension/pressure's
fixed theta=pi/2).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from _plotting import plt, save_all
from crystallite.grid import Grid
from crystallite.verification import EllipticalHoleInPlateCase

MATRIX_LAME_LAMBDA = 1.0
MATRIX_LAME_MU = 0.7
# sqrt(a*b), held fixed across the sweep so both semi-axes stay small
# relative to the domain (dilute periodic images) and to each other
# (resolvable on the grid) at every aspect ratio.
CHARACTERISTIC_SIZE = 0.035
CONTRAST = 1.0e-3
GRID_SHAPE = (512, 512, 1)
MAGNITUDE = 0.01
ASPECT_RATIOS = np.array([0.25, 0.5, 1.0, 2.0, 4.0])  # b/a

# How far outside the boundary to read the *numerical* stress, as an
# absolute physical distance -- see the module docstring for why this is
# not normalized by semi_axis_b. Must still clear the diffuse transition:
# the closest grid point just outside the boundary is partway through the
# smoothed soft-to-matrix transition (its properties haven't reached the
# true matrix yet), which badly *under*-reports the peak.
NUMERIC_EDGE_OFFSET = 0.008


def _semi_axes(aspect_ratio):
    """(a, b) with a*b = CHARACTERISTIC_SIZE**2 and b/a = aspect_ratio."""
    a = CHARACTERISTIC_SIZE / np.sqrt(aspect_ratio)
    b = CHARACTERISTIC_SIZE * np.sqrt(aspect_ratio)
    return a, b


def _build_case(a, b):
    grid = Grid(shape=GRID_SHAPE, lengths=(1.0, 1.0, 1.0))
    smoothing_width = 2.0 * grid.spacing[0] / b
    return EllipticalHoleInPlateCase(
        grid, matrix_lame_lambda=MATRIX_LAME_LAMBDA, matrix_lame_mu=MATRIX_LAME_MU,
        semi_axis_a=a, semi_axis_b=b, contrast=CONTRAST,
        center=(0.5, 0.5), smoothing_width=smoothing_width,
    )


def _boundary_point_and_normal(case, theta):
    """(x, y) on the ellipse boundary at parametric angle `theta`
    (absolute grid coordinates), and the true outward unit normal there
    (not generally the radial direction, off the semi-axis tips)."""
    a, b = case.semi_axis_a, case.semi_axis_b
    x = case.center[0] + a * np.cos(theta)
    y = case.center[1] + b * np.sin(theta)
    nx, ny = np.cos(theta) / a, np.sin(theta) / b
    norm = np.hypot(nx, ny)
    return x, y, nx / norm, ny / norm


def _near_edge_numeric(case, solver, load, theta):
    """Numeric (sigma_xx, sigma_yy, sigma_xy)/MAGNITUDE at
    NUMERIC_EDGE_OFFSET beyond the boundary point at `theta`, along its
    true outward normal (bilinearly interpolated off the solved field)."""
    eps_bar = case.macro_strain(load, MAGNITUDE, solver=solver)
    sol = solver.solve(eps_bar, tol=1.0e-6, max_iterations=8000)
    sigma = np.asarray(sol.stress)

    grid = case.grid
    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    xb, yb, nx, ny = _boundary_point_and_normal(case, theta)
    point = np.array([[xb + NUMERIC_EDGE_OFFSET * nx, yb + NUMERIC_EDGE_OFFSET * ny]])

    interp_xx = RegularGridInterpolator((x1, x2), sigma[0, 0, :, :, 0])
    interp_yy = RegularGridInterpolator((x1, x2), sigma[1, 1, :, :, 0])
    interp_xy = RegularGridInterpolator((x1, x2), sigma[0, 1, :, :, 0])
    sigma_xx = float(interp_xx(point)[0]) / MAGNITUDE
    sigma_yy = float(interp_yy(point)[0]) / MAGNITUDE
    sigma_xy = float(interp_xy(point)[0]) / MAGNITUDE
    return sigma_xx, sigma_yy, sigma_xy, sol.converged, sol.iterations


def _edge_analytic(case, load, theta):
    """Analytic (sigma_xx, sigma_yy, sigma_xy)/MAGNITUDE exactly at the
    boundary point `theta`, from
    :meth:`EllipticalHoleInPlateCase.boundary_stress` (point-wise, not
    grid-interpolated -- see the module docstring)."""
    sigma_xx, sigma_yy, sigma_xy = case.boundary_stress(load, MAGNITUDE, theta=theta)
    return float(sigma_xx) / MAGNITUDE, float(sigma_yy) / MAGNITUDE, float(sigma_xy) / MAGNITUDE


def _pressure_numeric(case, solver, theta):
    """Numeric (sigma_xx, sigma_yy)/MAGNITUDE for uniform internal
    pressure MAGNITUDE (zero remote stress) -- built from the "biaxial"
    numerical solve by subtracting the uniform background stress
    MAGNITUDE*I, exactly as ``hole_in_plate.py``'s pressure case does."""
    sxx, syy, _, converged, iterations = _near_edge_numeric(case, solver, "biaxial", theta)
    return sxx - 1.0, syy - 1.0, converged, iterations


def _pressure_analytic(case, theta):
    sxx, syy, _ = _edge_analytic(case, "biaxial", theta)
    return sxx - 1.0, syy - 1.0


def _sweep(load, theta, extract_numeric, extract_analytic, labels, title, filename):
    """Shared driver: run the numeric solve and analytic evaluation at
    each aspect ratio, plus a smooth analytic reference curve, and save
    the comparison figure."""
    numeric_a, numeric_b = [], []
    analytic_a, analytic_b = [], []
    for ratio in ASPECT_RATIOS:
        a, b = _semi_axes(ratio)
        case = _build_case(a, b)
        solver = case.solver()
        va, vb, converged, iterations = extract_numeric(case, solver, theta)
        pa, pb = extract_analytic(case, theta)
        print(f"{load} b/a={ratio}: converged={converged} iterations={iterations} "
              f"numeric=({va:.3f},{vb:.3f}) analytic_at_boundary=({pa:.3f},{pb:.3f})")
        numeric_a.append(va)
        numeric_b.append(vb)
        analytic_a.append(pa)
        analytic_b.append(pb)

    fine_ratios = np.geomspace(ASPECT_RATIOS.min(), ASPECT_RATIOS.max(), 100)
    fine_a, fine_b = [], []
    for ratio in fine_ratios:
        a, b = _semi_axes(ratio)
        case = _build_case(a, b)
        pa, pb = extract_analytic(case, theta)
        fine_a.append(pa)
        fine_b.append(pb)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.plot(fine_ratios, fine_a, color="C0", label=f"{labels[0]} analytical (at boundary)")
    ax.plot(ASPECT_RATIOS, numeric_a, "o", color="C0", markersize=6, label=f"{labels[0]} numerical")
    ax.plot(fine_ratios, fine_b, color="C1", label=f"{labels[1]} analytical (at boundary)")
    ax.plot(ASPECT_RATIOS, numeric_b, "o", color="C1", markersize=6, label=f"{labels[1]} numerical")
    ax.set_xscale("log")
    ax.set_xlabel(r"$b / a$")
    ax.set_ylabel(r"$\sigma / \sigma_\infty$")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    save_all(fig, filename)
    plt.close(fig)


def _tension_numeric(case, solver, theta):
    sxx, syy, _, converged, iterations = _near_edge_numeric(case, solver, "tension", theta)
    return sxx, syy, converged, iterations


def _tension_analytic(case, theta):
    sxx, syy, _ = _edge_analytic(case, "tension", theta)
    return sxx, syy


def _shear_numeric(case, solver, theta):
    sxx, _, sxy, converged, iterations = _near_edge_numeric(case, solver, "shear", theta)
    return sxx, sxy, converged, iterations


def _shear_analytic(case, theta):
    sxx, _, sxy = _edge_analytic(case, "shear", theta)
    return sxx, sxy


def plot_tension():
    _sweep(
        "tension", np.pi / 2, _tension_numeric, _tension_analytic,
        (r"$\sigma_{11}$", r"$\sigma_{22}$"),
        "elliptical hole, tension: edge stress vs. aspect ratio",
        "elastic_deformation.elliptical_hole_aspect_ratio_tension",
    )


def plot_shear():
    _sweep(
        "shear", np.pi / 4, _shear_numeric, _shear_analytic,
        (r"$\sigma_{11}$", r"$\sigma_{12}$"),
        "elliptical hole, shear: edge stress vs. aspect ratio",
        "elastic_deformation.elliptical_hole_aspect_ratio_shear",
    )


def plot_pressure():
    _sweep(
        "pressure", np.pi / 2, _pressure_numeric, _pressure_analytic,
        (r"$\sigma_{11}$", r"$\sigma_{22}$"),
        "elliptical hole, pressure: edge stress vs. aspect ratio",
        "elastic_deformation.elliptical_hole_aspect_ratio_pressure",
    )


if __name__ == "__main__":
    plot_tension()
    plot_shear()
    plot_pressure()
