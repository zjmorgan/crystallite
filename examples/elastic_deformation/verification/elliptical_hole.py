"""Maximum (principal) stress around an elliptical hole, under remote
tension, shear, and a remote stress gradient ("moment"/bending loading),
as a function of the hole's aspect ratio -- the classical Inglis-type
verification (peak stress vs. shape), not a spatial profile at one fixed
shape. Matches this project's original (pre-crystallite) reference
material's own versions of these three figures (``hooke.tex``'s
``elliptical.hole``/``elliptical.hole.shear``/``elliptical.hole.moment``
frames), reverse-engineered directly from that reference's own plotting
script (``hooke.py``) after an earlier hoop-stress-based version of this
module was found not to match it.

Three things changed from that earlier version, all found by reading
``hooke.py``'s own tension/shear/moment functions directly rather than
guessing from the figures alone:

- **The semi-axis convention is swapped.** ``hooke.py`` sweeps a variable
  it labels ``b/a`` and plots tension as ``f(r) = 1 + 2/r``. Checked
  directly against this project's own already-validated closed form
  (:func:`crystallite.verification.elastic_deformation._ellipse_void_cartesian_stress`,
  whose own semantics are ``a`` = semi-axis parallel to the tension load,
  ``b`` = perpendicular, concentration growing as ``1 + 2*(b/a)``):
  reproducing ``1 + 2/r`` exactly requires feeding that closed form
  ``b/a = 1/r``, i.e. the swept ``r`` is this project's own ``a/b``, not
  ``b/a`` -- the two codebases' ``a``/``b`` labels are effectively swapped
  relative to each other. ``_semi_axes`` below is written for this
  project's own closed form, so it assigns ``a = CHARACTERISTIC_SIZE *
  sqrt(ratio)`` (grows with ``ratio``) and ``b = CHARACTERISTIC_SIZE /
  sqrt(ratio)`` (shrinks with it) -- verified to reproduce ``1 + 2/ratio``
  (tension) and ``(ratio+1)**2/ratio`` (shear, see below) to 4 significant
  figures at every one of the 7 swept ratios, using the closed form alone,
  no grid solve involved.
- **The tracked quantity is principal stress, not hoop stress.** An even
  earlier version read one fixed Cartesian component at a fixed boundary
  angle; the version right before this one switched to hoop stress
  (:math:`\\sigma_{\\theta\\theta}`, from the true local normal/tangent at
  each swept boundary point) reasoning that a traction-free boundary
  singles it out physically. That reasoning is correct as far as it
  goes -- right at the boundary, where :math:`\\sigma_{rr}=0` exactly, hoop
  stress and the larger-magnitude principal stress coincide -- but it
  wasn't the source of the actual mismatch (the semi-axis swap above was),
  and ``hooke.py`` itself never computes a hoop stress at all: its own
  shear case (``elliptical_hole_shear()``) explicitly builds principal
  stresses from the raw Cartesian field
  (:math:`\\sigma_{I,II}=\\tfrac12(\\sigma_{11}+\\sigma_{22})\\pm\\sqrt{\\tfrac14(\\sigma_{11}-\\sigma_{22})^2+\\sigma_{12}^2}`)
  and takes their max/min. Reading principal stress directly needs no
  local-normal geometry at all (it's basis-independent), so it's what this
  version uses -- still sampled at boundary points located via the true
  outward normal (:func:`_boundary_points_and_normals`, unchanged), since
  that part was never in question, only the stress formula built from what
  gets interpolated there.
- **Moment is normalized locally, not globally.** ``hooke.py``'s own
  moment figure is labelled :math:`\\sigma/(\\gamma_\\infty b)` -- it
  divides each aspect ratio's raw stress by *that ellipse's own physical
  semi-axis* ``b``, not a single fixed scale. That isn't a labeling quirk:
  a uniform remote load's stress concentration depends only on shape
  (aspect ratio), but a stress-*gradient* ("moment") load's stress
  amplitude also depends on the absolute size of the feature it bends
  around (stress ~ gradient x length scale) -- so a fixed global
  normalization (this module's own ``MAGNITUDE``, held constant across the
  sweep) bakes in a spurious absolute-size confound that tension/shear
  never had, since holding ``sqrt(a*b)`` fixed across the sweep (needed
  here for dilute periodic images and grid resolution, see
  ``CHARACTERISTIC_SIZE``) doesn't hold either individual semi-axis fixed.
  ``hooke.py``'s own exact absolute-size progression for this figure ties
  back to its own VTK grid setup and isn't recoverable from its plotting
  script alone, so this module does not attempt to reproduce its specific
  numbers -- instead it applies the same physical correction (divide by
  ``GRADIENT_MAGNITUDE * case.semi_axis_b``, that case's own semi-axis, in
  place of the constant ``MAGNITUDE``) within this project's own
  established ``sqrt(a*b)``-fixed geometry, and checks internal
  numeric-vs-analytic agreement rather than matching ``hooke.py``'s literal
  curve -- the same standard every other analytic reference in this file
  is held to.

For tension, only the maximum principal stress is tracked (one curve,
always positive) -- matching ``hooke.py``'s own tension figure, which only
ever shows one. Shear and moment track *both* the maximum and minimum
principal stress (two curves, one always positive and one always
negative) -- matching ``hooke.py``'s own ``s``/``t`` (max/min) pair for
both of those figures.

The moment case uses
:meth:`EllipticalHoleInPlateCase.periodic_gradient_inhomogeneity_stress`:
the exact isolated finite-contrast solution (Muskhelishvili potentials
under the ellipse's conformal map, exactly quadratic inside), summed over
periodic images -- no FFT and no numerically probed Eshelby tensor, so
the analytic curve carries no grid-discretization error of its own. An
earlier version used an Eshelby-dipole eigenstrain construction
(since removed);
its curve was noisy in aspect ratio (hard-mask pixelization and
probe-calibration error) and its claimed 2% agreement with the numeric
solver did not hold (4-25% observed).

For each aspect ratio, the hole is a soft elliptical inclusion
(:class:`crystallite.verification.EllipticalHoleInPlateCase`) with a
sharp, Lanczos-dealiased boundary, evolved to equilibrium by
:class:`crystallite.elastic_deformation.ElasticDeformation`; both the
numeric and analytic stress are read off the same small fixed physical
distance (`NUMERIC_EDGE_OFFSET`) outside the boundary, along the true
outward normal at each swept boundary point -- an absolute distance, not
one scaled by `b`, since the stress actually decays over a length set by
the *local radius of curvature* there (varies with position and aspect
ratio, e.g. a**2/b at the b-axis tip), not by `b` itself; a `b`-scaled
offset overshoots that decay length for large ratios and undershoots it
for small ones, producing a spurious non-monotonic curve unrelated to the
physics (an earlier version of this script did exactly that).

Pressure is not part of this sweep: ``hooke.py``'s own
``elliptical_hole_pressure()`` is captioned "stress *across* an elliptic
cylindrical hole" and plotted against position (x2/r0 in [-4, 4]), not
aspect ratio -- a spatial profile at one fixed shape, the same kind of
figure ``hole_in_plate.py``'s own pressure case is, not a
max-vs-aspect-ratio sweep. It has its own function here,
:func:`plot_biaxial_profile`. That figure is a traction-free hole under
remote tension in both directions (not an internally pressurized one).

``ASPECT_RATIOS`` matches ``hooke.py``'s own sweep exactly:
``0.125, 0.25, 0.5, 1, 2, 4, 8``.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from _plotting import analytic_numeric_curve, component_legend, plt, save_all
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
GRADIENT_MAGNITUDE = MAGNITUDE / CHARACTERISTIC_SIZE
ASPECT_RATIOS = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])  # a/b, see module docstring
N_IMAGES = 1  # periodic image cutoff for the analytic curves (already converged here)
N_THETA = 360  # boundary angles swept per aspect ratio to find the true principal-stress extrema

# How far outside the boundary to read the *numerical* stress, as an
# absolute physical distance -- see the module docstring for why this is
# not normalized by semi_axis_b. Must still clear the dealiased edge:
# the closest grid point just outside the boundary is partway through the
# smoothed soft-to-matrix transition (its properties haven't reached the
# true matrix yet), which badly *under*-reports the peak.
NUMERIC_EDGE_OFFSET = 0.008
APEX_OFFSET = 0.002  # distance beyond the apex at which sigma_11 is read (~1 grid cell, clear of the dealiased edge)

_THETAS = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)


def _semi_axes(aspect_ratio):
    """(a, b) with a*b = CHARACTERISTIC_SIZE**2 and a/b = aspect_ratio --
    see module docstring for why this is a/b, not b/a."""
    a = CHARACTERISTIC_SIZE * np.sqrt(aspect_ratio)
    b = CHARACTERISTIC_SIZE / np.sqrt(aspect_ratio)
    return a, b


def _build_case(a, b):
    grid = Grid(shape=GRID_SHAPE, lengths=(1.0, 1.0, 1.0))
    return EllipticalHoleInPlateCase(
        grid, matrix_lame_lambda=MATRIX_LAME_LAMBDA, matrix_lame_mu=MATRIX_LAME_MU,
        semi_axis_a=a, semi_axis_b=b, contrast=CONTRAST,
        center=(0.5, 0.5), dealias=True,
    )


def _boundary_points_and_normals(case, thetas):
    """(x, y) on the ellipse boundary at parametric angles `thetas`
    (absolute grid coordinates), and the true outward unit normal there
    (not generally the radial direction, off the semi-axis tips)."""
    a, b = case.semi_axis_a, case.semi_axis_b
    x = case.center[0] + a * np.cos(thetas)
    y = case.center[1] + b * np.sin(thetas)
    nx, ny = np.cos(thetas) / a, np.sin(thetas) / b
    norm = np.hypot(nx, ny)
    return x, y, nx / norm, ny / norm


def _principal_stresses(sigma_xx, sigma_yy, sigma_xy):
    r"""Max/min in-plane principal stress (eigenvalues of the local 2x2
    Cartesian stress tensor) -- basis-independent, unlike hoop stress, so
    it needs no local normal/tangent frame; matches
    ``hooke.py``'s own ``elliptical_hole_shear()`` construction exactly."""
    center = 0.5 * (sigma_xx + sigma_yy)
    radius = np.sqrt(0.25 * (sigma_xx - sigma_yy) ** 2 + sigma_xy ** 2)
    return center + radius, center - radius


def _interpolate_stress_at_boundary(case, sigma_xx_field, sigma_yy_field, sigma_xy_field, thetas):
    """Raw (undivided) Cartesian stress at NUMERIC_EDGE_OFFSET beyond every
    boundary point in `thetas`, along each one's own true outward normal
    (bilinearly interpolated off the given fields)."""
    grid = case.grid
    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    xb, yb, nx, ny = _boundary_points_and_normals(case, thetas)
    points = np.stack([xb + NUMERIC_EDGE_OFFSET * nx, yb + NUMERIC_EDGE_OFFSET * ny], axis=-1)

    interp_xx = RegularGridInterpolator((x1, x2), sigma_xx_field)
    interp_yy = RegularGridInterpolator((x1, x2), sigma_yy_field)
    interp_xy = RegularGridInterpolator((x1, x2), sigma_xy_field)
    return interp_xx(points), interp_yy(points), interp_xy(points)


def _principal_numeric_sweep(case, solver, load):
    """(max, min) principal-stress/MAGNITUDE arrays over `_THETAS`, from a
    single equilibrium solve under remote `load` ("tension" or "shear")."""
    eps_bar = case.macro_strain(load, MAGNITUDE, solver=solver)
    sol = solver.solve(eps_bar, tol=1.0e-6, max_iterations=8000)
    sigma = np.asarray(sol.stress)
    sxx, syy, sxy = _interpolate_stress_at_boundary(
        case, sigma[0, 0, :, :, 0], sigma[1, 1, :, :, 0], sigma[0, 1, :, :, 0], _THETAS
    )
    pmax, pmin = _principal_stresses(sxx, syy, sxy)
    return pmax / MAGNITUDE, pmin / MAGNITUDE, sol.converged, sol.iterations


def _principal_analytic_sweep(case, load):
    """(max, min) principal-stress/MAGNITUDE arrays over `_THETAS`, from
    :meth:`EllipticalHoleInPlateCase.periodic_inhomogeneity_stress`."""
    stress = np.asarray(case.periodic_inhomogeneity_stress(load, MAGNITUDE, n_images=N_IMAGES))
    sxx, syy, sxy = _interpolate_stress_at_boundary(
        case, stress[0, 0, :, :, 0], stress[1, 1, :, :, 0], stress[0, 1, :, :, 0], _THETAS
    )
    pmax, pmin = _principal_stresses(sxx, syy, sxy)
    return pmax / MAGNITUDE, pmin / MAGNITUDE


def _apex_sigma11(case, sigma_xx_field, sign):
    """sigma_11 just outside the opening's apex -- the tip of the semi-axis
    `b` (along x2, perpendicular to the x1 load) on the +x2 side for
    `sign=+1`, the -x2 side for `sign=-1` -- at `APEX_OFFSET` beyond the
    boundary."""
    grid = case.grid
    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    interp = RegularGridInterpolator((x1, x2), sigma_xx_field)
    point = [case.center[0], case.center[1] + sign * (case.semi_axis_b + APEX_OFFSET)]
    return float(interp([point])[0])


def _tension_apex_numeric(case, solver):
    eps_bar = case.macro_strain("tension", MAGNITUDE, solver=solver)
    sol = solver.solve(eps_bar, tol=1.0e-6, max_iterations=8000)
    sigma = np.asarray(sol.stress)
    return _apex_sigma11(case, sigma[0, 0, :, :, 0], +1.0) / MAGNITUDE, sol.converged, sol.iterations


def _tension_apex_analytic(case):
    stress = np.asarray(case.periodic_inhomogeneity_stress("tension", MAGNITUDE, n_images=N_IMAGES))
    return _apex_sigma11(case, stress[0, 0, :, :, 0], +1.0) / MAGNITUDE


def _moment_apex_numeric(case, solver):
    """(sigma_11 at +apex, sigma_11 at -apex), normalized locally by
    `GRADIENT_MAGNITUDE * case.semi_axis_b` -- see module docstring for
    why moment can't use the shared global `MAGNITUDE`."""
    eps_gradient = case.macro_strain_gradient("moment", GRADIENT_MAGNITUDE, solver=solver)
    origin = (case.center[0], case.center[1], 0.0)
    sol = solver.solve(
        np.zeros((3, 3)), tol=1.0e-6, max_iterations=8000,
        macro_strain_gradient=eps_gradient, gradient_origin=origin,
    )
    sigma = np.asarray(sol.stress)
    scale = GRADIENT_MAGNITUDE * case.semi_axis_b
    top = _apex_sigma11(case, sigma[0, 0, :, :, 0], +1.0) / scale
    bottom = _apex_sigma11(case, sigma[0, 0, :, :, 0], -1.0) / scale
    return (top, bottom), sol.converged, sol.iterations


def _moment_apex_analytic(case):
    stress = np.asarray(case.periodic_gradient_inhomogeneity_stress(GRADIENT_MAGNITUDE))
    scale = GRADIENT_MAGNITUDE * case.semi_axis_b
    top = _apex_sigma11(case, stress[0, 0, :, :, 0], +1.0) / scale
    bottom = _apex_sigma11(case, stress[0, 0, :, :, 0], -1.0) / scale
    return (top, bottom)


def _sweep(load, extract_numeric, extract_analytic, labels, title, filename, ylabel):
    """Shared driver: run the numeric solve and analytic evaluation at
    each aspect ratio, plus a smooth analytic reference curve, and save
    the comparison figure. `labels`/the extractors carry either one
    quantity (tension: max principal stress only) or two (shear/moment:
    max and min)."""
    numeric = [[] for _ in labels]
    analytic = [[] for _ in labels]
    for ratio in ASPECT_RATIOS:
        a, b = _semi_axes(ratio)
        case = _build_case(a, b)
        solver = case.solver()
        v_numeric, converged, iterations = extract_numeric(case, solver)
        v_analytic = extract_analytic(case)
        print(f"{load} a/b={ratio}: converged={converged} iterations={iterations} "
              f"numeric={tuple(round(v, 3) for v in v_numeric)} "
              f"analytic={tuple(round(v, 3) for v in v_analytic)}")
        for i, v in enumerate(v_numeric):
            numeric[i].append(v)
        for i, v in enumerate(v_analytic):
            analytic[i].append(v)

    fine_ratios = np.linspace(ASPECT_RATIOS.min(), ASPECT_RATIOS.max(), 100)
    fine = [[] for _ in labels]
    for ratio in fine_ratios:
        a, b = _semi_axes(ratio)
        case = _build_case(a, b)
        for i, v in enumerate(extract_analytic(case)):
            fine[i].append(v)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    for i, label in enumerate(labels):
        color = f"C{i}"
        analytic_numeric_curve(
            ax, fine_ratios, fine[i], numeric[i], label, color,
            downsample=1, x_numeric=ASPECT_RATIOS, markersize=6,
        )
    ax.set_xlim(ASPECT_RATIOS.min(), ASPECT_RATIOS.max())
    ax.set_xlabel(r"$a / b$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    component_legend(ax)
    save_all(fig, filename)
    plt.close(fig)


def _tension_numeric(case, solver):
    value, converged, iterations = _tension_apex_numeric(case, solver)
    return (value,), converged, iterations


def _tension_analytic(case):
    return (_tension_apex_analytic(case),)


def _shear_numeric(case, solver):
    pmax, pmin, converged, iterations = _principal_numeric_sweep(case, solver, "shear")
    return (float(np.max(pmax)), float(np.min(pmin))), converged, iterations


def _shear_analytic(case):
    pmax, pmin = _principal_analytic_sweep(case, "shear")
    return (float(np.max(pmax)), float(np.min(pmin)))


def _moment_numeric(case, solver):
    return _moment_apex_numeric(case, solver)


def _moment_analytic(case):
    return _moment_apex_analytic(case)


def plot_tension():
    _sweep(
        "tension", _tension_numeric, _tension_analytic,
        (r"$\sigma_{11}$",),
        "elliptical hole, tension: apex stress vs. aspect ratio",
        "elastic_deformation.elliptical_hole_aspect_ratio_tension",
        r"$\sigma_{11} / \sigma_\infty$",
    )


def plot_shear():
    _sweep(
        "shear", _shear_numeric, _shear_analytic,
        (r"$\sigma_1$", r"$\sigma_2$"),
        "elliptical hole, shear: extremal principal stress vs. aspect ratio",
        "elastic_deformation.elliptical_hole_aspect_ratio_shear",
        r"$\sigma_{1,2} / \sigma_\infty$",
    )


def plot_moment():
    _sweep(
        "moment", _moment_numeric, _moment_analytic,
        (r"$\sigma_{11}^{\mathrm{apex},+}$", r"$\sigma_{11}^{\mathrm{apex},-}$"),
        "elliptical hole, moment: apex stress vs. aspect ratio",
        "elastic_deformation.elliptical_hole_aspect_ratio_moment",
        r"$\sigma_{11} / (\gamma_\infty b)$",
    )


# -- Biaxial tension ("pressure" in hooke.py's naming): a spatial profile
# at one fixed aspect ratio, not part of the aspect-ratio sweep above --
# see module docstring. A traction-free hole under equal remote tension
# in both directions. --

PROFILE_ASPECT_RATIO = 0.5  # a/b: tall along x2 (b/a = 2), matching hooke.pdf; apex sigma_11 = 2*b/a = 4
PROFILE_HOLE_RADIUS = 0.1  # b, matching hole_in_plate.py's circular hole radius
PROFILE_N_IMAGES = 6


def _profile_case():
    b = PROFILE_HOLE_RADIUS
    a = PROFILE_ASPECT_RATIO * b
    return _build_case(a, b), a, b


def _biaxial_line_numeric(case, solver, b):
    """Return (xi, sigma_11, sigma_22)/MAGNITUDE along x1=center[0], x2
    varying, from the "biaxial" numerical solve."""
    eps_bar = case.macro_strain("biaxial", MAGNITUDE, solver=solver)
    sol = solver.solve(eps_bar, tol=1.0e-6, max_iterations=8000)
    sigma = np.asarray(sol.stress)

    grid = case.grid
    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    col = np.argmin(np.abs(x1 - case.center[0]))

    xi = (x2 - case.center[1]) / b
    sigma_xx = sigma[0, 0, col, :, 0] / MAGNITUDE
    sigma_yy = sigma[1, 1, col, :, 0] / MAGNITUDE
    return xi, sigma_xx, sigma_yy


def _biaxial_line_analytic(case, b):
    grid = case.grid
    stress = np.asarray(case.periodic_inhomogeneity_stress("biaxial", MAGNITUDE, n_images=PROFILE_N_IMAGES))
    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    col = np.argmin(np.abs(x1 - case.center[0]))

    xi = (x2 - case.center[1]) / b
    sigma_xx = stress[0, 0, col, :, 0] / MAGNITUDE
    sigma_yy = stress[1, 1, col, :, 0] / MAGNITUDE
    return xi, sigma_xx, sigma_yy


def plot_biaxial_profile():
    print("biaxial tension profile:")
    case, a, b = _profile_case()
    solver = case.solver()
    xi, sxx_num, syy_num = _biaxial_line_numeric(case, solver, b)
    _, sxx_an, syy_an = _biaxial_line_analytic(case, b)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.set_xlim(-4, 4)
    analytic_numeric_curve(ax, xi, sxx_an, sxx_num, r"$\sigma_{11}$", "C0")
    analytic_numeric_curve(ax, xi, syy_an, syy_num, r"$\sigma_{22}$", "C1")
    ax.set_xlabel(r"$x_2 / b$")
    ax.set_ylabel(r"$\sigma / \sigma_\infty$")
    component_legend(ax)
    save_all(fig, "elastic_deformation.elliptical_hole_pressure")
    plt.close(fig)


if __name__ == "__main__":
    plot_tension()
    plot_shear()
    plot_moment()
    plot_biaxial_profile()
