r"""Interior heat flux of an *elliptical* inhomogeneity, whose interior field is
exactly uniform (Eshelby), against its shape and against the anisotropy of the
matrix -- as a solid analytic line and numerical markers (Fourier's law,
:math:`q_i=\kappa_{ij}X_j`).

* **Aspect ratio** (``aspect_ratio_soft``/``_hard``): an ellipse of semi-axes
  :math:`a,b` with :math:`\sqrt{ab}` held fixed, in an isotropic matrix, under
  a thermal driving force at 45 degrees to the axes so both heat flux components are nonzero,
  against :math:`a/b`.
* **Matrix rotation** (``matrix_rotation_soft``/``_hard``): a circle in a
  matrix with :math:`\kappa_\mathrm{max}/\kappa_\mathrm{min}=4` (principal
  conductivities :math:`2` and :math:`1/2`, geometric mean :math:`1`), the matrix
  rotated about :math:`x_3` by :math:`\varphi` against a fixed thermal driving force along
  :math:`x_1`. The interior heat flux has period 180 degrees (a rank-2 tensor) and
  gains a transverse component wherever the principal axes are off the field.

The heat flux is the interior :math:`\kappa_1X_\mathrm{in}`, normalized by
:math:`\kappa_\mathrm{ref}|\bar X|` with :math:`\kappa_\mathrm{ref}` the matrix
thermal conductivity (isotropic) or its geometric-mean principal value (=1).

The analytic curve is
:meth:`crystallite.verification.heat_conduction.HeatInclusionCase.periodic_interior_heat_flux`
-- the exact interior of the periodic array from the anisotropic lattice-sum
depolarization tensor (no FFT, no grid), which reduces to the isolated result
as the inclusion shrinks. Numerics average the solver's heat flux over the middle of
the inclusion.

Numeric results are cached in ``figures/_numeric_cache``.
"""

from __future__ import annotations

import numpy as np

from _plotting import analytic_numeric_curve, cached_numeric, component_legend, plt, save_all
from crystallite.grid import Grid
from crystallite.verification.heat_conduction import HeatInclusionCase

GRID_SHAPE = (512, 512, 1)
CHARACTERISTIC_SIZE = 0.08  # sqrt(a*b)
CIRCLE_RADIUS = 0.1
INTERIOR_RADIUS = 0.5  # elliptical radius bounding the region averaged for the numeric heat flux
ASPECT_RATIOS = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
N_FINE = 50
ROTATION_STEP_DEGREES = 15.0
ROTATION_FINE_DEGREES = 3.0
PRINCIPAL_CONDUCTIVITIES = (2.0, 0.5)  # geometric mean 1
CONTRASTS = {"soft": 0.25, "hard": 4.0}
DIAGONAL_FIELD = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
AXIAL_FIELD = np.array([1.0, 0.0, 0.0])


def _rotated_matrix(angle):
    k1, k2 = PRINCIPAL_CONDUCTIVITIES
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return r @ np.diag([k1, k2, 1.0]) @ r.T


def _shape_case(ratio, contrast):
    a, b = CHARACTERISTIC_SIZE * np.sqrt(ratio), CHARACTERISTIC_SIZE / np.sqrt(ratio)
    return HeatInclusionCase(Grid(shape=GRID_SHAPE), 1.0, a, b, contrast=contrast)


def _rotation_case(angle, contrast):
    return HeatInclusionCase(
        Grid(shape=GRID_SHAPE), _rotated_matrix(angle), CIRCLE_RADIUS, CIRCLE_RADIUS,
        contrast=contrast,
    )


def _numeric_flux(case, field):
    solution = case.solver().solve(field, tol=1.0e-6, max_iterations=6000)
    inside = np.asarray(case.elliptical_radius) < INTERIOR_RADIUS
    flux = np.asarray(solution.flux)
    return [float(flux[i][inside].mean()) for i in range(2)], solution.converged, solution.iterations


def _analytic_flux(case, field):
    return list(case.periodic_interior_heat_flux(field)[:2])


def plot_aspect_ratio(label):
    contrast = CONTRASTS[label]

    def compute():
        rows = []
        for ratio in ASPECT_RATIOS:
            values, converged, iterations = _numeric_flux(_shape_case(ratio, contrast), DIAGONAL_FIELD)
            print(f"{label} a/b={ratio}: converged={converged} iterations={iterations} "
                  f"numeric={tuple(round(v, 4) for v in values)} "
                  f"analytic={tuple(round(v, 4) for v in _analytic_flux(_shape_case(ratio, contrast), DIAGONAL_FIELD))}")
            rows.append(values)
        return np.array(rows)

    numeric = cached_numeric(
        f"conduction_shape_{label}_N{GRID_SHAPE[0]}_s{CHARACTERISTIC_SIZE}_ir{INTERIOR_RADIUS}"
        f"_r{'-'.join(str(r) for r in ASPECT_RATIOS)}", compute,
    )
    fine_ratios = np.linspace(ASPECT_RATIOS.min(), ASPECT_RATIOS.max(), N_FINE)
    fine = np.array([_analytic_flux(_shape_case(r, contrast), DIAGONAL_FIELD) for r in fine_ratios])

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    for i, name in enumerate((r"$q_1$", r"$q_2$")):
        analytic_numeric_curve(
            ax, fine_ratios, fine[:, i], numeric[:, i], name, f"C{i}", downsample=1,
            x_numeric=ASPECT_RATIOS, markersize=6,
        )
    ax.set_xlim(ASPECT_RATIOS.min(), ASPECT_RATIOS.max())
    ax.set_xlabel(r"$a / b$")
    ax.set_ylabel(r"$q_\mathrm{in} / (\kappa_0 |\bar{X}|)$")
    ax.set_title(rf"elliptical inclusion, $\kappa_1/\kappa_0={contrast:g}$: interior heat flux vs. aspect ratio")
    component_legend(ax)
    save_all(fig, f"heat_conduction.elliptical_inhomogeneity_aspect_ratio_{label}")
    plt.close(fig)


def plot_matrix_rotation(label):
    contrast = CONTRASTS[label]
    angles = np.arange(0.0, 180.0 + 0.5 * ROTATION_STEP_DEGREES, ROTATION_STEP_DEGREES)

    def compute():
        rows = []
        for angle in angles:
            values, converged, iterations = _numeric_flux(
                _rotation_case(np.deg2rad(angle), contrast), AXIAL_FIELD
            )
            print(f"{label} phi={angle}: converged={converged} iterations={iterations}")
            rows.append(values)
        return np.array(rows)

    numeric = cached_numeric(
        f"conduction_rotation_{label}_N{GRID_SHAPE[0]}_R{CIRCLE_RADIUS}_ir{INTERIOR_RADIUS}"
        f"_k{PRINCIPAL_CONDUCTIVITIES}_step{ROTATION_STEP_DEGREES}", compute,
    )
    fine_angles = np.arange(0.0, 180.0 + 0.5 * ROTATION_FINE_DEGREES, ROTATION_FINE_DEGREES)
    fine = np.array([
        _analytic_flux(_rotation_case(np.deg2rad(a), contrast), AXIAL_FIELD) for a in fine_angles
    ])

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    for i, name in enumerate((r"$q_1$", r"$q_2$")):
        analytic_numeric_curve(
            ax, fine_angles, fine[:, i], numeric[:, i], name, f"C{i}", downsample=1,
            x_numeric=angles, markersize=4,
        )
    ax.set_xlim(0.0, 180.0)
    ax.set_xticks(np.arange(0, 181, 45))
    ax.set_xlabel(r"$\varphi\ (^\circ)$")
    ax.set_ylabel(r"$q_\mathrm{in} / (\kappa_\mathrm{ref} |\bar{X}|)$")
    ax.set_title(rf"circle, $\kappa_1/\kappa_0={contrast:g}$, $\kappa_\mathrm{{max}}/\kappa_\mathrm{{min}}=4$: interior heat flux vs. matrix rotation")
    component_legend(ax)
    save_all(fig, f"heat_conduction.elliptical_inhomogeneity_matrix_rotation_{label}")
    plt.close(fig)


if __name__ == "__main__":
    for name in CONTRASTS:
        plot_aspect_ratio(name)
    for name in CONTRASTS:
        plot_matrix_rotation(name)
