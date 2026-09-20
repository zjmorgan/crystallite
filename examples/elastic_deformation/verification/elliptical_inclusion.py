"""Interior stress of an elliptical inclusion versus its aspect ratio, for a
uniform eigenstrain :math:`\\varepsilon_{11}`, :math:`\\varepsilon_{12}` or
:math:`\\varepsilon_{13}` -- the classical Eshelby result (the stress inside an
ellipse carrying a uniform eigenstrain is itself uniform, and depends only on
the shape and Poisson's ratio), the eigenstrain counterpart of
``elliptical_hole.py``'s peak-stress sweeps and the same figure this project's
original reference material (``hooke.pdf``) showed.

The aspect ratios and size convention are those of ``elliptical_hole.py``
(``ASPECT_RATIOS``, ``a/b`` with ``sqrt(a*b)`` held fixed), imported from it so
the two sweeps cannot drift apart (the grid is finer, see ``GRID_SHAPE``). Each figure plots the nonzero interior
components against ``a/b`` as a solid analytic line and numerical markers:
``eta0`` (:math:`\\varepsilon_{11}`) gives :math:`\\sigma_{11}` and
:math:`\\sigma_{22}`, ``eta5`` (:math:`\\varepsilon_{12}`) gives
:math:`\\sigma_{12}`, and ``eta4`` (:math:`\\varepsilon_{13}`, antiplane) gives
:math:`\\sigma_{13}`, all normalized by :math:`\\mu\\varepsilon^0`.

The analytic curve is
:meth:`crystallite.verification.EllipticalHoleInPlateCase.periodic_prescribed_eigenstrain_interior_stress`
-- the interior stress of the *periodic* array, from the exact lattice-sum
Eshelby tensor (no FFT, no grid), so it includes the small periodic-cell
correction (the area fraction here is ~0.4%) and reduces to the classical
isolated result as the ellipse shrinks. The numeric solve applies the same
eigenstrain through :class:`crystallite.elastic_deformation.ElasticDeformation`
in a homogeneous matrix (``contrast=1``); its interior stress is the mean over
the central part of the ellipse (the field is uniform there). The eigenstrain
region is smoothed with the same Lanczos filter ``elliptical_hole.py`` uses for
its boundary: a sharp pixelated mask at the slender ratios is only ~6 cells
across and its area, and hence the interior stress, would be off by the
mask's area error.
"""

from __future__ import annotations

import numpy as np

from _plotting import analytic_numeric_curve, component_legend, plt, save_all
from crystallite.grid import Grid
from crystallite.verification import EllipticalHoleInPlateCase
from elliptical_hole import ASPECT_RATIOS, CHARACTERISTIC_SIZE, _semi_axes

# Finer than the hole sweep's 512^2: each solve here is a single iteration (a
# homogeneous matrix makes the reference operator exact), so resolution is
# nearly free, and it matters -- at the slender ratios the ellipse is only ~6
# cells across at 512^2 and the numeric interior stress is off by up to 4%
# there, against <= 1.3% at 1024^2 (20% -> 2% -> 0.4% at a/b=1/8 for 256^2,
# 512^2, 1024^2).
GRID_SHAPE = (1024, 1024, 1)
MATRIX_LAME_LAMBDA = 1.0
MATRIX_LAME_MU = 0.7
EIGENSTRAIN_MAGNITUDE = 0.01
INTERIOR_RADIUS = 0.5  # elliptical radius bounding the region averaged for the numeric interior stress
N_FINE = 60  # points on the smooth analytic curve

# label -> (eigenstrain indices, [(stress component, legend label)])
_CASES = {
    "eta0": ((0, 0), [((0, 0), r"$\sigma_{11}$"), ((1, 1), r"$\sigma_{22}$")]),
    "eta5": ((0, 1), [((0, 1), r"$\sigma_{12}$")]),
    "eta4": ((0, 2), [((0, 2), r"$\sigma_{13}$")]),
}
_YLABEL = r"$\sigma / (\mu \varepsilon^0)$"
_LEGEND_LOC = {"eta0": "center right"}  # keep clear of the curves' left end
_TITLES = {
    "eta0": r"elliptical inclusion, $\varepsilon_{11}$: interior stress vs. aspect ratio",
    "eta5": r"elliptical inclusion, $\varepsilon_{12}$: interior stress vs. aspect ratio",
    "eta4": r"elliptical inclusion, $\varepsilon_{13}$: interior stress vs. aspect ratio",
}


def _build_case(a, b):
    grid = Grid(shape=GRID_SHAPE, lengths=(1.0, 1.0, 1.0))
    return EllipticalHoleInPlateCase(
        grid, matrix_lame_lambda=MATRIX_LAME_LAMBDA, matrix_lame_mu=MATRIX_LAME_MU,
        semi_axis_a=a, semi_axis_b=b, contrast=1.0, center=(0.5, 0.5),
    )


def _eigenstrain_tensor(indices):
    e = np.zeros((3, 3))
    e[indices[0], indices[1]] = e[indices[1], indices[0]] = EIGENSTRAIN_MAGNITUDE
    return e


def _smoothed_eigenstrain_field(case, eigenstrain_tensor):
    """The eigenstrain field with its indicator Lanczos-smoothed (area
    preserving), as the hole sweep smooths its material boundary."""
    grid = case.grid
    field = np.asarray(case.eigenstrain_field(eigenstrain_tensor))
    smoothed = grid.ifft(grid.fft(field) * grid.lanczos_filter)
    return np.real(np.asarray(smoothed))


def _numeric_interior(case, solver, eigenstrain_tensor, components):
    field = _smoothed_eigenstrain_field(case, eigenstrain_tensor)
    sol = solver.solve(
        np.zeros((3, 3)), eigenstrain=field, tol=1.0e-6, max_iterations=5000
    )
    stress = np.asarray(sol.stress)
    inside = np.asarray(case.elliptical_radius) < INTERIOR_RADIUS
    scale = MATRIX_LAME_MU * EIGENSTRAIN_MAGNITUDE
    values = [float(np.mean(stress[i, j][inside])) / scale for (i, j) in components]
    return values, sol.converged, sol.iterations


def _analytic_interior(case, eigenstrain_tensor, components):
    stress = np.asarray(case.periodic_prescribed_eigenstrain_interior_stress(eigenstrain_tensor))
    scale = MATRIX_LAME_MU * EIGENSTRAIN_MAGNITUDE
    return [float(stress[i, j]) / scale for (i, j) in components]


def plot_inclusion_sweep(label):
    indices, stress_components = _CASES[label]
    components = [c for c, _ in stress_components]
    eigenstrain_tensor = _eigenstrain_tensor(indices)

    numeric = [[] for _ in components]
    for ratio in ASPECT_RATIOS:
        a, b = _semi_axes(ratio)
        case = _build_case(a, b)
        values, converged, iterations = _numeric_interior(
            case, case.solver(), eigenstrain_tensor, components
        )
        exact = _analytic_interior(case, eigenstrain_tensor, components)
        print(f"{label} a/b={ratio}: converged={converged} iterations={iterations} "
              f"numeric={tuple(round(v, 4) for v in values)} "
              f"analytic={tuple(round(v, 4) for v in exact)}")
        for i, v in enumerate(values):
            numeric[i].append(v)

    fine_ratios = np.linspace(ASPECT_RATIOS.min(), ASPECT_RATIOS.max(), N_FINE)
    fine = [[] for _ in components]
    for ratio in fine_ratios:
        a, b = _semi_axes(ratio)
        for i, v in enumerate(_analytic_interior(_build_case(a, b), eigenstrain_tensor, components)):
            fine[i].append(v)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    for i, (_, legend_label) in enumerate(stress_components):
        analytic_numeric_curve(
            ax, fine_ratios, fine[i], numeric[i], legend_label, f"C{i}",
            downsample=1, x_numeric=ASPECT_RATIOS, markersize=6,
        )
    ax.set_xlim(ASPECT_RATIOS.min(), ASPECT_RATIOS.max())
    ax.set_xlabel(r"$a / b$")
    ax.set_ylabel(_YLABEL)
    ax.set_title(_TITLES[label])
    component_legend(ax, loc=_LEGEND_LOC.get(label, "best"))
    save_all(fig, f"elastic_deformation.elliptical_inclusion_aspect_ratio_{label}")
    plt.close(fig)


if __name__ == "__main__":
    for name in _CASES:
        plot_inclusion_sweep(name)
