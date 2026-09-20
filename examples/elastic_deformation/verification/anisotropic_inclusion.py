"""Interior stress of a circular inclusion carrying a uniform eigenstrain in an
*anisotropic* matrix -- the anisotropy counterpart of ``elliptical_inclusion.py``
and the last of the classical figures from this project's original reference
material (``hooke.pdf``).

Two families of figures, each sampling only the (uniform) stress in the centre of
the circle, as a solid analytic line and numerical markers:

* **Anisotropy sweeps** at a fixed orientation, for the aspect ratios
  ``RATIOS = 1/8 ... 8``: ``eta0`` (:math:`\\varepsilon_{11}`, giving
  :math:`\\sigma_{11}` and :math:`\\sigma_{22}`) and ``eta5``
  (:math:`\\varepsilon_{12}`, giving :math:`\\sigma_{12}`) in a cubic matrix
  against its Zener ratio :math:`A_{\\mathrm Z}=2c_{44}/(c_{11}-c_{12})`, and
  ``eta4`` (:math:`\\varepsilon_{13}`, antiplane, giving :math:`\\sigma_{13}`) in
  a matrix whose two antiplane shear moduli differ, against
  :math:`c_{44}/c_{55}`.
* **Rotation** at a fixed anisotropy: the matrix is rotated about :math:`x_3`
  from 0 to 360 degrees while the eigenstrain stays fixed, which shows the
  symmetry of the response -- a period of 90 degrees for the cubic matrix
  (``eta0``, ``eta5``) and 180 degrees for the antiplane one (``eta4``), whose
  interior gains a :math:`\\sigma_{23}` (and, for the cubic matrix,
  :math:`\\sigma_{12}`, :math:`\\sigma_{11}`, :math:`\\sigma_{22}`) component
  away from the crystal axes. ``eta0`` and ``eta5`` plot both
  ``ROTATION_RATIOS`` (:math:`A_{\\mathrm Z}=1/8` and ``8``), since a cubic
  crystal's Zener ratio is a rotational invariant -- no rotation turns one into
  the other. ``eta4`` plots only :math:`c_{44}/c_{55}=8`: inverting that ratio
  swaps :math:`c_{44}\\leftrightarrow c_{55}`, which is exactly what a 90-degree
  turn does to this tensor, so :math:`r` and :math:`1/r` are the same matrix
  material and would give the same curve, 90 degrees apart.

The moduli are held so that only the anisotropy changes: the cubic matrix keeps
its bulk modulus :math:`K=\\lambda+2\\mu/3` and :math:`c'=(c_{11}-c_{12})/2=\\mu`
fixed and varies :math:`c_{44}`, so :math:`A_{\\mathrm Z}=1` is exactly the
isotropic medium (:math:`\\lambda=1`, :math:`\\mu=0.7`) and stresses are
normalized by :math:`c'\\varepsilon^0`; the antiplane matrix is isotropic except
for :math:`c_{44}=\\mu\\sqrt r` and :math:`c_{55}=\\mu/\\sqrt r` (:math:`r=c_{44}/
c_{55}`), keeping :math:`\\sqrt{c_{44}c_{55}}=\\mu`, and is normalized by
:math:`\\mu\\varepsilon^0`.

The analytic curve is
:meth:`crystallite.verification.anisotropic_inclusion.AnisotropicInclusionCase.periodic_interior_stress`
-- the exact interior stress of the periodic array from the anisotropic
lattice-sum Eshelby tensor (no FFT, no grid), which reduces to Eshelby's isolated
result as the circle shrinks. The numeric solve is
:class:`crystallite.elastic_deformation.ElasticDeformation` with the matrix's
full stiffness tensor, the same eigenstrain (Lanczos-smoothed, area preserving)
in a homogeneous matrix; each solve converges in one iteration.

Numeric results are cached in ``figures/_numeric_cache`` (see
``_plotting.cached_numeric``) so restyling a figure does not repeat the solves.
"""

from __future__ import annotations

import numpy as np

from _plotting import (
    analytic_numeric_curve, cached_numeric, component_legend, plt, save_all,
)
from crystallite.grid import Grid
from crystallite.verification.anisotropic_inclusion import (
    AnisotropicInclusionCase,
    antiplane_orthotropic_stiffness,
    cubic_from_zener,
    rotate_stiffness,
)

LAME_LAMBDA = 1.0
LAME_MU = 0.7  # = c' of the cubic matrix and sqrt(c44*c55) of the antiplane one
RADIUS = 0.1
GRID_SHAPE = (512, 512, 1)
EIGENSTRAIN_MAGNITUDE = 0.01
INTERIOR_RADIUS = 0.5  # elliptical radius bounding the region averaged for the numeric stress
RATIOS = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
# eta0/eta5: both, since a cubic crystal's Zener ratio is a rotational invariant
# (no rotation relates A_Z to 1/A_Z). eta4: only one -- inverting c44/c55 is
# exactly what a 90-degree turn does to that tensor, so r and 1/r give the same
# curve, 90 degrees apart.
ROTATION_RATIOS = {"eta0": (0.125, 8.0), "eta5": (0.125, 8.0), "eta4": (8.0,)}
ROTATION_STEP_DEGREES = 15.0  # numerical markers
ROTATION_FINE_DEGREES = 3.0  # analytic line
N_FINE_SWEEP = 80

# label -> (eigenstrain index pair, matrix builder, sweep components, rotation components)
_CASES = {
    "eta0": ((0, 0), "cubic", [(0, 0), (1, 1)], [(0, 0), (1, 1), (0, 1)]),
    "eta5": ((0, 1), "cubic", [(0, 1)], [(0, 0), (1, 1), (0, 1)]),
    "eta4": ((0, 2), "antiplane", [(0, 2)], [(0, 2), (1, 2)]),
}
_STRESS_LABELS = {
    (0, 0): r"$\sigma_{11}$", (1, 1): r"$\sigma_{22}$", (0, 1): r"$\sigma_{12}$",
    (0, 2): r"$\sigma_{13}$", (1, 2): r"$\sigma_{23}$",
}
_EIGENSTRAIN_NAMES = {"eta0": r"\varepsilon_{11}", "eta5": r"\varepsilon_{12}", "eta4": r"\varepsilon_{13}"}
_RATIO_LABELS = {
    "cubic": r"$A_{\mathrm{Z}} = 2 c_{44} / (c_{11} - c_{12})$",
    "antiplane": r"$c_{44} / c_{55}$",
}
_YLABELS = {"cubic": r"$\sigma / (c' \varepsilon^0)$", "antiplane": r"$\sigma / (\mu \varepsilon^0)$"}


def _stiffness(kind, ratio):
    if kind == "cubic":
        return cubic_from_zener(ratio, LAME_LAMBDA, LAME_MU)
    return antiplane_orthotropic_stiffness(ratio, LAME_LAMBDA, LAME_MU)


def _eigenstrain_tensor(indices):
    e = np.zeros((3, 3))
    e[indices[0], indices[1]] = e[indices[1], indices[0]] = EIGENSTRAIN_MAGNITUDE
    return e


def _case(stiffness, grid):
    return AnisotropicInclusionCase(grid, stiffness, RADIUS, RADIUS)


def _numeric_values(stiffness, eigenstrain_tensor, components):
    """Mean stress over the centre of the circle, per component, / (mu*eps0)."""
    grid = Grid(shape=GRID_SHAPE, lengths=(1.0, 1.0, 1.0))
    case = _case(stiffness, grid)
    solution = case.solver().solve(
        np.zeros((3, 3)), eigenstrain=case.eigenstrain_field(eigenstrain_tensor),
        tol=1.0e-6, max_iterations=5000,
    )
    stress = np.asarray(solution.stress)
    inside = np.asarray(case.elliptical_radius) < INTERIOR_RADIUS
    scale = LAME_MU * EIGENSTRAIN_MAGNITUDE
    values = [float(np.mean(stress[i, j][inside])) / scale for (i, j) in components]
    return values, bool(solution.converged), int(solution.iterations)


def _analytic_values(stiffness, eigenstrain_tensor, components):
    grid = Grid(shape=(16, 16, 1), lengths=(1.0, 1.0, 1.0))  # only the cell size matters
    stress = _case(stiffness, grid).periodic_interior_stress(eigenstrain_tensor)
    scale = LAME_MU * EIGENSTRAIN_MAGNITUDE
    return [float(stress[i, j]) / scale for (i, j) in components]


def _key(kind, label, what, extra=""):
    return (f"anisotropic_{kind}_{label}_{what}_R{RADIUS}_N{GRID_SHAPE[0]}_lam{LAME_LAMBDA}_mu{LAME_MU}"
            f"_e{EIGENSTRAIN_MAGNITUDE}_ir{INTERIOR_RADIUS}{extra}")


def plot_sweep(label):
    indices, kind, components, _ = _CASES[label]
    eigenstrain_tensor = _eigenstrain_tensor(indices)

    def compute():
        rows = []
        for ratio in RATIOS:
            values, converged, iterations = _numeric_values(
                _stiffness(kind, ratio), eigenstrain_tensor, components
            )
            rows.append(values)
            print(f"{label} ratio={ratio}: converged={converged} iterations={iterations} "
                  f"numeric={tuple(round(v, 4) for v in values)} "
                  f"analytic={tuple(round(v, 4) for v in _analytic_values(_stiffness(kind, ratio), eigenstrain_tensor, components))}")
        return np.array(rows)

    numeric = cached_numeric(_key(kind, label, "sweep", f"_r{'-'.join(str(r) for r in RATIOS)}"), compute)
    fine_ratios = np.linspace(RATIOS.min(), RATIOS.max(), N_FINE_SWEEP)
    fine = np.array([
        _analytic_values(_stiffness(kind, r), eigenstrain_tensor, components) for r in fine_ratios
    ])

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    for i, component in enumerate(components):
        analytic_numeric_curve(
            ax, fine_ratios, fine[:, i], numeric[:, i], _STRESS_LABELS[component], f"C{i}",
            downsample=1, x_numeric=RATIOS, markersize=6,
        )
    ax.set_xlim(RATIOS.min(), RATIOS.max())
    ax.set_xlabel(_RATIO_LABELS[kind])
    ax.set_ylabel(_YLABELS[kind])
    ax.set_title(rf"anisotropic matrix, ${_EIGENSTRAIN_NAMES[label]}$: interior stress vs. anisotropy")
    component_legend(ax)
    save_all(fig, f"elastic_deformation.anisotropic_inclusion_sweep_{label}")
    plt.close(fig)


def plot_rotation(label, ratio):
    indices, kind, _, components = _CASES[label]
    eigenstrain_tensor = _eigenstrain_tensor(indices)
    base = _stiffness(kind, ratio)
    angles = np.arange(0.0, 360.0 + 0.5 * ROTATION_STEP_DEGREES, ROTATION_STEP_DEGREES)

    def compute():
        rows = []
        for angle in angles:
            values, converged, iterations = _numeric_values(
                rotate_stiffness(base, np.deg2rad(angle)), eigenstrain_tensor, components
            )
            rows.append(values)
        print(f"{label} ratio={ratio}: {len(angles)} rotation solves done "
              f"(last: converged={converged}, iterations={iterations})")
        return np.array(rows)

    numeric = cached_numeric(_key(kind, label, "rotation", f"_ratio{ratio}_step{ROTATION_STEP_DEGREES}"), compute)
    fine_angles = np.arange(0.0, 360.0 + 0.5 * ROTATION_FINE_DEGREES, ROTATION_FINE_DEGREES)
    fine = np.array([
        _analytic_values(rotate_stiffness(base, np.deg2rad(a)), eigenstrain_tensor, components)
        for a in fine_angles
    ])

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    for i, component in enumerate(components):
        analytic_numeric_curve(
            ax, fine_angles, fine[:, i], numeric[:, i], _STRESS_LABELS[component], f"C{i}",
            downsample=1, x_numeric=angles, markersize=4,
        )
    ax.set_xlim(0.0, 360.0)
    ax.set_xticks(np.arange(0, 361, 90))
    ax.set_xlabel(r"$\varphi\ (^\circ)$")
    ax.set_ylabel(_YLABELS[kind])
    ratio_name = "A_{\\mathrm{Z}}" if kind == "cubic" else "c_{44}/c_{55}"
    ax.set_title(rf"${_EIGENSTRAIN_NAMES[label]}$, ${ratio_name}={ratio:g}$: interior stress vs. matrix rotation")
    component_legend(ax)
    tag = f"{ratio:g}".replace(".", "p")
    save_all(fig, f"elastic_deformation.anisotropic_inclusion_rotation_{label}_ratio{tag}")
    plt.close(fig)


if __name__ == "__main__":
    for name in _CASES:
        plot_sweep(name)
    for name in _CASES:
        for ratio in ROTATION_RATIOS[name]:
            plot_rotation(name, ratio)
