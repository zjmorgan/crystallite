"""Analytical vs. numerical stress around an elliptical region carrying a
*prescribed* eigenstrain in an otherwise homogeneous matrix -- the
elliptical counterpart of ``cylindrical_inhomogeneity.py`` (a classical
Eshelby "inclusion" problem, no stiffness contrast at all, despite the
reference material's own "elliptical inhomogeneity" naming).

Three eigenstrain components, matching the reference material's own
breakdown: an in-plane normal component (``eta0``, eps_11), an
out-of-plane/antiplane shear component (``eta4``, eps_13 -- handled
correctly by the same machinery as everything else here even though the
grid has a single point along x3, since antiplane elasticity is exactly
"u3 as an ordinary function of x1, x2", already how the diffuse-boundary
solver represents e.g. a screw dislocation's displacement field), and an
in-plane shear component (``eta5``, eps_12).

Both the analytic reference
(:meth:`crystallite.verification.EllipticalHoleInPlateCase.periodic_prescribed_eigenstrain_solution`)
and the numeric solve (via
:class:`crystallite.elastic_deformation.ElasticDeformation`'s
`eigenstrain` support) reuse the exact same machinery as
``cylindrical_inhomogeneity.py`` -- since this is a pure eigenstrain
problem (no stiffness contrast to equate), no elliptical Eshelby tensor
is needed even for the elliptical geometry, just the ellipse's own
Fourier transform
(:func:`crystallite.verification.elastic_deformation._ellipse_fourier_transform`,
a straightforward affine rescaling of the disk's) in place of the disk's.
"""

from __future__ import annotations

import numpy as np

from _plotting import analytic_numeric_curve, plt, save_all
from crystallite.grid import Grid
from crystallite.verification import EllipticalHoleInPlateCase

MATRIX_LAME_LAMBDA = 1.0
MATRIX_LAME_MU = 0.7
SEMI_AXIS_A = 0.15
SEMI_AXIS_B = 0.08
GRID_SHAPE = (256, 256, 1)
EIGENSTRAIN_MAGNITUDE = 0.01


def _eigenstrain_tensor(label):
    e = np.zeros((3, 3))
    if label == "eta0":
        e[0, 0] = EIGENSTRAIN_MAGNITUDE
    elif label == "eta4":
        e[0, 2] = e[2, 0] = EIGENSTRAIN_MAGNITUDE
    elif label == "eta5":
        e[0, 1] = e[1, 0] = EIGENSTRAIN_MAGNITUDE
    else:
        raise ValueError(f"unknown label {label!r}")
    return e


def _build_case():
    grid = Grid(shape=GRID_SHAPE, lengths=(1.0, 1.0, 1.0))
    return EllipticalHoleInPlateCase(
        grid, matrix_lame_lambda=MATRIX_LAME_LAMBDA, matrix_lame_mu=MATRIX_LAME_MU,
        semi_axis_a=SEMI_AXIS_A, semi_axis_b=SEMI_AXIS_B, contrast=1.0, center=(0.5, 0.5),
    )


# Which Cartesian stress components are physically nonzero along the
# x1=center probe line for each eigenstrain -- eta4 (antiplane eps_13)
# and eta5 (in-plane eps_12) both have a node there (same reasoning as
# cylindrical_inhomogeneity.py's eta5), so only one component each is
# meaningful to plot.
_COMPONENTS = {
    "eta0": ((0, 0), (1, 1)),
    "eta4": ((0, 2), None),
    "eta5": ((0, 1), None),
}
_LABELS = {
    (0, 0): r"$\sigma_{11}$", (1, 1): r"$\sigma_{22}$",
    (0, 1): r"$\sigma_{12}$", (0, 2): r"$\sigma_{13}$",
}


def _line_through_hole(case, solver, eigenstrain_tensor, components):
    grid = case.grid
    eigenstrain_field = case.eigenstrain_field(eigenstrain_tensor)
    sol = solver.solve(
        np.zeros((3, 3)), eigenstrain=eigenstrain_field, tol=1.0e-6, max_iterations=5000
    )
    print(f"  numerical: converged={sol.converged}, iterations={sol.iterations}, "
          f"residual={sol.residual_norm:.3g}")
    sigma = np.asarray(sol.stress)

    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    col = np.argmin(np.abs(x1 - case.center[0]))

    scale = MATRIX_LAME_MU * EIGENSTRAIN_MAGNITUDE
    xi = (x2 - case.center[1]) / SEMI_AXIS_B
    comp_a, comp_b = components
    value_a = sigma[comp_a[0], comp_a[1], col, :, 0] / scale
    value_b = sigma[comp_b[0], comp_b[1], col, :, 0] / scale if comp_b is not None else None
    return xi, value_a, value_b


def _periodic_analytic_line(case, eigenstrain_tensor, components):
    grid = case.grid
    _, stress = case.periodic_prescribed_eigenstrain_solution(eigenstrain_tensor)
    stress = np.asarray(stress)

    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    col = np.argmin(np.abs(x1 - case.center[0]))

    scale = MATRIX_LAME_MU * EIGENSTRAIN_MAGNITUDE
    xi = (x2 - case.center[1]) / SEMI_AXIS_B
    comp_a, comp_b = components
    value_a = stress[comp_a[0], comp_a[1], col, :, 0] / scale
    value_b = stress[comp_b[0], comp_b[1], col, :, 0] / scale if comp_b is not None else None
    return xi, value_a, value_b


def plot_eigenstrain(label):
    eigenstrain_tensor = _eigenstrain_tensor(label)
    components = _COMPONENTS[label]
    print(f"{label}:")
    case = _build_case()
    solver = case.solver()
    xi, sa, sb = _line_through_hole(case, solver, eigenstrain_tensor, components)
    _, pa, pb = _periodic_analytic_line(case, eigenstrain_tensor, components)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.set_xlim(-6, 6)
    analytic_numeric_curve(ax, xi, pa, sa, _LABELS[components[0]], "C0")
    if components[1] is not None:
        analytic_numeric_curve(ax, xi, pb, sb, _LABELS[components[1]], "C1")
    ax.set_xlabel(r"$x_2 / b$")
    ax.set_ylabel(r"$\sigma / (\mu \varepsilon^0)$")
    ax.legend(fontsize=7, ncol=2)
    save_all(fig, f"elastic_deformation.elliptical_eigenstrain_{label}")
    plt.close(fig)


if __name__ == "__main__":
    plot_eigenstrain("eta0")
    plot_eigenstrain("eta4")
    plot_eigenstrain("eta5")
