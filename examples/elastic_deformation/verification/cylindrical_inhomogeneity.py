"""Analytical vs. numerical stress around a circular region carrying a
*prescribed* eigenstrain (a stress-free transformation strain, e.g. a
precipitate with a lattice mismatch) in an otherwise homogeneous matrix --
the classical Eshelby "inclusion" problem. Despite the reference
material's own "cylindrical inhomogeneity" naming for this case, there is
no stiffness contrast at all here (that's the *other* already-completed
group, ``cylindrical_inclusion.py``'s soft/hard/rigid) -- the region is
literally the same material as the matrix, just carrying an eigenstrain.

Five eigenstrain components, matching the reference material's own
breakdown: a single in-plane normal component (``eta0``, eps_11), the
out-of-plane normal component (``eta2``, eps_33 -- nonzero even on this
project's pseudo-2D grid, since it is a prescribed *material* strain, not
a kinematic one, and plane strain only forces the *kinematic* eps_33
fluctuation to vanish), an out-of-plane/antiplane shear component
(``eta4``, eps_13 -- ported directly from
``elliptical_inhomogeneity.py``'s already-working ``eta4`` case, since it
exercises exactly the same generic eigenstrain machinery this module
already relies on for ``eta0``/``eta2``/``eta5``; the reference
material's own "Cubic matrix"/"Transversely isotropic matrix" sub-variants
of this case -- ``zener``, ``zener.angle``, ``principal``/``secondary`` --
are deliberately *not* ported here, since they require genuinely
anisotropic elasticity, which this project's isotropic-only
:meth:`crystallite.elastic_deformation.ElasticDeformation.stress` does not
support at all -- a much larger undertaking than filling in this module's
missing isotropic component), an in-plane shear component (``eta5``,
eps_12), that same shear expressed as a pure deviatoric pair instead
(``eta5_alt``, eps_11=-eps_22=eta0>0 -- the reference material's own
``eigenstrain.5.alt``: physically the *same* shear misfit as ``eta5``,
just described in a frame rotated 45 degrees, where a pure eps_12
becomes a pure eps_11=-eps_22 -- see
:func:`crystallite.verification.elastic_deformation._ellipse_inhomogeneity_exterior_stress`'s
own docstring, whose shear-coupling fix was independently checked
against exactly this rotated-diagonal equivalence), and full hydrostatic
dilatation (``dilation``, eps_11=eps_22=eps_33 all equal).

The analytic reference is
:meth:`crystallite.verification.HoleInPlateCase.periodic_prescribed_eigenstrain_solution`,
an exact periodic Khachaturyan-Shatalov Fourier solve for the prescribed
eigenstrain (not an approximation, and not image-summed -- this is a
*homogeneous*-matrix problem, so the same Eshelby-eigenstrain-in-Fourier-
space machinery used for the circular hole/inclusion cases applies
directly and exactly, with no "equivalent inclusion" step needed since
there is no actual stiffness contrast to equate). The numeric side uses
:class:`crystallite.elastic_deformation.ElasticDeformation`'s
`eigenstrain` support (new: a prescribed stress-free transformation
strain entering Hooke's law as ``sigma(x) = C(x):(eps(x) -
eigenstrain(x))``, added specifically to cover this case) with
`contrast=1` (homogeneous), which converges quickly (the reference-medium
preconditioner is then exact) and matches the analytic curve closely even
close to and inside the disk -- unlike the finite-contrast cases, there
is no stiffness-jump ill-conditioning here.
"""

from __future__ import annotations

import numpy as np

from _plotting import analytic_numeric_curve, plt, save_all
from crystallite.grid import Grid
from crystallite.verification import HoleInPlateCase

MATRIX_LAME_LAMBDA = 1.0
MATRIX_LAME_MU = 0.7
HOLE_RADIUS = 0.1
GRID_SHAPE = (256, 256, 1)
EIGENSTRAIN_MAGNITUDE = 0.01

# label -> eigenstrain tensor (3, 3)
def _eigenstrain_tensor(label):
    e = np.zeros((3, 3))
    if label == "eta0":
        e[0, 0] = EIGENSTRAIN_MAGNITUDE
    elif label == "eta2":
        e[2, 2] = EIGENSTRAIN_MAGNITUDE
    elif label == "eta4":
        e[0, 2] = e[2, 0] = EIGENSTRAIN_MAGNITUDE
    elif label == "eta5":
        e[0, 1] = e[1, 0] = EIGENSTRAIN_MAGNITUDE
    elif label == "eta5_alt":
        e[0, 0] = EIGENSTRAIN_MAGNITUDE
        e[1, 1] = -EIGENSTRAIN_MAGNITUDE
    elif label == "dilation":
        e[0, 0] = e[1, 1] = e[2, 2] = EIGENSTRAIN_MAGNITUDE
    else:
        raise ValueError(f"unknown label {label!r}")
    return e


def _build_case():
    grid = Grid(shape=GRID_SHAPE, lengths=(1.0, 1.0, 1.0))
    return HoleInPlateCase(
        grid, matrix_lame_lambda=MATRIX_LAME_LAMBDA, matrix_lame_mu=MATRIX_LAME_MU,
        hole_radius=HOLE_RADIUS, contrast=1.0, center=(0.5, 0.5),
    )


# Which Cartesian stress components are physically nonzero along the
# x1=center probe line for each eigenstrain -- eta5 (pure shear
# eigenstrain eps_12) has the same node/peak symmetry along the x2 axis
# that shear *loading* does elsewhere in this module: sigma_11 and
# sigma_22 vanish there exactly (checked directly), so plotting them is
# just floating-point noise around zero, not signal -- sigma_12 is the
# nonzero component to compare instead. eta4 (antiplane eps_13) has the
# same node there too, same reasoning as elliptical_inhomogeneity.py's
# own eta4 case -- sigma_13 is the nonzero component. eta5_alt (the same
# shear, as eps_11=-eps_22) instead has sigma_11/sigma_22 as its nonzero
# pair along this line -- matching the reference material's own
# eigenstrain.5.alt figure, which plots exactly those two.
_COMPONENTS = {
    "eta0": ((0, 0), (1, 1)),
    "eta2": ((0, 0), (1, 1)),
    "eta4": ((0, 2), None),
    "eta5": ((0, 1), None),
    "eta5_alt": ((0, 0), (1, 1)),
    "dilation": ((0, 0), (1, 1)),
}


def _line_through_hole(case, solver, eigenstrain_tensor, components):
    """Return (xi, value_a, value_b)/(mu*EIGENSTRAIN_MAGNITUDE) along the
    line x1=center[0], x2 varying, for the given (i,j) stress components
    (either may be None)."""
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
    xi = (x2 - case.center[1]) / HOLE_RADIUS
    comp_a, comp_b = components
    value_a = sigma[comp_a[0], comp_a[1], col, :, 0] / scale
    value_b = sigma[comp_b[0], comp_b[1], col, :, 0] / scale if comp_b is not None else None
    return xi, value_a, value_b


def _periodic_analytic_line(case, eigenstrain_tensor, components):
    """Return (xi, value_a, value_b)/(mu*EIGENSTRAIN_MAGNITUDE) along the
    same line, from
    :meth:`HoleInPlateCase.periodic_prescribed_eigenstrain_solution`."""
    grid = case.grid
    _, stress = case.periodic_prescribed_eigenstrain_solution(eigenstrain_tensor)
    stress = np.asarray(stress)

    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    col = np.argmin(np.abs(x1 - case.center[0]))

    scale = MATRIX_LAME_MU * EIGENSTRAIN_MAGNITUDE
    xi = (x2 - case.center[1]) / HOLE_RADIUS
    comp_a, comp_b = components
    value_a = stress[comp_a[0], comp_a[1], col, :, 0] / scale
    value_b = stress[comp_b[0], comp_b[1], col, :, 0] / scale if comp_b is not None else None
    return xi, value_a, value_b


_LABELS = {
    (0, 0): r"$\sigma_{11}$", (1, 1): r"$\sigma_{22}$",
    (0, 1): r"$\sigma_{12}$", (0, 2): r"$\sigma_{13}$",
}


def plot_eigenstrain(label):
    eigenstrain_tensor = _eigenstrain_tensor(label)
    components = _COMPONENTS[label]
    print(f"{label}:")
    case = _build_case()
    solver = case.solver()
    xi, sa, sb = _line_through_hole(case, solver, eigenstrain_tensor, components)
    _, pa, pb = _periodic_analytic_line(case, eigenstrain_tensor, components)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.set_xlim(-4, 4)
    analytic_numeric_curve(ax, xi, pa, sa, _LABELS[components[0]], "C0")
    if components[1] is not None:
        analytic_numeric_curve(ax, xi, pb, sb, _LABELS[components[1]], "C1")
    ax.set_xlabel(r"$x_2 / r_0$")
    ax.set_ylabel(r"$\sigma / (\mu \varepsilon^0)$")
    ax.legend(fontsize=7, ncol=2)
    save_all(fig, f"elastic_deformation.eigenstrain_{label}")
    plt.close(fig)


if __name__ == "__main__":
    plot_eigenstrain("eta0")
    plot_eigenstrain("eta2")
    plot_eigenstrain("eta4")
    plot_eigenstrain("eta5")
    plot_eigenstrain("eta5_alt")
    plot_eigenstrain("dilation")
