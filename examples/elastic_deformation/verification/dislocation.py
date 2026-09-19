"""Stress ahead of one tip of a dislocation, modeled the same way this
project's original (pre-crystallite) reference material draws it
(``hooke.tex``'s ``screw``/``edge``/``edge.alt`` frames): an extremely
eccentric elliptical eigenstrain inclusion
(:class:`crystallite.verification.EllipticalHoleInPlateCase`), long axis
along x1 (the dislocation line), a single grid spacing thin along x2 --
just with the eigenstrain component the misfit carries changed between
the three classical dislocation characters:

- ``screw``: eigenstrain eps_23 (Burgers vector out of plane), matching
  ``hooke.tex``'s own ``screw`` figure, which reads sigma_23. An earlier
  version of this module prescribed eps_13 instead (misreading the
  illustration, not a math error -- see below) and, having checked
  directly that sigma_23 is *exactly* zero along this sampling line for a
  pure eps_13 misfit (mirror symmetry under x2 -> -x2), read sigma_13
  instead to get a nonzero curve. The fix is the mirror image of that
  finding, not a contradiction of it: swapping eps_13 for eps_23 swaps
  which of sigma_13/sigma_23 is the one that vanishes by that same
  symmetry (confirmed directly from
  :func:`crystallite.verification.elastic_deformation._ellipse_inhomogeneity_antiplane_exterior_stress`'s
  own closed form: on this sampling line, sigma_13 depends only on the
  prescribed eps_13, sigma_23 only on eps_23, never both) -- so sigma_23
  is now the nonzero, smoothly decaying quantity, matching the
  illustration exactly rather than needing to read a different component
  to compensate for a misprescribed one.
- ``edge`` (glide): eigenstrain eps_12 (Burgers vector in-plane, parallel
  to the cut). Reads sigma_12 ahead of the tip, matching ``hooke.tex``'s
  own ``edge`` figure.
- ``climb``: eigenstrain eps_22 (a normal, not shear, misfit
  perpendicular to the cut -- the "extra half-plane of atoms" picture of
  an edge dislocation, ``hooke.tex``'s own ``edge.alt`` figure, whose
  inserted-half-plane symbol is drawn rotated 90 degrees from ``edge``'s
  own). Reads *both* sigma_22 and sigma_11 ahead of the tip -- unlike
  ``edge``, where sigma_11/sigma_22 are exactly zero on this line
  (checked directly), a pure eps_22 misfit leaves sigma_11 genuinely
  nonzero there too, comparable in size to sigma_22 itself, matching
  ``hooke.tex``'s own ``edge.alt`` figure, which likewise plots both.

All three read directly ahead of the tip (x2=center[1], x1 beyond the
tip) with no polar rotation, for the same reason ``thin_crack.py``'s
axis-aligned geometry needs none: the slit's own faces are normal to x2,
so the traction on them is exactly sigma_2j.

Sampled over ``xi`` (distance from the dislocation's own *center*, not
the tip, normalized by ``HOLE_RADIUS``) in ``[1, 4]`` -- the same window,
and the same radius (``HOLE_RADIUS``, matching ``hole_in_plate.py``'s/
``cylindrical_inhomogeneity.py``'s own circular hole/inhomogeneity, in
place of an earlier, unrelated ``SEMI_AXIS_A=0.125``) used throughout
this project's crack/hole/inhomogeneity family, for a directly comparable
view across all of them.

The analytic reference is
:meth:`crystallite.verification.EllipticalHoleInPlateCase.periodic_prescribed_eigenstrain_void_stress`
(the closed-form H/T-tensor image sum, this project's own generalization
of the original ``eshelby.cpp`` reference's ``e()``/``f()``/``g()``
recipe to a directly-prescribed eigenstrain) -- not the FFT construction
(:meth:`~crystallite.verification.EllipticalHoleInPlateCase.periodic_prescribed_eigenstrain_solution`):
checked directly (see ``thin_crack.py``'s own history) that the FFT
construction's shape-function aliasing washes out the near-core
concentration for a slit this thin, while the closed form (no shape-FFT
step to alias) does not. Only usable here at all because of two separate
fixes: the in-plane H-tensor's shear-coupling terms (needed for ``edge``)
were off by an exact factor of 2, and ``eshelby.cpp``'s own antiplane
H/T-tensor terms (needed for ``screw``) had a genuine derivation error,
not merely a missing factor -- both fixed directly in
:func:`crystallite.verification.elastic_deformation._ellipse_inhomogeneity_exterior_stress`
and
:func:`~crystallite.verification.elastic_deformation._ellipse_inhomogeneity_antiplane_exterior_stress`
respectively; see those functions' own docstrings.
"""

from __future__ import annotations

import numpy as np

from _plotting import plt, save_all
from crystallite.grid import Grid
from crystallite.verification import EllipticalHoleInPlateCase

MATRIX_LAME_LAMBDA = 1.0
MATRIX_LAME_MU = 0.7
HOLE_RADIUS = 0.1  # matches hole_in_plate.py/cylindrical_inhomogeneity.py's circular hole
GRID_SHAPE = (256, 256, 1)
EIGENSTRAIN_MAGNITUDE = 0.01
N_IMAGES = 1  # periodic image cutoff for the analytic curve (already converged here)

# label -> ((i, j) eigenstrain component prescribed, tuple of
# ((i, j) stress component, plot label) read directly ahead of the tip --
# more than one for "climb", which has a genuinely nonzero sigma_11 too.
_MODES = {
    "screw": ((1, 2), (((1, 2), r"$\sigma_{23}$"),)),
    "edge": ((0, 1), (((0, 1), r"$\sigma_{12}$"),)),
    "climb": ((1, 1), (((1, 1), r"$\sigma_{22}$"), ((0, 0), r"$\sigma_{11}$"))),
}


def _build_case():
    grid = Grid(shape=GRID_SHAPE, lengths=(1.0, 1.0, 1.0))
    semi_axis_b = grid.spacing[1]  # one grid spacing wide, full width
    return EllipticalHoleInPlateCase(
        grid, matrix_lame_lambda=MATRIX_LAME_LAMBDA, matrix_lame_mu=MATRIX_LAME_MU,
        semi_axis_a=HOLE_RADIUS, semi_axis_b=semi_axis_b, contrast=1.0, center=(0.5, 0.5),
    )


def _eigenstrain_tensor(component):
    e = np.zeros((3, 3))
    i, j = component
    e[i, j] = e[j, i] = EIGENSTRAIN_MAGNITUDE
    return e


def _xi_and_mask(case):
    """Grid row at x2=center[1], and the boolean mask/``xi`` (distance
    from the dislocation's own *center*, normalized by `HOLE_RADIUS`) for
    ``xi`` in ``[1, 4]`` -- `xi=1` exactly at the tip, since
    `semi_axis_a == HOLE_RADIUS`."""
    grid = case.grid
    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    row = np.argmin(np.abs(x2 - case.center[1]))
    xi = (x1 - case.center[0]) / HOLE_RADIUS
    mask = (xi >= 1.0) & (xi <= 4.0)
    return row, mask, xi[mask]


def _ahead_of_tip_numeric(case, solver, eigenstrain_tensor, components):
    """Return (xi, [values, ...])/(mu*EIGENSTRAIN_MAGNITUDE) ahead of the
    tip for every (i, j) in `components`, from a single solve."""
    eigenstrain_field = case.eigenstrain_field(eigenstrain_tensor)
    sol = solver.solve(
        np.zeros((3, 3)), eigenstrain=eigenstrain_field, tol=1.0e-6, max_iterations=5000
    )
    print(f"  numerical: converged={sol.converged}, iterations={sol.iterations}, "
          f"residual={sol.residual_norm:.3g}")
    sigma = np.asarray(sol.stress)

    row, mask, xi = _xi_and_mask(case)
    scale = MATRIX_LAME_MU * EIGENSTRAIN_MAGNITUDE
    values = [sigma[i, j, mask, row, 0] / scale for i, j in components]
    return xi, values


def _ahead_of_tip_analytic(case, eigenstrain_tensor, components):
    """Return (xi, [values, ...])/(mu*EIGENSTRAIN_MAGNITUDE) ahead of the
    tip for every (i, j) in `components`, from a single closed-form
    evaluation."""
    stress = np.asarray(
        case.periodic_prescribed_eigenstrain_void_stress(eigenstrain_tensor, n_images=N_IMAGES)
    )
    row, mask, xi = _xi_and_mask(case)
    scale = MATRIX_LAME_MU * EIGENSTRAIN_MAGNITUDE
    values = [stress[i, j, mask, row, 0] / scale for i, j in components]
    return xi, values


def plot_mode(label):
    eigenstrain_component, stress_components = _MODES[label]
    print(f"{label}:")
    case = _build_case()
    solver = case.solver()
    eigenstrain_tensor = _eigenstrain_tensor(eigenstrain_component)
    components = [component for component, _ in stress_components]

    xi_num, s_num = _ahead_of_tip_numeric(case, solver, eigenstrain_tensor, components)
    xi_an, s_an = _ahead_of_tip_analytic(case, eigenstrain_tensor, components)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.set_xlim(1.0, 4.0)
    for color, (_, plot_label), values_num, values_an in zip(
        ("C0", "C1"), stress_components, s_num, s_an
    ):
        ax.plot(xi_an, values_an, "-", color=color, label=f"{plot_label} analytical")
        ax.plot(xi_num, values_num, "o", color=color, ms=4, label=f"{plot_label} numerical")
    ax.set_xlabel(r"$x_1 / r_0$")
    ax.set_ylabel(r"$\sigma / (\mu \varepsilon^0)$")
    ax.legend(fontsize=7)
    save_all(fig, f"elastic_deformation.dislocation_{label}")
    plt.close(fig)


if __name__ == "__main__":
    plot_mode("screw")
    plot_mode("edge")
    plot_mode("climb")
