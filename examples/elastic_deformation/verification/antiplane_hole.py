"""Analytical vs. numerical antiplane shear stress across a circular
hole, under remote antiplane shear -- the "Mode III crack" case from this
project's original (pre-crystallite) reference material (``crack.cpp``),
reinterpreted correctly rather than transcribed.

Checking ``crack.cpp`` directly first (rather than assuming its "Mode
I"/"Mode II"/"Mode III" naming meant three genuinely different closed
forms) turned up that its actual computed formula -- and its own output
filename, ``kirsch.txt`` -- is exactly the classical circular-hole Kirsch
solution already implemented in this project as
``hole_in_plate.py``'s tension ("Mode I") and shear ("Mode II") cases;
verified directly, not assumed: evaluating both formulas at matching
points gives identical values (to floating-point precision) once
``crack.cpp``'s own background-subtracted convention is accounted for.
So "Mode I"/"Mode II" needed no new code at all -- they are a circular
hole under remote tension/shear, illustrating the qualitative shape of a
crack-tip plastic zone via the classical stress-concentration solution,
not the sharp :math:`1/\\sqrt{r}` linear-elastic-fracture-mechanics
crack-tip singularity a literal crack has.

"Mode III" (antiplane shear) is different: no existing closed form in
this project covered it, so this module derives one --
:func:`crystallite.verification.elastic_deformation._antiplane_polar_stress`,
the classical circular-hole-under-remote-antiplane-shear solution.
Antiplane elasticity reduces to a scalar Laplace problem for
:math:`u_3(x_1,x_2)` (:math:`\\sigma_{i3}=\\mu\\,\\partial_i u_3`,
equilibrium :math:`\\nabla^2 u_3=0`), considerably simpler than the
biharmonic in-plane problem :func:`~crystallite.verification.elastic_deformation._tension_polar_stress`
solves -- verified three independent ways before use here (harmonic
equilibrium, the traction-free boundary condition holding at every angle,
and the closed form matching direct finite-difference derivatives of
:math:`u_3` to numerical precision), plus a classical sanity check: the
hoop stress concentration factor this gives is exactly 2 (vs. 3 for the
in-plane tension case), the well-known antiplane result.

Remote loading is :math:`\\sigma_{13}` (not :math:`\\sigma_{23}`) --
matching this project's own convention of loading along the axis
:math:`\\theta=0` is measured from (:meth:`~crystallite.verification.HoleInPlateCase.remote_stress`'s
:math:`\\sigma_{11}` for "tension"), confirmed against the reference
material's own "Mode III crack" illustration, which loads its vertical
crack along the same horizontal axis its "Mode I"/"Mode II" tension/shear
illustrations do. Along this module's own probe line (:math:`\\theta=\\pi/2`,
straight up from the hole, same line ``hole_in_plate.py`` uses),
:math:`\\sigma_{23}` has a node there and :math:`\\sigma_{13}` carries the
concentration -- the antiplane analog of tension's :math:`\\sigma_{11}`
being the interesting component on that same line, not :math:`\\sigma_{22}`.

Same 1D line-scan-through-the-hole convention as ``hole_in_plate.py``,
and the same image-summed ``eshelby.cpp``-recipe periodic reference, via
:meth:`crystallite.verification.HoleInPlateCase.periodic_antiplane_inhomogeneity_stress`
-- the finite-`contrast` generalization of
:meth:`~crystallite.verification.HoleInPlateCase.periodic_antiplane_void_stress`,
mirroring ``hole_in_plate.py``'s own move from
:meth:`~crystallite.verification.HoleInPlateCase.periodic_void_stress` to
:meth:`~crystallite.verification.HoleInPlateCase.periodic_inhomogeneity_stress`:
this module's numeric solve uses the same soft (``CONTRAST=1.0e-3``), not
literally void, hole as ``hole_in_plate.py``'s, so the void-only closed
form left the same void-vs-actual-contrast mismatch here that motivated
that generalization there.

That in-plane method also recenters its image sum to
:meth:`~crystallite.verification.HoleInPlateCase.periodic_analytic_solution`'s
own domain-mean stress (the Eshelby-eigenstrain FFT construction,
calibrated to this solver's strain-, not stress-, controlled boundary
condition) rather than the nominal remote magnitude ``eshelby.cpp``
itself recenters to. The antiplane counterpart of that FFT construction,
:meth:`~crystallite.verification.HoleInPlateCase.periodic_antiplane_analytic_solution`,
now exists too (an antiplane equivalent eigenstrain, solved exactly via
the same fully general :meth:`ElasticDeformation.reference_green` this
project's in-plane cases already use -- nothing about that machinery is
actually specific to in-plane loading), so
``periodic_antiplane_inhomogeneity_stress`` recenters to *its* domain
mean, exactly paralleling ``periodic_inhomogeneity_stress``.
"""

from __future__ import annotations

import numpy as np

from _plotting import analytic_numeric_curve, plt, save_all
from crystallite.grid import Grid
from crystallite.verification import HoleInPlateCase

MATRIX_LAME_LAMBDA = 1.0
MATRIX_LAME_MU = 0.7
HOLE_RADIUS = 0.1
CONTRAST = 1.0e-3
GRID_SHAPE = (256, 256, 1)
MAGNITUDE = 0.01
N_IMAGES = 1


def _build_case():
    grid = Grid(shape=GRID_SHAPE, lengths=(1.0, 1.0, 1.0))
    return HoleInPlateCase(
        grid, matrix_lame_lambda=MATRIX_LAME_LAMBDA, matrix_lame_mu=MATRIX_LAME_MU,
        hole_radius=HOLE_RADIUS, contrast=CONTRAST, center=(0.5, 0.5), dealias=True,
    )


def _line_through_hole(case, solver):
    """Return (xi, sigma_13)/MAGNITUDE along the line x1=center[0], x2
    varying."""
    grid = case.grid
    eps_bar = case.macro_antiplane_strain(MAGNITUDE, solver=solver)
    sol = solver.solve(eps_bar, tol=1.0e-6, max_iterations=3000)
    print(f"antiplane: converged={sol.converged}, iterations={sol.iterations}, "
          f"residual={sol.residual_norm:.3g}")
    sigma = np.asarray(sol.stress)

    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    col = np.argmin(np.abs(x1 - case.center[0]))

    xi = (x2 - case.center[1]) / HOLE_RADIUS
    sigma_13 = sigma[0, 2, col, :, 0] / MAGNITUDE
    return xi, sigma_13


def _periodic_analytic_line(case):
    """Return (xi, sigma_13)/MAGNITUDE along the same line, from
    :meth:`HoleInPlateCase.periodic_antiplane_inhomogeneity_stress`."""
    grid = case.grid
    stress = np.asarray(case.periodic_antiplane_inhomogeneity_stress(MAGNITUDE, n_images=N_IMAGES))

    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    col = np.argmin(np.abs(x1 - case.center[0]))

    xi = (x2 - case.center[1]) / HOLE_RADIUS
    sigma_13 = stress[0, 2, col, :, 0] / MAGNITUDE
    return xi, sigma_13


def plot_antiplane():
    case = _build_case()
    solver = case.solver()
    xi, sa = _line_through_hole(case, solver)
    _, pa = _periodic_analytic_line(case)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.set_xlim(-4, 4)
    analytic_numeric_curve(ax, xi, pa, sa, r"$\sigma_{13}$", "C0")
    ax.set_xlabel(r"$x_2 / r_0$")
    ax.set_ylabel(r"$\sigma / \tau_\infty$")
    ax.legend(fontsize=7, ncol=2)
    save_all(fig, "elastic_deformation.antiplane_hole")
    plt.close(fig)


if __name__ == "__main__":
    plot_antiplane()
