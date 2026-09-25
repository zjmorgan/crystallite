r"""Heat flux across a circular *inhomogeneity* -- hole, soft, hard and perfectly
conducting -- under a uniform far-field thermal driving force: Fourier's law,
:math:`q=\kappaX` with :math:`X=-\nablaT`
(:class:`crystallite.conduction.SteadyConduction`).

Two cuts through the center of the inclusion (coordinates measured from its
center), with the field along :math:`x_1`:
the transverse cut (:math:`x_1=c_1`, :math:`x_2` varying) and the parallel cut
(:math:`x_2=c_2`, :math:`x_1` varying, through the stagnation points). On both,
:math:`q_2=0` by symmetry, so :math:`q_1` is plotted, normalized by the
far-field heat flux :math:`q_\infty=\kappa_0X_\infty` in the matrix.

Analytic reference:
:meth:`crystallite.verification.heat_conduction.HeatInclusionCase.periodic_heat_flux`
-- the exact interior from the lattice-sum depolarization tensor and the
exterior from images of the isolated dipole field, no FFT and no grid, so it
includes the periodic-cell correction (area fraction ~3%). The interior
heat flux is :math:`\kappa_1X_\mathrm{in}`, uniform. For the perfect conductor
:math:`X_\mathrm{in}=0` while :math:`\kappa_1\to\infty`, and the product has the
finite limit :math:`-\kappa_0X^*` (:math:`2q_\infty/(1-f)` for a circle, with
:math:`f` the area fraction), which is what is drawn. The numeric solve uses
:math:`\kappa_1/\kappa_0=10^3` (float32 noise grows beyond that), so its interior
sits a few percent above the limit, from the finite interface width.

Numerics use a narrow ``tanh`` interface (``SMOOTHING_WIDTH`` grid spacings):
a hard or perfectly conducting inclusion needs it, because an arithmetic-mean
interface grows the conductor's effective radius, and the exterior dipole with
its square.

Numeric results are cached in ``figures/_numeric_cache``.
"""

from __future__ import annotations

import numpy as np

from _plotting import analytic_numeric_curve, cached_numeric, component_legend, plt, save_all
from crystallite.grid import Grid
from crystallite.verification.heat_conduction import HeatInclusionCase

MATRIX_CONDUCTIVITY = 1.0
RADIUS = 0.1
GRID_SHAPE = (512, 512, 1)
SMOOTHING_WIDTH = 0.5
FIELD = (1.0, 0.0, 0.0)
CENTER = (0.5, 0.5)

# label -> (contrast, title)
CASES = {
    "hole": (1.0e-3, "insulating hole"),
    "soft": (0.5, r"soft inclusion, $\kappa_1/\kappa_0=1/2$"),
    "hard": (1.5, r"hard inclusion, $\kappa_1/\kappa_0=3/2$"),
    "perfect": (float("inf"), "perfectly conducting inclusion"),
}


def _case(contrast):
    grid = Grid(shape=GRID_SHAPE, lengths=(1.0, 1.0, 1.0))
    return HeatInclusionCase(
        grid, MATRIX_CONDUCTIVITY, RADIUS, RADIUS, contrast=contrast, center=CENTER,
        smoothing_width=SMOOTHING_WIDTH,
    )


def _cuts(field, case):
    """``(xi, q1 along x2, q1 along x1)`` of a ``(3,) + grid.shape`` heat flux, the
    two cuts through the center, against ``x / r0`` (from the center)."""
    x1 = np.asarray(case.grid.x[0])[:, 0, 0]
    x2 = np.asarray(case.grid.x[1])[0, :, 0]
    column = np.argmin(np.abs(x1 - CENTER[0]))
    row = np.argmin(np.abs(x2 - CENTER[1]))
    scale = MATRIX_CONDUCTIVITY * FIELD[0]
    return (x2 - CENTER[1]) / RADIUS, field[0, column, :, 0] / scale, field[0, :, row, 0] / scale


def _numeric_flux(contrast):
    solution = _case(contrast).solver().solve(FIELD, tol=1.0e-6, max_iterations=30000)
    print(f"  numeric: converged={solution.converged} iterations={solution.iterations} "
          f"residual={solution.residual_norm:.3g}")
    return np.asarray(solution.flux)


def plot_inhomogeneity(label):
    contrast, title = CASES[label]
    print(f"{label} (contrast={contrast}):")
    case = _case(contrast)
    numeric = cached_numeric(
        f"conduction_inhomogeneity_{label}_c{contrast}_N{GRID_SHAPE[0]}_w{SMOOTHING_WIDTH}_R{RADIUS}",
        lambda: _numeric_flux(contrast),
    )
    analytic = case.periodic_heat_flux(FIELD)

    xi, a1, a2 = _cuts(analytic, case)
    _, n1, n2 = _cuts(numeric, case)
    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.set_xlim(-4, 4)
    analytic_numeric_curve(ax, xi, a1, n1, r"$q_1(x_2)$", "C0", downsample=8)
    analytic_numeric_curve(ax, xi, a2, n2, r"$q_1(x_1)$", "C1", downsample=8)
    ax.set_xlabel(r"$x / r_0$")
    ax.set_ylabel(r"$q_1 / q_\infty$")
    ax.set_title(title)
    component_legend(ax)
    save_all(fig, f"heat_conduction.inhomogeneity_{label}")
    plt.close(fig)


if __name__ == "__main__":
    for name in CASES:
        plot_inhomogeneity(name)
