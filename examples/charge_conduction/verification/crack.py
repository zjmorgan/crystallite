r"""Current density ahead of the tip of an insulating crack, under a mean electric field
across it (Ohm's law) -- flow forced around the crack, with the tip singularity
:math:`j_1/j_\infty\to x_2/\sqrt{x_2^2-a^2}` (isolated).

The crack is a genuinely thin insulating ellipse, oriented like the elastic
``thin_crack.py``: long axis (half-length :math:`a=r_0`) along :math:`x_2`,
faces normal to :math:`x_1`, half-width ``SLIT_HALF_WIDTH_PIXELS`` grid spacings
(a 5-pixel slit), field along :math:`x_1`, across it. The scan is along the
crack axis (:math:`x_1=c_1`, with :math:`x_2` measured from the crack center) from
:math:`1.1a` outward, :math:`j_1` normalized
by :math:`j_\infty`.

Analytic reference:
:meth:`crystallite.verification.charge_conduction.ChargeInclusionCase.periodic_current_density`
-- the exact-in-elliptic-coordinates perturbation
(:func:`crystallite.verification.charge_conduction.elliptical_perturbation`) of the
periodic array, at the same contrast and slit width the numeric solve uses. It
includes the periodic-cell correction (the crack spans 20% of the cell: at
:math:`1.1a` the isolated value is 2.4, the periodic one 2.3).

The very near-tip sample (< ~1.15 a) is left out: the tip radius of curvature
:math:`b^2/a` is far below a grid cell, so the numeric peak there is
resolution-limited, not converged (as with the elastic ``thin_crack.py``).

Numeric results are cached in ``figures/_numeric_cache``.
"""

from __future__ import annotations

import numpy as np

from _plotting import analytic_numeric_curve, cached_numeric, component_legend, plt, save_all
from crystallite.grid import Grid
from crystallite.verification.charge_conduction import ChargeInclusionCase

CRACK_HALF_LENGTH = 0.1
SLIT_HALF_WIDTH_PIXELS = 2.5
CONTRAST = 1.0e-3
SMOOTHING_WIDTH = 0.5
GRID_SHAPE = (512, 512, 1)
FIELD = (1.0, 0.0, 0.0)
XI_MIN, XI_MAX = 1.1, 4.0
N_MODES = 1024  # the slit is ~1/300 of the cell: the lattice sum needs modes to match


def _case():
    grid = Grid(shape=GRID_SHAPE, lengths=(1.0, 1.0, 1.0))
    width = SLIT_HALF_WIDTH_PIXELS * grid.spacing[1]
    return ChargeInclusionCase(
        grid, 1.0, width, CRACK_HALF_LENGTH, contrast=CONTRAST, smoothing_width=SMOOTHING_WIDTH
    )


def _numeric_flux():
    solution = _case().solver().solve(FIELD, tol=1.0e-6, max_iterations=30000)
    print(f"  numeric: converged={solution.converged} iterations={solution.iterations} "
          f"residual={solution.residual_norm:.3g}")
    return np.asarray(solution.flux)


def plot_crack():
    case = _case()
    numeric = cached_numeric(
        f"conduction_crack_x2_N{GRID_SHAPE[0]}_a{CRACK_HALF_LENGTH}_hw{SLIT_HALF_WIDTH_PIXELS}"
        f"_w{SMOOTHING_WIDTH}_c{CONTRAST}", _numeric_flux,
    )
    analytic = case.periodic_current_density(FIELD, n_modes=N_MODES)

    x2 = np.asarray(case.grid.x[1])[0, :, 0]
    column = np.argmin(np.abs(np.asarray(case.grid.x[0])[:, 0, 0] - case.center[0]))
    xi = (x2 - case.center[1]) / CRACK_HALF_LENGTH
    keep = (xi >= XI_MIN) & (xi <= XI_MAX)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    analytic_numeric_curve(
        ax, xi[keep], analytic[0][column, :, 0][keep], numeric[0][column, :, 0][keep],
        r"$j_1$", "C0", downsample=6,
    )
    ax.set_xlim(XI_MIN, XI_MAX)
    ax.set_xlabel(r"$x_2 / a$")
    ax.set_ylabel(r"$j_1 / j_\infty$")
    ax.set_title("current density ahead of an insulating crack tip")
    component_legend(ax)
    save_all(fig, "charge_conduction.crack")
    plt.close(fig)


if __name__ == "__main__":
    plot_crack()
