r"""Heat flux from a cylinder generating heat uniformly -- the one case with a
source term, so heat only: charge is conserved and has no analog (a static
charged cylinder is electrostatics, ``examples/electrostatics``).

Fourier's law with a source: :math:`\nabla\cdot q=\varphi`,
:math:`q=\kappa X`, :math:`X=-\nabla T`. The heat flux is radial,
:math:`\varphi r/2` inside and :math:`\varphi r_0^2/(2r)` outside for an isolated
cylinder of radius :math:`r_0`; a periodic array has no steady state unless a
uniform sink balances the generation, so the :math:`k=0` mode is dropped (the
solver does this for a nonzero-mean source).

The cut is the line through the center along :math:`x_1` (:math:`x_2=c_2`, with
:math:`x_1` measured from the center), where the heat flux is purely :math:`q_1`, radial and odd about the center,
normalized by :math:`\varphi r_0/2`, the flux at the surface. By symmetry the
:math:`x_2` cut is the same curve in :math:`q_2`.

Analytic reference:
:func:`crystallite.verification.heat_conduction.heat_generation_flux` -- the exact
Fourier series with the analytic Bessel transform of the disk (no pixelated
mask), synthesized on a finer grid. It is independent of the thermal
conductivity for an isotropic matrix. The numeric solve uses the
Lanczos-smoothed indicator, as the operator is only exact on band-limited
sources.
"""

from __future__ import annotations

import numpy as np

from _plotting import analytic_numeric_curve, cached_numeric, component_legend, plt, save_all
from crystallite.conduction import SteadyConduction
from crystallite.grid import Grid
from crystallite.verification.heat_conduction import heat_generation_flux

THERMAL_CONDUCTIVITY = 1.7
RADIUS = 0.1
GENERATION = 1.0
GRID_SHAPE = (512, 512, 1)
CENTER = (0.5, 0.5)


def _grid():
    return Grid(shape=GRID_SHAPE, lengths=(1.0, 1.0, 1.0))


def _numeric_flux():
    grid = _grid()
    x, y = np.asarray(grid.x[0]), np.asarray(grid.x[1])
    indicator = np.where(np.hypot(x - CENTER[0], y - CENTER[1]) < RADIUS, 1.0, 0.0)
    indicator = np.real(np.asarray(grid.ifft(grid.fft(indicator.astype(np.float32)) * grid.lanczos_filter)))
    solution = SteadyConduction(grid, THERMAL_CONDUCTIVITY).solve(
        [0.0, 0.0, 0.0], source=GENERATION * indicator, tol=1.0e-6
    )
    print(f"  numeric: converged={solution.converged} iterations={solution.iterations} "
          f"residual={solution.residual_norm:.3g}")
    return np.asarray(solution.flux)


def _cut(flux, grid):
    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    row = np.argmin(np.abs(x2 - CENTER[1]))
    return (x1 - CENTER[0]) / RADIUS, flux[0, :, row, 0] / (0.5 * GENERATION * RADIUS)


def plot_heat_generation():
    grid = _grid()
    numeric = cached_numeric(
        f"conduction_heat_generation_N{GRID_SHAPE[0]}_R{RADIUS}_k{THERMAL_CONDUCTIVITY}", _numeric_flux
    )
    analytic = heat_generation_flux(
        grid, RADIUS, GENERATION, center=CENTER, thermal_conductivity=THERMAL_CONDUCTIVITY
    )
    xi, analytic_q = _cut(analytic, grid)
    _, numeric_q = _cut(numeric, grid)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.set_xlim(-4, 4)
    analytic_numeric_curve(ax, xi, analytic_q, numeric_q, r"$q_1$", "C0", downsample=8)
    ax.set_xlabel(r"$x_1 / r_0$")
    ax.set_ylabel(r"$q_1 / (\varphi r_0 / 2)$")
    ax.set_title("cylinder generating heat uniformly")
    component_legend(ax)
    save_all(fig, "heat_conduction.heat_generation")
    plt.close(fig)


if __name__ == "__main__":
    plot_heat_generation()
