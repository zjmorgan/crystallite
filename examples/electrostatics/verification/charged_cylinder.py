"""Electric field of a uniformly charged cylinder, from
:class:`crystallite.electrostatics.Electrostatics`.

Isolated, the field is radial, :math:`E_r=\\rho r/(2\\varepsilon_0)` inside and
:math:`\\rho r_0^2/(2\\varepsilon_0r)` outside a cylinder of radius :math:`r_0`.
A periodic array needs a uniform neutralizing background (the :math:`k=0`
mode is dropped), which the solver and the reference both apply.

The cut is the line through the center along :math:`x_1`, normalized by
:math:`\\rho r_0/(2\\varepsilon_0)`, the field at the surface.

Analytic reference:
:func:`crystallite.verification.electrostatics.charged_disk_field` -- the exact
Bessel series of the disk (no pixelated mask), synthesized on a finer grid. The
numeric solve uses the Lanczos-smoothed charge, as a sharp edge rings at the
grid scale.
"""

from __future__ import annotations

import numpy as np

from _plotting import analytic_numeric_curve, component_legend, plt, save_all
from crystallite.electrostatics import Electrostatics
from crystallite.grid import Grid
from crystallite.verification.electrostatics import charged_disk_field, smoothed_disk

EPSILON_0 = 1.0
CHARGE_DENSITY = 1.0
RADIUS = 0.1
GRID_SHAPE = (512, 512, 1)
CENTER = (0.5, 0.5)


def _cut(field, grid):
    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    row = np.argmin(np.abs(x2 - CENTER[1]))
    return (x1 - CENTER[0]) / RADIUS, field[0, :, row, 0] / (0.5 * CHARGE_DENSITY * RADIUS / EPSILON_0)


def plot_charged_cylinder():
    grid = Grid(shape=GRID_SHAPE, lengths=(1.0, 1.0, 1.0))
    charge = CHARGE_DENSITY * smoothed_disk(grid, RADIUS, CENTER)
    numeric = np.asarray(Electrostatics(grid, EPSILON_0).solve(charge_density=charge).electric_field)
    analytic = charged_disk_field(grid, RADIUS, CHARGE_DENSITY, EPSILON_0, CENTER)
    xi, analytic_cut = _cut(analytic, grid)
    _, numeric_cut = _cut(numeric, grid)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.set_xlim(-4, 4)
    analytic_numeric_curve(ax, xi, analytic_cut, numeric_cut, r"$E_1$", "C0", downsample=8)
    ax.set_xlabel(r"$(x_1 - c_1) / r_0$")
    ax.set_ylabel(r"$E_1 / (\rho r_0 / 2\varepsilon_0)$")
    ax.set_title("uniformly charged cylinder")
    component_legend(ax)
    save_all(fig, "electrostatics.charged_cylinder")
    plt.close(fig)


if __name__ == "__main__":
    plot_charged_cylinder()
