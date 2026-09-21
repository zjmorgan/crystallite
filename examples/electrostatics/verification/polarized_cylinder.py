"""Field of a uniformly polarized cylinder, from
:class:`crystallite.electrostatics.Electrostatics`.

The polarization is uniform along :math:`x_1` in a disk of radius :math:`r_0`.
Its bound charge is a surface charge :math:`P\\cdot n`; isolated, the field
inside is uniform, :math:`E_\\mathrm{in}=-P/2\\varepsilon_0`, and outside a
dipole. Two cuts through the center are plotted, :math:`x_1` (along
:math:`P`: the normal component of :math:`E` jumps by :math:`P/\\varepsilon_0` at
the surface) and :math:`x_2` (across :math:`P`: the tangential component is
continuous), for two quantities:

* the electric field :math:`E_1`, normalized by :math:`P/\\varepsilon_0`;
* the displacement :math:`D_1=\\varepsilon_0E_1+P`, normalized by :math:`P`,
  whose normal component is continuous.

Analytic reference:
:func:`crystallite.verification.electrostatics.polarized_disk_fields` -- images
of the isolated dipole field with the exact periodic interior
:math:`-\\tfrac12(1-f)P/\\varepsilon_0` (area fraction :math:`f`; the isolated
value is :math:`-P/2\\varepsilon_0`) and a zero cell-mean field. The solve uses
the Lanczos-smoothed polarization, so within a few grid cells of the surface the
numeric field is the smoothed surface charge and the curves differ there.
"""

from __future__ import annotations

import numpy as np

from _plotting import analytic_numeric_curve, component_legend, plt, save_all
from crystallite.electrostatics import Electrostatics
from crystallite.grid import Grid
from crystallite.verification.electrostatics import polarized_disk_fields, smoothed_disk

EPSILON_0 = 1.0
POLARIZATION = 1.0
RADIUS = 0.1
GRID_SHAPE = (512, 512, 1)
CENTER = (0.5, 0.5)

def _cuts(field, grid):
    """``(xi, F_1 along x_2, F_1 along x_1)`` of a ``(3,) + grid.shape`` field
    (through the center), for the two cuts against ``(x - c) / r_0``."""
    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    column = np.argmin(np.abs(x1 - CENTER[0]))
    row = np.argmin(np.abs(x2 - CENTER[1]))
    return (x2 - CENTER[1]) / RADIUS, field[0, column, :, 0], field[0, :, row, 0]


def _fields():
    grid = Grid(shape=GRID_SHAPE, lengths=(1.0, 1.0, 1.0))
    polarization = np.zeros((3,) + grid.shape, dtype=np.float32)
    polarization[0] = POLARIZATION * smoothed_disk(grid, RADIUS, CENTER)
    solution = Electrostatics(grid, EPSILON_0).solve(polarization=polarization)
    reference = polarized_disk_fields(grid, RADIUS, [POLARIZATION, 0.0, 0.0], EPSILON_0, CENTER)
    return grid, (np.asarray(solution.electric_field), np.asarray(solution.displacement)), reference


def plot_polarized_cylinder():
    grid, numeric, analytic = _fields()
    for index, (quantity, scale, name) in enumerate(
        (("E", r"P / \varepsilon_0", "field"), ("D", "P", "displacement"))
    ):
        normalization = POLARIZATION / EPSILON_0 if index == 0 else POLARIZATION
        xi, a_transverse, a_parallel = _cuts(analytic[index], grid)
        _, n_transverse, n_parallel = _cuts(numeric[index], grid)
        fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
        ax.set_xlim(-4, 4)
        analytic_numeric_curve(
            ax, xi, a_transverse / normalization, n_transverse / normalization,
            rf"${quantity}_1(x_2)$", "C0", downsample=8,
        )
        analytic_numeric_curve(
            ax, xi, a_parallel / normalization, n_parallel / normalization,
            rf"${quantity}_1(x_1)$", "C1", downsample=8,
        )
        ax.set_xlabel(r"$(x - c) / r_0$")
        ax.set_ylabel(rf"${quantity}_1 / ({scale})$" if " " in scale else rf"${quantity}_1 / {scale}$")
        ax.set_title("uniformly polarized cylinder")
        component_legend(ax)
        save_all(fig, f"electrostatics.polarized_cylinder_{name}")
        plt.close(fig)


if __name__ == "__main__":
    plot_polarized_cylinder()
