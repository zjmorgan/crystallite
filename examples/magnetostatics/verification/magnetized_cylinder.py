"""Field of a uniformly magnetized cylinder, from
:class:`crystallite.magnetostatics.Magnetostatics`.

The magnetization is uniform along :math:`x_1` in a disk of radius :math:`r_0`.
Its bound "magnetic charge" is a surface charge :math:`M\\cdot n`; isolated,
the field inside is uniform, the demagnetizing field :math:`H_\\mathrm{in}=-M/2`,
and outside a dipole. Two cuts through the center are plotted (coordinates measured from it), :math:`x_1` (along
:math:`M`: the normal component of :math:`H` jumps by :math:`M` at the surface)
and :math:`x_2` (across :math:`M`: the tangential component is continuous), for
two quantities:

* the magnetic field :math:`H_1`, normalized by :math:`M`;
* the flux density :math:`B_1=\\mu_0(H_1+M)`, normalized by :math:`\\mu_0M`,
  whose normal component is continuous.

The same mathematics as the uniformly polarized cylinder in
``examples/electrostatics`` (:math:`\\varepsilon=1`, :math:`P\\to M`,
:math:`E\\to H`, :math:`D\\to B/\\mu_0`), shown with the magnetic quantities.

Analytic reference:
:func:`crystallite.verification.magnetostatics.magnetized_disk_fields` -- images
of the isolated dipole field with the exact periodic interior
:math:`-\\tfrac12(1-f)M` (area fraction :math:`f`) and a zero cell-mean field.
The solve uses the Lanczos-smoothed magnetization, so within a few grid cells of
the surface the numeric field is the smoothed surface charge and the curves
differ there.
"""

from __future__ import annotations

import numpy as np

from _plotting import analytic_numeric_curve, component_legend, plt, save_all
from crystallite.grid import Grid
from crystallite.magnetostatics import Magnetostatics
from crystallite.verification.magnetostatics import magnetized_disk_fields, smoothed_disk

PERMEABILITY = 1.0  # mu_0, nondimensional
MAGNETIZATION = 1.0
RADIUS = 0.1
GRID_SHAPE = (512, 512, 1)
CENTER = (0.5, 0.5)


def _cuts(field, grid):
    """``(xi, F_1 along x_2, F_1 along x_1)`` of a ``(3,) + grid.shape`` field
    (through the center), for the two cuts against ``x / r_0`` (from the center)."""
    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    column = np.argmin(np.abs(x1 - CENTER[0]))
    row = np.argmin(np.abs(x2 - CENTER[1]))
    return (x2 - CENTER[1]) / RADIUS, field[0, column, :, 0], field[0, :, row, 0]


def _fields():
    grid = Grid(shape=GRID_SHAPE, lengths=(1.0, 1.0, 1.0))
    magnetization = np.zeros((3,) + grid.shape, dtype=np.float32)
    magnetization[0] = MAGNETIZATION * smoothed_disk(grid, RADIUS, CENTER)
    solution = Magnetostatics(grid, PERMEABILITY).solve(magnetization)
    reference = magnetized_disk_fields(
        grid, RADIUS, [MAGNETIZATION, 0.0, 0.0], PERMEABILITY, CENTER
    )
    return grid, (np.asarray(solution.magnetic_field), np.asarray(solution.flux_density)), reference


def plot_magnetized_cylinder():
    grid, numeric, analytic = _fields()
    for index, (quantity, scale, name) in enumerate(
        (("H", "M", "field"), ("B", r"\mu_0 M", "flux_density"))
    ):
        normalization = MAGNETIZATION if index == 0 else PERMEABILITY * MAGNETIZATION
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
        ax.set_xlabel(r"$x / r_0$")
        ax.set_ylabel(rf"${quantity}_1 / ({scale})$" if " " in scale else rf"${quantity}_1 / {scale}$")
        ax.set_title("uniformly magnetized cylinder")
        component_legend(ax)
        save_all(fig, f"magnetostatics.magnetized_cylinder_{name}")
        plt.close(fig)


if __name__ == "__main__":
    plot_magnetized_cylinder()
