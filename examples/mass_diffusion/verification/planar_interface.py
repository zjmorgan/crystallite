"""``planar.interface.pgf``: analytical vs. numerical diffuse planar
interface, reproducing the comparison from the presentation.

A single tanh interface is only a valid equilibrium on an *infinite*
domain: a periodic domain's left and right boundaries are identified, so
a kink from ``c_alpha`` to ``c_beta`` cannot close up on itself without a
second, opposite-signed interface (an antikink) to bring the field back.
The exact periodic equilibrium of ``kappa * c'' = f'(c)`` is a Jacobi
elliptic ``sn`` profile (:func:`crystallite.mass_diffusion.cahn_hilliard_periodic_profile`),
not a single tanh -- comparing a numerically relaxed profile against a
naive single tanh (as an earlier version of this script did) shows
Gibbs-ringing artifacts at the domain wrap and a slow, spurious-looking
drift, because the field is correctly relaxing *away* from that invalid
target and toward the true kink-antikink equilibrium.

The numerical profile starts from the exact periodic profile at a
deliberately different (larger) gradient-energy coefficient -- itself an
exact equilibrium of a *different* kappa, so still smooth and fully
periodic -- and relaxes under :class:`crystallite.mass_diffusion.MassDiffusion`
(``scheme="semi_implicit"``) toward the correct one, converging to within
RMS ~3e-4 of the analytic profile.
"""

from __future__ import annotations

import numpy as np

from _plotting import plt, save_all
from crystallite.mass_diffusion import (
    MassDiffusion,
    cahn_hilliard_periodic_profile,
    double_well_curvature,
    double_well_equilibrium_width,
)
from crystallite.grid import Grid

A, KAPPA = 1.0, 2.0e-3
LEFT, RIGHT = -1.0, 1.0
INITIAL_KAPPA = 1.5 * KAPPA


def plot_planar_interface():
    width = double_well_equilibrium_width(A, LEFT, RIGHT, KAPPA)
    grid = Grid(shape=(256, 1, 1), lengths=(1.0, 1.0, 1.0))
    length = grid.lengths[0]
    x = np.asarray(grid.x[0]).reshape(-1)
    center = 0.5 * length

    analytic = np.asarray(
        cahn_hilliard_periodic_profile(x, length, A, LEFT, RIGHT, KAPPA, center=center)
    )
    initial = np.asarray(
        cahn_hilliard_periodic_profile(
            x, length, A, LEFT, RIGHT, INITIAL_KAPPA, center=center
        )
    )

    solver = MassDiffusion(
        grid, mobility=1.0, gradient_energy=KAPPA,
        bulk_free_energy_coefficient=A, left_well=LEFT, right_well=RIGHT,
        scheme="semi_implicit", reference_mobility=1.0,
        reference_gradient_energy=KAPPA,
        reference_curvature=double_well_curvature(0.0, a=A, c_alpha=LEFT, c_beta=RIGHT),
    )

    field = initial.reshape(grid.shape)
    for _ in range(2000):
        field = solver.step(field, 5.0e-6)
    numerical = np.asarray(field).reshape(-1)

    zoom = (x > center - 6 * width) & (x < center + 6 * width)

    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    ax.plot(x[zoom], analytic[zoom], color="C0", label="analytical (periodic)")
    ax.plot(
        x[zoom], numerical[zoom], "o", color="C1", markersize=3,
        markevery=2, label="numerical (relaxed)",
    )
    ax.set_xlabel(r"position $x$")
    ax.set_ylabel(r"composition $c$")
    ax.legend()
    save_all(fig, "planar.interface")
    plt.close(fig)

    rms_error = np.sqrt(np.mean((analytic[zoom] - numerical[zoom]) ** 2))
    print(f"analytic width = {width:.6g}, RMS error = {rms_error:.3g}")


if __name__ == "__main__":
    plot_planar_interface()
