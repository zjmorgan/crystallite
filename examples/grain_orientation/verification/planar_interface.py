"""Analytical vs. numerical planar grain boundary.

The analytic profile is the exact kink-antikink equilibrium
(:class:`crystallite.verification.PlanarGrainBoundaryCase`, reusing
:func:`crystallite.mass_diffusion.cahn_hilliard_periodic_profile` directly
-- algebraically the same quartic double-well ODE as the Cahn-Hilliard
case, just for one active order parameter with the rest held at zero). The
numerical profile starts from the same exact profile at a deliberately
different (larger) gradient-energy coefficient and relaxes under
:class:`crystallite.grain_orientation.GrainOrientation` toward the correct
one.
"""

from __future__ import annotations

import numpy as np

from _plotting import plt, save_all
from crystallite.grain_orientation import GrainOrientation
from crystallite.grid import Grid
from crystallite.verification import PlanarGrainBoundaryCase

A, B, GAMMA = 1.0, 1.0, 1.5
GRADIENT_ENERGY = 2.0e-3
INITIAL_GRADIENT_ENERGY = 1.5 * GRADIENT_ENERGY


def plot_planar_interface():
    grid = Grid(shape=(256, 1, 1), lengths=(1.0, 1.0, 1.0))
    analytic_case = PlanarGrainBoundaryCase(
        grid, barrier_coefficient=A, quartic_coefficient=B,
        gradient_energy=GRADIENT_ENERGY, center=0.5,
    )
    initial_case = PlanarGrainBoundaryCase(
        grid, barrier_coefficient=A, quartic_coefficient=B,
        gradient_energy=INITIAL_GRADIENT_ENERGY, center=0.5,
    )

    model = GrainOrientation(
        grid, mobility=1.0, gradient_energy=GRADIENT_ENERGY,
        barrier_coefficient=A, quartic_coefficient=B, cross_coefficient=GAMMA,
    )

    eta = initial_case.profile()
    for _ in range(500):
        eta = model.step(eta, 1.0)

    x = np.asarray(grid.x[0]).reshape(-1)
    analytic = np.asarray(analytic_case.profile())[0].reshape(-1)
    numerical = np.asarray(eta)[0].reshape(-1)

    width = analytic_case.width
    eta0 = analytic_case.eta0
    center = analytic_case.center * grid.lengths[0]
    zoom = (x > center - 6 * width) & (x < center + 6 * width)

    # Nondimensionalize position by the interface width and eta by its
    # saturation value eta0, so the profile always spans roughly [-6, 6] and
    # [-1, 1] regardless of the chosen barrier/quartic/gradient coefficients.
    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    ax.plot(
        (x[zoom] - center) / width, analytic[zoom] / eta0, color="C0",
        label="analytical (periodic)",
    )
    ax.plot(
        (x[zoom] - center) / width, numerical[zoom] / eta0, "o", color="C1",
        markersize=3, markevery=2, label="numerical (relaxed)",
    )
    ax.set_xlabel(r"$(x - x_0) / \delta$")
    ax.set_ylabel(r"$\eta / \eta_0$")
    ax.legend()
    save_all(fig, "grain_orientation.planar_interface")
    plt.close(fig)

    rms_error = np.sqrt(np.mean((analytic[zoom] - numerical[zoom]) ** 2))
    print(f"analytic width = {width:.6g}, RMS error = {rms_error:.3g}")


if __name__ == "__main__":
    plot_planar_interface()
