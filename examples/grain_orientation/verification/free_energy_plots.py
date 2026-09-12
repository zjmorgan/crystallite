"""Homogeneous free-energy landscape for the grain-orientation model.

Reproduces the reference presentation's own figure -- the local free
energy ``f(eta1, eta2)`` as a function of ``eta1`` for a few fixed values
of ``eta2`` (n=2 orientations) -- using
:meth:`crystallite.grain_orientation.GrainOrientation.homogeneous_energy`
directly, not a hand-transcribed formula.
"""

from __future__ import annotations

import numpy as np

from _plotting import plt, save_all
from crystallite.grain_orientation import GrainOrientation
from crystallite.grid import Grid

A, B, GAMMA = 1.0, 1.0, 1.5


def plot_homogeneous_energy():
    # homogeneous_energy is a pointwise function of eta -- it never touches
    # the grid -- so a minimal placeholder grid is enough here.
    grid = Grid(shape=(2, 1, 1))
    model = GrainOrientation(
        grid, barrier_coefficient=A, quartic_coefficient=B, cross_coefficient=GAMMA
    )

    eta1 = np.linspace(-1.5, 1.5, 400)

    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    for eta2 in (0.0, 0.5, 1.0):
        eta = np.zeros((2, len(eta1)))
        eta[0, :] = eta1
        eta[1, :] = eta2
        f = np.asarray(model.homogeneous_energy(eta))
        ax.plot(eta1, f, label=rf"$\eta_2 = {eta2:g}$")

    ax.set_xlabel(r"$\eta_1$")
    ax.set_ylabel(r"free energy density $f$")
    ax.legend()
    save_all(fig, "grain_orientation.free_energy")
    plt.close(fig)


if __name__ == "__main__":
    plot_homogeneous_energy()
