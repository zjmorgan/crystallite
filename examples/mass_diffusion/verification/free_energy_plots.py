"""Bulk free-energy schematics: ``gibbs.pgf`` and ``potential.pgf``.

Reproduces the two conceptual free-energy figures from the original
presentation using :func:`crystallite.mass_diffusion.chemical_free_energy`
rather than a hand-drawn schematic.
"""

from __future__ import annotations

import numpy as np

from _plotting import plt, save_all
from crystallite.mass_diffusion import chemical_free_energy

C_ALPHA, C_BETA = 0.0, 1.0


def plot_gibbs():
    """Free energy of a nonuniform alpha/beta system vs. mole fraction.

    For this symmetric quartic double well both wells sit at f=0, so the
    common tangent construction between the coexisting alpha/beta phases
    is just the horizontal line f=0.
    """
    c = np.linspace(C_ALPHA - 0.1, C_BETA + 0.1, 400)
    f = chemical_free_energy(c, barrier_height=0.25, c_alpha=C_ALPHA, c_beta=C_BETA)

    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    ax.plot(c, f, color="C0")
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    ax.plot([C_ALPHA, C_BETA], [0.0, 0.0], "ko", markersize=5)
    ax.annotate(
        r"$\alpha$", (C_ALPHA, 0.0), textcoords="offset points",
        xytext=(-4, 10), ha="right",
    )
    ax.annotate(
        r"$\beta$", (C_BETA, 0.0), textcoords="offset points",
        xytext=(4, 10), ha="left",
    )
    ax.set_xlabel(r"mole fraction $X_k$")
    ax.set_ylabel(r"free energy $f$")
    save_all(fig, "gibbs")
    plt.close(fig)


def plot_potential():
    """Quartic barrier-height approximation to the referenced free energy,
    f0(X) = 16 f0(X0) [(c_beta-c)/(c_beta-c_alpha)]^2 [(c-c_alpha)/(c_beta-c_alpha)]^2,
    for a few barrier heights f0(X0).
    """
    c = np.linspace(C_ALPHA, C_BETA, 400)
    x0 = 0.5 * (C_ALPHA + C_BETA)

    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    for barrier_height in (0.1, 0.25, 0.5):
        f = chemical_free_energy(
            c, barrier_height=barrier_height, c_alpha=C_ALPHA, c_beta=C_BETA
        )
        ax.plot(c, f, label=rf"$f_0(X_0) = {barrier_height:g}$")
    ax.axvline(x0, color="0.5", linestyle="--", linewidth=1)
    ax.set_xlabel(r"mole fraction $X_k$")
    ax.set_ylabel(r"free energy $f_0(X)$")
    ax.legend()
    save_all(fig, "potential")
    plt.close(fig)


if __name__ == "__main__":
    plot_gibbs()
    plot_potential()
