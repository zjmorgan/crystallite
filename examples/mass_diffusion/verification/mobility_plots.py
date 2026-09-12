"""Composition-dependent mobility figures: ``mobility.pgf`` and
``interdiffusion.pgf``.

Both use :func:`crystallite.mobility.mobility` directly -- no new solver
code, just plotting an already-tested function
(see ``test_darken_degeneracy_matches_cahn_taylor_peak_location`` in
tests/test_mobility.py, which checks the closed-form peak location used
below).
"""

from __future__ import annotations

import numpy as np

from _plotting import plt, save_all
from crystallite.mobility import mobility

X = np.linspace(1e-3, 1 - 1e-3, 500)


def peak_location(zeta):
    """Closed-form composition maximizing M = X(1-X)(X M_j + (1-X) M_k)."""
    if zeta == 1:
        return 0.5
    return (
        (4 * zeta - 2) - np.sqrt((2 - 4 * zeta) ** 2 - 4 * zeta * (3 * zeta - 3))
    ) / (2 * (3 * zeta - 3))


def plot_mobility_vs_zeta():
    """Composition-dependent mobility for various zeta = M_k / M_j."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    for m_j, m_k in [(1.0, 0.2), (1.0, 1.0), (1.0, 3.0), (1.0, 8.0)]:
        zeta = m_k / m_j
        # Mobility is already reported in units of m_j, so M/m_j is the
        # dimensionless quantity plotted -- with m_j = 1 here the values are
        # unchanged, but the axis label reflects the actual scale used.
        values = mobility(X, m_k, m_j, temperature=1.0, degeneracy=True) / m_j
        (line,) = ax.plot(X, values, label=rf"$\zeta = {zeta:g}$")
        x_peak = peak_location(zeta)
        y_peak = mobility(x_peak, m_k, m_j, temperature=1.0, degeneracy=True) / m_j
        ax.plot(x_peak, y_peak, "o", color=line.get_color(), markersize=4)
    ax.set_xlabel(r"$X$")
    ax.set_ylabel(r"$M / M_j$")
    ax.legend()
    save_all(fig, "mobility")
    plt.close(fig)


def plot_interdiffusion_mechanisms():
    """Unequal-diffusivity mobility, showing the volume-dominated (linear,
    dilute-limit) and surface-dominated (parabolic, mid-composition)
    regimes within the same curve.
    """
    d_a, d_b = 1.0, 8.0

    # Nondimensionalize mobility by d_a, the smaller of the two end-member
    # diffusivities, so the plotted curve is independent of the absolute
    # diffusivity scale and only depends on the ratio d_b / d_a.
    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    full = mobility(X, d_a, d_b, temperature=1.0, degeneracy=True) / d_a
    ax.plot(X, full, color="C0", label="interdiffusion mobility")

    # Dilute-limit ("volume") asymptotes: M ~ (D/RT) X as X -> 0,
    # M ~ (D/RT) (1 - X) as X -> 1.
    dilute_lo = X
    dilute_hi = (1 - X) * (d_b / d_a)
    ax.plot(X, dilute_lo, "--", color="0.5", label="volume limit, " + r"$X\to0$")
    ax.plot(X, dilute_hi, ":", color="0.5", label="volume limit, " + r"$X\to1$")

    ax.set_ylim(0, 1.05 * np.max(full))
    ax.set_xlabel(r"$X$")
    ax.set_ylabel(r"$M / D_a$")
    ax.legend()
    save_all(fig, "interdiffusion")
    plt.close(fig)


if __name__ == "__main__":
    plot_mobility_vs_zeta()
    plot_interdiffusion_mechanisms()
