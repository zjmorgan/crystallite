"""Elastic energy of a composition modulation against its direction: what
selects the morphology of a misfit-driven spinodal decomposition.

A composition wave of amplitude :math:`a` along :math:`\\hat n` costs
:math:`\\tfrac14B(\\hat n)a^2` of elastic energy per unit volume (a dilatational
misfit :math:`\\varepsilon^*=\\varepsilon_0(c-c_\\mathrm{ref})\\delta_{ij}` in a
clamped periodic cell), and its elastic chemical potential is
:math:`\\mu_\\mathrm{el}=B(\\hat n)\\,c_k`. Plotted is :math:`B/B_\\mathrm{iso}`
against the in-plane angle :math:`\\theta` of :math:`\\hat n` from a cube axis,
for cubic crystals of Zener ratio :math:`A_\\mathrm{Z}=3, 1, 1/3` with the same
bulk modulus and :math:`c'` (so :math:`B_\\mathrm{iso}` is the isotropic value,
the flat :math:`A_\\mathrm{Z}=1` curve). The modulation prefers the elastically
soft direction, :math:`\\langle100\\rangle` for :math:`A_\\mathrm{Z}>1` and the
in-plane diagonal for :math:`A_\\mathrm{Z}<1`.

Analytic line: Khachaturyan's :math:`B(\\hat n)`
(:func:`crystallite.verification.chemomechanics.khachaturyan_b`), from the
stiffness alone. Markers: the projection of the solver's elastic potential
(:class:`crystallite.chemomechanics.CoherentDiffusion`, one
:class:`crystallite.elastic_deformation.ElasticDeformation` solve per mode) onto
each integer-wavevector mode of a 64x64 periodic cell, unfiltered.
"""

from __future__ import annotations

import math

import numpy as np

from _plotting import analytic_numeric_curve, component_legend, plt, save_all
from crystallite.chemomechanics import CoherentDiffusion
from crystallite.elastic_deformation import ElasticDeformation
from crystallite.grid import Grid
from crystallite.mass_diffusion import MassDiffusion
from crystallite.verification.anisotropic_inclusion import cubic_from_zener
from crystallite.verification.chemomechanics import khachaturyan_b

EPS0 = 0.05
GRID_N = 64
AMPLITUDE = 0.02
MAX_WAVENUMBER = 6
ZENER_RATIOS = ((3.0, r"$A_\mathrm{Z}=3$"), (1.0, r"$A_\mathrm{Z}=1$"), (1.0 / 3.0, r"$A_\mathrm{Z}=1/3$"))


def _mode_directions():
    """Integer wavevectors (m1, m2) with distinct directions in the first quadrant."""
    modes = {}
    for m1 in range(MAX_WAVENUMBER + 1):
        for m2 in range(MAX_WAVENUMBER + 1):
            if (m1, m2) != (0, 0) and math.gcd(m1, m2) == 1:
                modes[(m1, m2)] = math.degrees(math.atan2(m2, m1))
    return modes


def _numeric(zener, modes):
    grid = Grid(shape=(GRID_N, GRID_N, 1))
    stiffness = cubic_from_zener(zener, 1.0, 0.7)
    coupled = CoherentDiffusion(
        MassDiffusion(grid),
        ElasticDeformation(grid, 1.0, 0.7, 1.0, 0.7, stiffness=stiffness),
        EPS0 * np.eye(3), reference_composition=0.5, dealias=False,
    )
    x, y = np.asarray(grid.x[0]), np.asarray(grid.x[1])
    values = []
    for m1, m2 in modes:
        wave = np.cos(2.0 * np.pi * (m1 * x + m2 * y))
        solution = coupled.solve_elastic(0.5 + AMPLITUDE * wave, tol=1.0e-6)
        potential = np.asarray(coupled.elastic_potential(solution))
        values.append(2.0 * np.mean(potential * wave) / AMPLITUDE)
    return np.array(values), stiffness


def plot_elastic_anisotropy():
    modes = _mode_directions()
    order = sorted(modes, key=modes.get)
    angles = np.array([modes[m] for m in order])
    fine = np.linspace(0.0, 90.0, 181)
    reference = khachaturyan_b(cubic_from_zener(1.0, 1.0, 0.7), EPS0 * np.eye(3), [1.0, 0.0, 0.0])

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    for index, (zener, label) in enumerate(ZENER_RATIOS):
        numeric, stiffness = _numeric(zener, order)
        analytic = np.array([
            khachaturyan_b(stiffness, EPS0 * np.eye(3), [math.cos(math.radians(t)), math.sin(math.radians(t)), 0.0])
            for t in fine
        ])
        analytic_numeric_curve(
            ax, fine, analytic / reference, numeric / reference, label, f"C{index}",
            downsample=1, x_numeric=angles, markersize=3,
        )
    ax.set_xlim(0.0, 90.0)
    ax.set_xticks(np.arange(0, 91, 15))
    ax.set_xlabel(r"$\theta\ (^\circ)$")
    ax.set_ylabel(r"$B / B_\mathrm{iso}$")
    ax.set_title("elastic stiffness of a composition modulation")
    component_legend(ax)
    save_all(fig, "chemomechanics.elastic_anisotropy")
    plt.close(fig)


if __name__ == "__main__":
    plot_elastic_anisotropy()
