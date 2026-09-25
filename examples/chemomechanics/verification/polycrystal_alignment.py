"""Each grain of a misfit-driven polycrystal decomposes along its own soft
direction.

A periodic Voronoi polycrystal of cubic crystals, each grain turned by a random
in-plane angle :math:`\\theta_g`, decomposes from noise driven by a dilatational
misfit strain alone (the two phases share one stiffness,
:meth:`crystallite.chemomechanics.CoherentDiffusion.polycrystal`). The
modulation of grain :math:`g` lies along its elastically soft direction, the
crystal axes for a Zener ratio :math:`A_\\mathrm{Z}>1` (interfaces at
:math:`\\theta_g` modulo 90 degrees) and the crystal diagonals for
:math:`A_\\mathrm{Z}<1` (:math:`\\theta_g+45^\\circ`); this is the prediction of
:func:`crystallite.verification.chemomechanics.khachaturyan_b` (the energy
minimum of a modulation, checked against the solver in
``elastic_anisotropy.py``) and needs no other input than the grain's orientation.

Plotted against the predicted interface angle is the measured minus predicted
angle, per grain, from the four-fold order parameter of the interfaces in the
grain's interior
(:func:`crystallite.verification.chemomechanics.interface_orientation`): the
analytic line is zero, the markers the measured errors. Only grains with a
nontrivial interior (at least 400 points, 6 points in from the boundary) and a
clear modulation (strength above 0.25) are shown, as the alignment of a grain
smaller than a wavelength or two is not defined; the number is printed.

Numeric results are cached in ``figures/_numeric_cache``.
"""

from __future__ import annotations

import numpy as np

from _plotting import analytic_numeric_curve, cached_numeric, component_legend, plt, save_all
from crystallite.chemomechanics import CoherentDiffusion  # noqa: F401  (documented above)
from crystallite.grid import Grid
from crystallite.verification.chemomechanics import interface_orientation

import importlib.util
import os

_TEMPLATE = os.path.join(os.path.dirname(__file__), os.pardir, "polycrystal.py")
_spec = importlib.util.spec_from_file_location("chemomechanics_polycrystal", _TEMPLATE)
polycrystal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(polycrystal)

GRID_N = 128
N_GRAINS = 8
STEPS = 500
MARGIN = 6
MIN_PIXELS = 400
MIN_STRENGTH = 0.25
CASES = ((3.0, 0.0, r"$A_\mathrm{Z}=3$"), (1.0 / 3.0, 45.0, r"$A_\mathrm{Z}=1/3$"))


def _wrap(angle):
    """Wrap an angle difference to (-45, 45] degrees (the cubic symmetry)."""
    return (angle + 45.0) % 90.0 - 45.0


def _measure(zener, shift):
    grid = Grid(shape=(GRID_N, GRID_N, 1))
    microstructure, coupled = polycrystal.build_case(grid, n_grains=N_GRAINS, zener=zener)
    composition = polycrystal.decompose(coupled, steps=STEPS)[:, :, 0]
    rows = []
    for grain in range(microstructure.n_grains):
        rotation = microstructure.orientations[grain]
        crystal = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])) % 90.0
        mask = microstructure.grain_interior(grain, margin=MARGIN)[:, :, 0]
        if mask.sum() < MIN_PIXELS:
            continue
        angle, strength = interface_orientation(composition, weight=mask)
        predicted = (crystal + shift) % 90.0
        rows.append((predicted, _wrap(angle - predicted), strength, int(mask.sum())))
    return np.array(rows)


def plot_polycrystal_alignment():
    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    for index, (zener, shift, label) in enumerate(CASES):
        rows = cached_numeric(
            f"polycrystal_alignment_A{zener:.3f}_N{GRID_N}_g{N_GRAINS}_s{STEPS}_m{MARGIN}",
            lambda z=zener, s=shift: _measure(z, s),
        )
        shown = rows[rows[:, 2] > MIN_STRENGTH]
        print(f"{label}: {len(shown)} of {len(rows)} grains shown, "
              f"max |error| = {np.abs(shown[:, 1]).max():.1f} deg")
        order = np.argsort(shown[:, 0])
        analytic_numeric_curve(
            ax, np.array([0.0, 90.0]), np.array([0.0, 0.0]), shown[order, 1], label, f"C{index}",
            downsample=1, x_numeric=shown[order, 0], markersize=6,
        )
    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(-15.0, 15.0)
    ax.set_xticks(np.arange(0, 91, 15))
    ax.set_xlabel(r"predicted interface angle $(^\circ)$")
    ax.set_ylabel(r"measured $-$ predicted $(^\circ)$")
    ax.set_title("misfit-driven polycrystal: interface orientation per grain")
    component_legend(ax)
    save_all(fig, "chemomechanics.polycrystal_alignment")
    plt.close(fig)


if __name__ == "__main__":
    plot_polycrystal_alignment()
