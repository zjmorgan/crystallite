"""2D coherent spinodal decomposition of a cubic *polycrystal*, driven by misfit
strain alone, using :meth:`crystallite.chemomechanics.CoherentDiffusion.polycrystal`.

The grain structure is a periodic Voronoi tessellation
(:class:`crystallite.microstructure.Microstructure`) with a random in-plane
orientation of the cubic crystal in every grain. All grains, and both phases,
share one crystal-frame stiffness; the composition-dependent misfit strain is
the only driver. In each grain the decomposition follows *that grain's* elastically
soft direction, so the modulations are aligned with the crystal axes
(:math:`\\langle100\\rangle`, Zener ratio :math:`A_\\mathrm{Z}>1`) of every grain
separately and change direction across the boundaries.

Left: the crystal orientation of each grain (modulo the cubic 90 degrees), with
its crystal axes drawn. Right: the final composition, with the grain boundaries
(white).

The stiffness varies from grain to grain, so each step is a conjugate-gradient
elastic solve (referenced to the grains' Voigt average, warm-started from the
previous step) rather than the homogeneous solver's single iteration.

Plots use the project's shared style,
:func:`crystallite.visualization.configure_pgf`.
"""

from __future__ import annotations

import numpy as np

from crystallite.chemomechanics import CoherentDiffusion
from crystallite.grid import Grid
from crystallite.mass_diffusion import MassDiffusion
from crystallite.microstructure import Microstructure, rotation_from_bunge_euler
from crystallite.verification.anisotropic_inclusion import cubic_from_zener

ZENER_RATIO = 3.0
MISFIT = 0.05
STIFFNESS_SCALE = 25.0  # nondimensional: sets the elastic energy against the barrier
GRADIENT_ENERGY = 2.0e-4
STABILIZER = 2.0  # semi-implicit reference curvature, for the explicitly treated elastic term


def build_case(grid, n_grains=8, seed=3, zener=ZENER_RATIO):
    """A random Voronoi polycrystal and its coupled solver.

    Returns
    -------
    microstructure : Microstructure
    coupled : CoherentDiffusion
    """
    angles = np.random.default_rng(seed).uniform(0.0, 90.0, n_grains)
    microstructure = Microstructure.voronoi(
        grid, n_grains, orientations=rotation_from_bunge_euler(angles, 0.0, 0.0), seed=seed
    )
    diffusion = MassDiffusion(
        grid, mobility=1.0, gradient_energy=GRADIENT_ENERGY, left_well=0.0, right_well=1.0,
        scheme="semi_implicit", reference_curvature=STABILIZER, dealias=True,
    )
    coupled = CoherentDiffusion.polycrystal(
        diffusion, microstructure, STIFFNESS_SCALE * cubic_from_zener(zener, 1.0, 0.7),
        MISFIT * np.eye(3), reference_composition=0.5,
    )
    return microstructure, coupled


def decompose(coupled, steps=500, time_step=2.0e-4, seed=0, mean=0.5):
    """Decompose from noise about the mean composition (0.5 is the unstable midpoint);
    returns the final field."""
    grid = coupled.grid
    rng = np.random.default_rng(seed)
    composition = (mean + 0.01 * rng.standard_normal(grid.shape)).astype(np.float32)
    displacement = None
    for _ in range(steps):
        composition, solution = coupled.step(
            composition, time_step, displacement=displacement, tol=1.0e-4, max_iterations=300
        )
        displacement = solution.displacement
    return np.asarray(composition)


def run_example(shape=(128, 128, 1), n_grains=8, steps=500, save_path=None, show_plot=False):
    """Decompose a random cubic polycrystal.

    Returns
    -------
    composition : ndarray
    microstructure : Microstructure
    """
    grid = Grid(shape=shape)
    microstructure, coupled = build_case(grid, n_grains=n_grains)
    composition = decompose(coupled, steps=steps)

    if save_path is not None or show_plot:
        try:
            from scipy.ndimage import gaussian_filter

            from crystallite.visualization import configure_pgf

            plt = configure_pgf()
        except ImportError:  # pragma: no cover
            print("Matplotlib is not installed; skipping plot output for this run.")
            return composition, microstructure

        ids = microstructure.grain_ids[:, :, 0]
        rotations = microstructure.orientations
        crystal_angle = np.degrees(np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])) % 90.0
        # smooth boundaries: the 1/2 contour of each grain's blurred (periodic) indicator
        indicators = [
            gaussian_filter((ids == grain).astype(float), sigma=2.0, mode="wrap")
            for grain in range(microstructure.n_grains)
        ]

        fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.5), constrained_layout=True)
        image = axes[0].imshow(
            crystal_angle[ids].T, origin="lower", cmap="twilight", vmin=0.0, vmax=90.0
        )
        axes[1].imshow(composition[:, :, 0].T, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
        x, y = np.indices(ids.shape)
        for grain in range(microstructure.n_grains):
            inside = ids == grain
            cx, cy = x[inside].mean(), y[inside].mean()
            for turn in (0.0, 90.0):
                angle = np.radians(crystal_angle[grain] + turn)
                axes[0].plot(
                    [cx - 9 * np.cos(angle), cx + 9 * np.cos(angle)],
                    [cy - 9 * np.sin(angle), cy + 9 * np.sin(angle)], color="white", lw=1.5,
                )
        for axis in axes:
            for indicator in indicators:
                axis.contour(indicator.T, levels=[0.5], colors="0.5", linewidths=1.2)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[0].set_title("crystal orientation")
        axes[1].set_title(rf"composition, $A_\mathrm{{Z}}={ZENER_RATIO:g}$")
        fig.colorbar(image, ax=axes[0], label=r"crystal angle $(^\circ)$", shrink=0.8)
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved plot to {save_path}")
        if show_plot:
            plt.show()
        plt.close(fig)
    return composition, microstructure


if __name__ == "__main__":
    run_example(save_path="coherent_polycrystal.png")
