"""2D coherent spinodal decomposition of a cubic crystal driven by misfit
strain alone, using :class:`crystallite.chemomechanics.CoherentDiffusion`.

The composition field starts as small noise about the unstable midpoint and
decomposes under Cahn-Hilliard diffusion. The two phases share one elastic
stiffness (a homogeneous cubic crystal) and differ only in lattice parameter:
a dilatational misfit strain :math:`\\varepsilon^*=\\varepsilon_0(c-c_\\mathrm{ref})
\\delta_{ij}` in a clamped periodic cell. Nothing else breaks the isotropy of
the diffusion problem, so the morphology is selected by the elastic anisotropy
alone: the modulations align with the elastically soft direction --
:math:`\\langle100\\rangle` (the grid axes) for a Zener ratio
:math:`A_\\mathrm{Z}>1`, the in-plane diagonal for :math:`A_\\mathrm{Z}<1`, and
neither for an isotropic crystal, :math:`A_\\mathrm{Z}=1`
(:mod:`examples.chemomechanics.verification.elastic_anisotropy` checks the
elastic energy against Khachaturyan's :math:`B(\\hat n)`).

The bottom row shows the structure factor of each final field, and its title the
weight of :math:`|c_k|^2` on the axes versus the diagonals,
:func:`crystallite.verification.chemomechanics.axis_diagonal_anisotropy`
(+1 all on the axes, -1 all on the diagonals). One realization scatters by about
:math:`\\pm0.15` around the statistical value, so the isotropic case, whose mean over
seeds is 0 (-0.07 +- 0.05 over eight 64x64 runs), can read a little off zero.

A positive ``STABILIZER`` (the semi-implicit scheme's reference curvature) is
needed for the explicitly treated elastic term at the grid scale; it only
changes the dynamics at :math:`O(\\Delta t)`.

Plots use the project's shared style,
:func:`crystallite.visualization.configure_pgf`.
"""

from __future__ import annotations

import numpy as np

from crystallite.chemomechanics import CoherentDiffusion
from crystallite.elastic_deformation import ElasticDeformation
from crystallite.grid import Grid
from crystallite.mass_diffusion import MassDiffusion
from crystallite.verification.anisotropic_inclusion import cubic_from_zener
from crystallite.verification.chemomechanics import axis_diagonal_anisotropy

MISFIT = 0.05  # eps0
STIFFNESS_SCALE = 25.0  # nondimensional: sets the elastic energy against the barrier
GRADIENT_ENERGY = 2.0e-4
STABILIZER = 2.0
ZENER_RATIOS = (3.0, 1.0, 1.0 / 3.0)


def build_coupled(grid, zener):
    """The coupled solver for a cubic crystal of Zener ratio `zener`."""
    stiffness = STIFFNESS_SCALE * cubic_from_zener(zener, 1.0, 0.7)
    elasticity = ElasticDeformation(grid, 1.0, 0.7, 1.0, 0.7, stiffness=stiffness)
    diffusion = MassDiffusion(
        grid, mobility=1.0, gradient_energy=GRADIENT_ENERGY, left_well=0.0, right_well=1.0,
        scheme="semi_implicit", reference_curvature=STABILIZER, dealias=True,
    )
    return CoherentDiffusion(
        diffusion, elasticity, MISFIT * np.eye(3), reference_composition=0.5
    )


def run_case(zener, shape=(128, 128, 1), steps=600, time_step=2.0e-4, seed=0):
    """Decompose from noise; returns the final composition field."""
    grid = Grid(shape=shape)
    coupled = build_coupled(grid, zener)
    rng = np.random.default_rng(seed)
    composition = (0.5 + 0.01 * rng.standard_normal(grid.shape)).astype(np.float32)
    displacement = None
    for _ in range(steps):
        composition, solution = coupled.step(
            composition, time_step, displacement=displacement, tol=1.0e-5, max_iterations=200
        )
        displacement = solution.displacement
    return np.asarray(composition)


def run_example(shape=(128, 128, 1), steps=600, save_path=None, show_plot=False):
    """Run all three Zener ratios.

    Returns
    -------
    fields : dict
        Zener ratio -> final composition field.
    anisotropy : dict
        Zener ratio -> axis-versus-diagonal structure-factor weight.
    """
    fields = {zener: run_case(zener, shape=shape, steps=steps) for zener in ZENER_RATIOS}
    anisotropy = {zener: axis_diagonal_anisotropy(field) for zener, field in fields.items()}

    if save_path is not None or show_plot:
        try:
            from crystallite.visualization import configure_pgf

            plt = configure_pgf()
        except ImportError:  # pragma: no cover
            print("Matplotlib is not installed; skipping plot output for this run.")
            return fields, anisotropy

        fig, axes = plt.subplots(2, 3, figsize=(9.0, 6.0), constrained_layout=True)
        for column, zener in enumerate(ZENER_RATIOS):
            field = fields[zener][:, :, 0]
            axes[0, column].imshow(field.T, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
            axes[0, column].set_title(rf"$A_\mathrm{{Z}}={zener:.3g}$")
            axes[0, column].set_xticks([])
            axes[0, column].set_yticks([])
            structure = np.abs(np.fft.fftshift(np.fft.fft2(field - field.mean()))) ** 2
            half = field.shape[0] // 4
            centre = field.shape[0] // 2
            axes[1, column].imshow(
                np.log10(structure[centre - half:centre + half, centre - half:centre + half].T + 1.0),
                origin="lower", cmap="magma",
            )
            axes[1, column].set_title(rf"axes $-$ diagonals: ${anisotropy[zener]:+.2f}$")
            axes[1, column].set_xticks([])
            axes[1, column].set_yticks([])
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved plot to {save_path}")
        if show_plot:
            plt.show()
        plt.close(fig)
    return fields, anisotropy


if __name__ == "__main__":
    _, values = run_example(save_path="coherent_spinodal.png")
    for zener, value in values.items():
        print(f"A_Z = {zener:5.2f}: axes - diagonals = {value:+.3f}")
