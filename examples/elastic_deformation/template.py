"""2D stress concentration around a soft circular hole under remote
uniaxial tension, using the heterogeneous-elasticity
:class:`crystallite.elastic_deformation.ElasticDeformation` solver.

A circular region much softer than the surrounding matrix (a
:class:`crystallite.verification.HoleInPlateCase`, with a diffuse rather
than sharp boundary -- the same "diffuse, not sharp" treatment used
everywhere else in this project) approximates a traction-free hole. Under
remote tension, the matrix around the hole shows the classic stress
concentration this solver is meant to capture: no eigenstrain, no
phase-field coupling, just Hooke's law for a spatially varying stiffness --
see :mod:`examples.elastic_deformation.verification.hole_in_plate` for a
quantitative comparison against the closed-form Kirsch solution.

Plots use the project's shared style,
:func:`crystallite.visualization.configure_pgf`.
"""

from __future__ import annotations

import numpy as np

from crystallite.grid import Grid
from crystallite.verification import HoleInPlateCase

MATRIX_LAME_LAMBDA = 1.0
MATRIX_LAME_MU = 0.7
HOLE_RADIUS = 0.1
CONTRAST = 1.0e-3
MAGNITUDE = 0.01


def build_case(grid):
    """Construct the soft-hole case used by this example."""
    smoothing_width = 2.0 * grid.spacing[0]
    return HoleInPlateCase(
        grid, matrix_lame_lambda=MATRIX_LAME_LAMBDA, matrix_lame_mu=MATRIX_LAME_MU,
        hole_radius=HOLE_RADIUS, contrast=CONTRAST, center=(0.5, 0.5),
        smoothing_width=smoothing_width,
    )


def run_example(shape=(256, 256, 1), save_path=None, show_plot=False):
    """Solve for the stress field around a hole under remote tension.

    Parameters
    ----------
    shape : tuple of int, default=(256, 256, 1)
        Real-space grid shape.
    save_path : str or None, default=None
        Optional output file for a field-map PNG image.
    show_plot : bool, default=False
        If True, display the plot with Matplotlib.

    Returns
    -------
    solution : ElasticSolution
    case : HoleInPlateCase
    """
    grid = Grid(shape=shape, lengths=(1.0, 1.0, 1.0))
    case = build_case(grid)
    solver = case.solver()
    eps_bar = case.macro_strain("tension", MAGNITUDE, solver=solver)
    solution = solver.solve(eps_bar, tol=1.0e-6, max_iterations=3000)
    print(
        f"converged={solution.converged}, iterations={solution.iterations}, "
        f"residual={solution.residual_norm:.3g}"
    )

    if save_path is not None or show_plot:
        try:
            from crystallite.visualization import configure_pgf

            plt = configure_pgf()
        except ImportError:  # pragma: no cover
            print("Matplotlib is not installed; skipping plot output for this run.")
            return solution, case

        sigma = np.asarray(solution.stress)
        sigma_xx, sigma_yy, sigma_xy = sigma[0, 0], sigma[1, 1], sigma[0, 1]
        von_mises = np.sqrt(
            sigma_xx**2 - sigma_xx * sigma_yy + sigma_yy**2 + 3.0 * sigma_xy**2
        )

        x = np.asarray(grid.x[0])[:, 0, 0]
        y = np.asarray(grid.x[1])[0, :, 0]

        fig, ax = plt.subplots(figsize=(4.5, 4.0), constrained_layout=True)
        im = ax.imshow(
            (von_mises[:, :, 0] / MAGNITUDE).T, origin="lower",
            extent=(x[0], x[-1], y[0], y[-1]), cmap="viridis",
        )
        circle = plt.Circle(case.center, HOLE_RADIUS, fill=False, color="w", linewidth=1)
        ax.add_patch(circle)
        ax.set_xlim(0.25, 0.75)
        ax.set_ylim(0.25, 0.75)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        fig.colorbar(im, ax=ax, label=r"von Mises $\sigma / \sigma_\infty$")

        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved plot to {save_path}")
        if show_plot:
            plt.show()
        plt.close(fig)

    return solution, case


if __name__ == "__main__":
    run_example(save_path="hole_tension_von_mises.png", show_plot=False)
