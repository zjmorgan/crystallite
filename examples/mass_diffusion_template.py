"""Simple 2D isotropic spinodal decomposition example.

This example uses the nonlinear ``MassDiffusion`` solver in a minimal
spinodal decomposition setup. The composition field is initialized with a
small random perturbation about the unstable midpoint and then evolves under
an isotropic mobility and isotropic gradient-energy tensor.
"""

from __future__ import annotations

import numpy as np

from crystallite.diffusion import MassDiffusion
from crystallite.phase import Grid


def initial_field(grid, amplitude=0.015):
    """Return a small-amplitude noisy field around the unstable state."""
    rng = np.random.default_rng(42)
    noise = amplitude * rng.standard_normal(grid.shape)
    return 0.5 + noise


def run_example(
    shape=(64, 64, 1),
    lengths=(1.0, 1.0, 1.0),
    time_step=1.0e-7,
    steps=600,
    snapshot_every=50,
    save_path=None,
    show_plot=False,
):
    """Advance a 2D isotropic spinodal decomposition run.

    Parameters
    ----------
    shape : tuple of int, default=(128, 128, 1)
        Real-space grid shape.
    lengths : tuple of float, default=(1.0, 1.0, 1.0)
        Domain length in each coordinate direction.
    time_step : float, default=5.0e-6
        Explicit integration time step.
    steps : int, default=200
        Number of time steps to take.
    snapshot_every : int, default=20
        Save a snapshot every so many steps.
    save_path : str or None, default=None
        Optional output file for a final PNG image.
    show_plot : bool, default=False
        If True, display the final field with Matplotlib.

    Returns
    -------
    field : ndarray
        Final composition field.
    snapshots : list of ndarray
        Stored composition snapshots for later plotting.
    """
    grid = Grid(shape=shape, lengths=lengths)
    field = initial_field(grid)
    solver = MassDiffusion(
        grid=grid,
        mobility=1.0,
        gradient_energy=1.5e-4,
        # Standard double-well bulk free energy: f(c) = 1/4(c^2 - 1)^2,
        # so f'(c) = c^3 - c.
        free_energy_derivative=lambda c: c**3 - c,
    )

    snapshots = []
    for step in range(steps):
        field = solver.step(field, time_step)
        if (step + 1) % snapshot_every == 0 or step == steps - 1:
            snapshots.append(field.copy())

    if save_path is not None or show_plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover
            print(
                "Matplotlib is not installed; skipping plot output for this run."
            )
            return field, snapshots

        fig, ax = plt.subplots(figsize=(4.5, 4.5), constrained_layout=True)
        image = ax.imshow(
            field[:, :, 0],
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
        )
        ax.set_title("Spinodal decomposition")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, label="composition")

        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved plot to {save_path}")

        if show_plot:
            plt.show()

        plt.close(fig)

    return field, snapshots


if __name__ == "__main__":
    field, snapshots = run_example(
        shape=(64, 64, 1),
        time_step=1.0e-7,
        steps=600,
        snapshot_every=50,
        save_path="spinodal_decomposition.png",
        show_plot=False,
    )
    print(f"final mean composition = {np.mean(field):.6f}")
    print(f"stored {len(snapshots)} snapshots")
