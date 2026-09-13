"""2D Ostwald ripening (grain growth) example using the multi-order-parameter
Allen-Cahn ``GrainOrientation`` solver.

``N_ORIENTATIONS`` order-parameter fields are seeded with small random noise
about 0, the same idea as :mod:`examples.mass_diffusion.template`'s
spinodal decomposition. The cross-coupling term makes whichever order
parameter is locally largest at a point grow and suppress the others
there, so a polycrystalline structure self-organizes directly from noise --
curvature-driven grain-boundary migration then coarsens it: small grains
shrink and vanish, larger ones grow, and the total free energy decreases
monotonically (Allen-Cahn's defining Lyapunov property). Left to run long
enough, coarsening runs all the way down to a single surviving grain --
unlike, say, a Potts/vertex model, plain multi-order-parameter Allen-Cahn
grain growth has nothing that stabilizes multiple grains indefinitely.

Getting a polycrystal to nucleate from noise at all is more parameter
sensitive here than for spinodal decomposition. Allen-Cahn (non-conserved)
dynamics have no finite-wavelength selection the way Cahn-Hilliard's
conserved dynamics do: the linearized growth rate ``L*(a - kappa*k^2)``
*decreases monotonically* with ``k``, rather than peaking at some nonzero
``k`` the way Cahn-Hilliard's does. That does not mean the spatially
uniform (``k=0``) mode necessarily wins -- it means a whole band of
long-wavelength modes grows at comparable rates, and which one ends up
dominating a given region is essentially set by the noise realization
there. If that band's characteristic wavelength,
``2*pi*sqrt(gradient_energy / barrier_coefficient)``, is too large a
fraction of the domain, there is only room for a handful of independent
"cells" and one orientation's random head start tends to sweep the whole
domain before local competition elsewhere can develop (confirmed
empirically: ``gradient_energy=1e-3`` at these other parameters collapses
to one grain almost immediately). ``GRADIENT_ENERGY`` below is chosen
small enough, relative to the domain, for many independent grains to
nucleate, while ``shape`` is fine enough for the resulting interfaces
(:func:`crystallite.mass_diffusion.double_well_equilibrium_width`-scale)
to still land on several grid points rather than being pinned to the
lattice.

Plots use the project's shared style,
:func:`crystallite.visualization.configure_pgf`.
"""

from __future__ import annotations

import numpy as np

from crystallite.grain_orientation import GrainOrientation
from crystallite.grid import Grid

N_ORIENTATIONS = 20
BARRIER_COEFFICIENT = 1.0
QUARTIC_COEFFICIENT = 1.0
CROSS_COEFFICIENT = 1.5
GRADIENT_ENERGY = 1.0e-4
MOBILITY = 1.0


def build_solver(grid):
    """Construct the isotropic ``GrainOrientation`` solver used by this
    example."""
    return GrainOrientation(
        grid,
        mobility=MOBILITY,
        gradient_energy=GRADIENT_ENERGY,
        barrier_coefficient=BARRIER_COEFFICIENT,
        quartic_coefficient=QUARTIC_COEFFICIENT,
        cross_coefficient=CROSS_COEFFICIENT,
    )


def initial_field(grid, n_orientations=N_ORIENTATIONS, amplitude=0.1, seed=0):
    """Return small-amplitude noise about 0 for every orientation field."""
    rng = np.random.default_rng(seed)
    return amplitude * rng.standard_normal((n_orientations,) + grid.shape)


def grain_count(eta):
    """Return the number of orientations that are dominant (largest in
    magnitude) at at least one grid point -- a simple proxy for the number
    of surviving grains.

    Each order parameter's double well has minima at both ``+eta0`` and
    ``-eta0``, so a saturated grain is not always positive -- comparing
    magnitude (not signed value) is what correctly identifies which
    orientation has committed at a point.
    """
    return len(np.unique(np.argmax(np.abs(np.asarray(eta)), axis=0)))


def run_example(
    shape=(192, 192, 1),
    lengths=(1.0, 1.0, 1.0),
    time_step=0.5,
    steps=3000,
    snapshot_every=600,
    save_path=None,
    show_plot=False,
):
    """Advance a 2D Ostwald ripening (grain growth) run.

    Parameters
    ----------
    shape : tuple of int, default=(192, 192, 1)
        Real-space grid shape.
    lengths : tuple of float, default=(1.0, 1.0, 1.0)
        Domain length in each coordinate direction.
    time_step : float, default=0.5
        Semi-implicit integration time step.
    steps : int, default=4000
        Number of time steps to take.
    snapshot_every : int, default=800
        Save a snapshot every so many steps.
    save_path : str or None, default=None
        Optional output file for a snapshot-grid PNG image.
    show_plot : bool, default=False
        If True, display the snapshots with Matplotlib.

    Returns
    -------
    eta : ndarray
        Final orientation fields, shape ``(N_ORIENTATIONS,) + shape``.
    snapshots : list of (int, ndarray)
        ``(step, eta)`` pairs recorded during the run.
    energy_history : list of (int, float)
        ``(step, free_energy)`` pairs recorded during the run.
    """
    grid = Grid(shape=shape, lengths=lengths)
    solver = build_solver(grid)
    eta = initial_field(grid)

    snapshots = [(0, np.asarray(eta).copy())]
    energy_history = [(0, float(solver.free_energy(eta)))]
    for step in range(steps):
        eta = solver.step(eta, time_step)
        if (step + 1) % snapshot_every == 0 or step == steps - 1:
            snapshots.append((step + 1, np.asarray(eta).copy()))
            energy_history.append((step + 1, float(solver.free_energy(eta))))

    if save_path is not None or show_plot:
        try:
            from crystallite.visualization import configure_pgf

            plt = configure_pgf()
        except ImportError:  # pragma: no cover
            print(
                "Matplotlib is not installed; skipping plot output for this run."
            )
            return eta, snapshots, energy_history

        fig, axes = plt.subplots(
            1, len(snapshots), figsize=(3.0 * len(snapshots), 3.2),
            constrained_layout=True,
        )
        for ax, (step, snap) in zip(axes, snapshots):
            grain_map = np.argmax(np.abs(snap), axis=0)[:, :, 0]
            ax.imshow(
                grain_map, origin="lower", cmap="viridis",
                vmin=0, vmax=N_ORIENTATIONS - 1, interpolation="nearest",
            )
            ax.set_title(rf"step {step}, {grain_count(snap)} grains")
            ax.set_xticks([])
            ax.set_yticks([])

        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved plot to {save_path}")

        if show_plot:
            plt.show()

        plt.close(fig)

    return eta, snapshots, energy_history


if __name__ == "__main__":
    eta, snapshots, energy_history = run_example(
        shape=(192, 192, 1),
        steps=3000,
        snapshot_every=600,
        save_path="ostwald_ripening.png",
        show_plot=False,
    )
    print(f"final grain count = {grain_count(eta)}")
    for step, energy in energy_history:
        print(f"step {step}: free_energy = {energy:.6f}")
