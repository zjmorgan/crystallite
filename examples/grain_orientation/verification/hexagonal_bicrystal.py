r"""A 2-grain periodic structure that starts *exactly* at the equilibrium
120-degree honeycomb geometry, and stays there.

:mod:`triple_junction` verifies the solver's curvature-driven relaxation
*toward* the 120-degree equal-tension equilibrium, from an arbitrary
(non-equilibrium) Voronoi corner. This is the complementary check: start
already at equilibrium and confirm the solver leaves it alone.

A triangular point lattice's Voronoi tessellation is a regular honeycomb --
every vertex exactly 120 degrees. Its index-2 rectangular sublattice (the
standard "centered rectangular" cell: one seed at a corner, one at the
center, in a box of aspect ratio :math:`\sqrt3:1`) is the *only* way to
reproduce that lattice exactly with a plain axis-aligned periodic domain and
the fewest grains -- :class:`crystallite.grid.Grid` has no sheared/oblique
periodicity, so this is as close to the true honeycomb as a rectangular grid
can get. (Odd grain counts, e.g. 3, are impossible this way: a rectangular
sublattice of a triangular lattice always has even index -- checked
exhaustively over a wide range of candidate sublattice vectors, and provably
true since the primitive rectangular cell always contains 2 points and any
rectangular sublattice is an integer multiple of it.)

Because there are only 2 labels, this construction can't be checked by
measuring wedge angles the way :mod:`triple_junction` does: at each true
120-degree vertex, 2 of the 3 meeting wedges necessarily share a label
(pigeonhole), and since those two are adjacent going around the vertex, they
merge into one contiguous arc -- a genuine triple point becomes
indistinguishable from a smooth 2-grain boundary in the label field alone,
at any sampling radius. So instead of angles, this checks the free energy: if the construction
really is already at equilibrium, ``GrainOrientation`` should show no
driving force at all. In practice the energy still drops once, sharply, in
the first few recorded steps -- that is the sharp (step-function) initial
condition relaxing to its proper diffuse equilibrium profile, a one-time
transient unrelated to the microstructure's shape (any sharp initial
condition has it, including :mod:`triple_junction`'s). What confirms the
shape itself is already at equilibrium is what happens *after* that: the
energy should go perfectly flat, unlike :mod:`triple_junction`'s continued
decrease from its genuinely non-equilibrium start.
"""

from __future__ import annotations

import numpy as np

from _plotting import plt, save_all
from crystallite.grain_orientation import GrainOrientation
from crystallite.grid import Grid
from crystallite.microstructure import periodic_voronoi

LY = 1.0
LX = np.sqrt(3.0) * LY
NY = 200
NX = int(round(NY * LX / LY))
GRADIENT_ENERGY = 1.0e-4
MOBILITY = 1.0
BARRIER_COEFFICIENT = 1.0
QUARTIC_COEFFICIENT = 1.0
CROSS_COEFFICIENT = 1.5
TIME_STEP = 0.5
STEPS = 800
SNAPSHOT_EVERY = 40
# The centered-rectangular basis of a triangular lattice: corner + center.
SEEDS = np.array([[0.0, 0.0], [LX / 2.0, LY / 2.0]])


def grain_labels(eta):
    return np.argmax(np.abs(np.asarray(eta)), axis=0)[:, :, 0]


def grain_count(eta):
    return len(np.unique(np.argmax(np.abs(np.asarray(eta)), axis=0)))


def plot_hexagonal_bicrystal():
    grid = Grid(shape=(NX, NY, 1), lengths=(LX, LY, 1.0))
    grain_ids, _ = periodic_voronoi(grid, 2, seeds=SEEDS)
    grain_ids = np.asarray(grain_ids)[:, :, 0]

    eta = np.zeros((2, NX, NY, 1), dtype=np.float32)
    for i in range(2):
        eta[i, :, :, 0] = grain_ids == i

    solver = GrainOrientation(
        grid, mobility=MOBILITY, gradient_energy=GRADIENT_ENERGY,
        barrier_coefficient=BARRIER_COEFFICIENT, quartic_coefficient=QUARTIC_COEFFICIENT,
        cross_coefficient=CROSS_COEFFICIENT,
    )

    def grain0_area(snap):
        labels = grain_labels(snap)
        return float(np.mean(labels == 0))

    steps_recorded = [0]
    energy = [float(solver.free_energy(eta))]
    area = [grain0_area(eta)]
    snapshots = [(0, eta.copy())]
    for step in range(STEPS):
        eta = np.asarray(solver.step(eta, TIME_STEP))
        if (step + 1) % SNAPSHOT_EVERY == 0:
            steps_recorded.append(step + 1)
            energy.append(float(solver.free_energy(eta)))
            area.append(grain0_area(eta))
            if (step + 1) % 200 == 0:
                snapshots.append((step + 1, eta.copy()))
    final_grain_count = grain_count(eta)

    fig, axes = plt.subplots(1, len(snapshots), figsize=(3.2 * len(snapshots), 3.4 / (LX / LY)),
                              constrained_layout=True)
    for ax, (step, snap) in zip(axes, snapshots):
        labels = grain_labels(snap)
        ax.imshow(np.swapaxes(labels, 0, 1), origin="lower", cmap="Set2", vmin=0, vmax=1,
                   interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"step {step}, {grain_count(snap)} grains")
    save_all(fig, "grain_orientation.hexagonal_bicrystal_microstructure")
    plt.close(fig)

    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(8.5, 3.5), constrained_layout=True)
    ax2.plot(steps_recorded, energy, "o-", color="C0", markersize=4)
    ax2.set_xlabel("step")
    ax2.set_ylabel("free energy density")
    ax3.plot(steps_recorded, area, "o-", color="C1", markersize=4)
    ax3.set_xlabel("step")
    ax3.set_ylabel("grain-0 area fraction")
    save_all(fig2, "grain_orientation.hexagonal_bicrystal_energy")
    plt.close(fig2)

    # The very first recorded steps include a one-time transient (the sharp
    # step-function initial condition relaxing to its proper diffuse
    # equilibrium profile) that is unrelated to the microstructure's shape;
    # excluding it isolates the actual shape-equilibrium check. The corners
    # do visibly round after that (a real, local effect -- a true stationary
    # diffuse triple junction curves slightly right at the vertex, unlike
    # this construction's perfectly sharp corner), but area is the cleaner
    # check for *net* boundary migration, since local corner rounding on its
    # own does not change how much of the domain each grain occupies.
    after_transient = energy[1:]
    shape_drift = (max(after_transient) - min(after_transient)) / abs(after_transient[0])
    area_drift = (max(area) - min(area)) / area[0]
    print(f"final grain count = {final_grain_count}")
    print(f"free energy: step 0 (sharp init) = {energy[0]:.6g}, "
          f"step {steps_recorded[1]} (profile relaxed) = {energy[1]:.6g}, "
          f"step {steps_recorded[-1]} = {energy[-1]:.6g}")
    print(f"shape-equilibrium drift (step {steps_recorded[1]} onward) = {shape_drift:.3e}")
    print(f"grain-0 area fraction: step 0 = {area[0]:.6g}, step {steps_recorded[-1]} = {area[-1]:.6g}, "
          f"relative drift = {area_drift:.3e}")


if __name__ == "__main__":
    plot_hexagonal_bicrystal()
