r"""Triple-junction angle relaxation: a quantitative check of the solver's
equilibrium geometry. Grain crystallography plays no role here (see below),
so this only exercises grain topology/shape, not orientation.

:class:`crystallite.grain_orientation.GrainOrientation` uses one
``cross_coefficient`` shared by every pair of order parameters, so the
boundary energy this model produces is isotropic -- the same for any pair of
grains, regardless of their crystallographic misorientation. For three equal
boundary tensions meeting at a point, force balance requires the three
angles to be equal, so the analytically known equilibrium is the
**120-degree Y-junction**, independent of orientation -- so the grain ids
here are built directly from :func:`crystallite.microstructure.periodic_voronoi`,
with no orientation pool at all.

A periodic Voronoi tessellation is generically 3-valent everywhere (a
higher-order junction needs an exactly-symmetric seed placement, which
generic seeds avoid) -- so this microstructure already has *only* triple
junctions, none of higher order. For 3 grains that means exactly **6** of
them, not 1: Euler's formula on the torus (``V - E + F = 0``) together with
every vertex being 3-valent (``3V = 2E``) gives ``V = 2F``, and ``F = 3``
grains here.

Three grains are seeded at a symmetric Voronoi triangle, so each of those 6
junctions starts with unequal angles (a generic Voronoi corner is not 120
degrees). Starting from that sharp (pixel-jagged) geometry, the RMS
deviation of the measured wedge angles from 120 degrees drops through the
run, averaged over all 6 junctions, without ever losing a grain --
demonstrating the solver's curvature-driven relaxation independently of
:mod:`circular_grain`'s single-shrinking-disk check.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from _plotting import plt, save_all
from crystallite.grain_orientation import GrainOrientation
from crystallite.grid import Grid
from crystallite.microstructure import periodic_voronoi

N = 192
GRADIENT_ENERGY = 1.0e-4
MOBILITY = 1.0
BARRIER_COEFFICIENT = 1.0
QUARTIC_COEFFICIENT = 1.0
CROSS_COEFFICIENT = 1.5
TIME_STEP = 0.5
STEPS = 800
SNAPSHOT_EVERY = 40
JUNCTION_RADIUS = 0.09  # physical units, for the angle measurement circle
SEEDS = np.array([[0.5, 0.65], [0.25, 0.3], [0.75, 0.3]])


def grain_labels(eta):
    """Index of the dominant order parameter at every point, 2D slice."""
    return np.argmax(np.abs(np.asarray(eta)), axis=0)[:, :, 0]


def grain_count(eta):
    return len(np.unique(np.argmax(np.abs(np.asarray(eta)), axis=0)))


def find_junctions(labels, n_grains=3, merge_distance=10.0):
    """Centroid of every connected cluster of pixels whose periodic
    4-neighborhood touches all `n_grains` labels, with nearby centroids
    merged (pixel-level fragmentation of a single corner can otherwise
    split it into several disconnected clusters)."""
    presence = np.zeros((n_grains,) + labels.shape, dtype=bool)
    for k in range(n_grains):
        mask = labels == k
        touching = mask.copy()
        for axis in (0, 1):
            for shift in (1, -1):
                touching |= np.roll(mask, shift, axis=axis)
        presence[k] = touching
    distinct = presence.sum(axis=0) == n_grains
    clustered, n_clusters = ndimage.label(distinct)
    centroids = [
        np.mean(np.nonzero(clustered == c), axis=1) for c in range(1, n_clusters + 1)
    ]
    merged, used = [], [False] * len(centroids)
    for i, c in enumerate(centroids):
        if used[i]:
            continue
        group, used[i] = [c], True
        for j in range(i + 1, len(centroids)):
            if not used[j] and np.hypot(*(centroids[j] - c)) < merge_distance:
                group.append(centroids[j])
                used[j] = True
        merged.append(tuple(np.mean(group, axis=0)))
    return merged


def angular_widths(labels, junction, radius_index, n_samples=1440, n_grains=3):
    """Angular width (degrees) each grain occupies on a circle of
    `radius_index` pixels around `junction` -- the three wedge angles."""
    jx, jy = junction
    n = labels.shape[0]
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    xs = np.round(jx + radius_index * np.cos(theta)).astype(int) % n
    ys = np.round(jy + radius_index * np.sin(theta)).astype(int) % n
    owner = labels[xs, ys]
    return np.array([(owner == k).sum() for k in range(n_grains)]) / n_samples * 360.0


def plot_triple_junction():
    grid = Grid(shape=(N, N, 1), lengths=(1.0, 1.0, 1.0))
    grain_ids, _ = periodic_voronoi(grid, 3, seeds=SEEDS)
    grain_ids = np.asarray(grain_ids)[:, :, 0]

    eta = np.zeros((3, N, N, 1), dtype=np.float32)
    for i in range(3):
        eta[i, :, :, 0] = grain_ids == i

    solver = GrainOrientation(
        grid, mobility=MOBILITY, gradient_energy=GRADIENT_ENERGY,
        barrier_coefficient=BARRIER_COEFFICIENT, quartic_coefficient=QUARTIC_COEFFICIENT,
        cross_coefficient=CROSS_COEFFICIENT,
    )

    radius_index = JUNCTION_RADIUS * N
    steps_recorded, rms_deviation, worst_spread = [0], [], []

    def measure(snap):
        labels = grain_labels(snap)
        junctions = find_junctions(labels)
        widths = np.array([angular_widths(labels, j, radius_index) for j in junctions])
        rms_deviation.append(float(np.sqrt(np.mean((widths - 120.0) ** 2))))
        worst_spread.append(float((widths.max(axis=1) - widths.min(axis=1)).max()))

    measure(eta)
    snapshots = [(0, eta.copy())]
    for step in range(STEPS):
        eta = np.asarray(solver.step(eta, TIME_STEP))
        if (step + 1) % SNAPSHOT_EVERY == 0:
            steps_recorded.append(step + 1)
            measure(eta)
            if (step + 1) % 200 == 0:
                snapshots.append((step + 1, eta.copy()))
    final_grain_count = grain_count(eta)

    fig, axes = plt.subplots(1, len(snapshots), figsize=(3.2 * len(snapshots), 3.4),
                              constrained_layout=True)
    for ax, (step, snap) in zip(axes, snapshots):
        labels = grain_labels(snap)
        ax.imshow(np.swapaxes(labels, 0, 1), origin="lower", cmap="Set2", vmin=0, vmax=2,
                   interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"step {step}, {grain_count(snap)} grains")
    save_all(fig, "grain_orientation.triple_junction_microstructure")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    ax2.plot(steps_recorded, rms_deviation, "o-", color="C0", markersize=4)
    ax2.axhline(0.0, color="0.6", linestyle="--", linewidth=1)
    ax2.set_xlabel("step")
    ax2.set_ylabel(r"RMS deviation from $120^\circ$")
    save_all(fig2, "grain_orientation.triple_junction_convergence")
    plt.close(fig2)

    print(f"final grain count = {final_grain_count}")
    print(f"RMS deviation from 120deg: step 0 = {rms_deviation[0]:.2f}, "
          f"step {steps_recorded[-1]} = {rms_deviation[-1]:.2f}")
    print(f"worst single-corner spread: step 0 = {worst_spread[0]:.1f}, "
          f"step {steps_recorded[-1]} = {worst_spread[-1]:.1f}")


if __name__ == "__main__":
    plot_triple_junction()
