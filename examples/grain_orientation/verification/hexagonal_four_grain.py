r"""A 4-grain periodic honeycomb structure that starts *exactly* at the
120-degree equilibrium, with every triple junction directly measurable.

:mod:`hexagonal_bicrystal` builds the same true honeycomb lattice with only
2 grains, but 2 labels can't properly color a honeycomb: every hexagon has 6
neighbors, and the honeycomb's dual (a triangular lattice) has odd cycles
(any 3 mutually-adjacent lattice points), so its chromatic number is 3, not
2. With only 2 labels, each true vertex necessarily has 2 of its 3 meeting
wedges sharing a label (pigeonhole) -- indistinguishable from a smooth
boundary in the label field, so :mod:`hexagonal_bicrystal` can only be
checked via free energy and area, not by directly measuring angles.

Naively picking one label per basis point of a small rectangular sublattice
does *not* fix this (a real trap that took two wrong attempts to get past).
First trap: a rectangular sublattice's own periodicity vector can itself be
one of the lattice's 6 nearest-neighbor bonds, forcing a label to be its own
periodic-image neighbor regardless of how the basis points are labeled.
Second trap: coloring by ``(i - j) mod 3`` on lattice point ``i*a1 + j*a2``
*is* a mathematically proper 3-coloring of the infinite lattice (no two
nearest-neighbor points ever share a color) -- but only if the rectangular
cell's own periodicity vectors also preserve that coloring, i.e. each
satisfies ``(delta_i - delta_j) % 3 == 0``. A rectangular cell chosen purely
for the right *area* (as in the first attempt) can easily fail this and
silently reintroduce the same-label-touches-itself bug, since replicating a
basis point's fixed label along a period that doesn't preserve the coloring
just relabels some other, differently-colored lattice point with the wrong
color. Requiring both constraints together, the smallest valid rectangular
cell is index 6 (``Lx=3``, ``Ly=sqrt(3)``), with colors ``[0,1,2,0,1,2]``
across its 6 basis points. To get 4 *visually distinct* grains rather than
3, one of the three proper colors (the two basis points that land on it) is
split into two labels -- since those two positions were never adjacent
under the valid 3-coloring to begin with, splitting them can only add
distinctions, never break the proper-coloring property (checked directly
with :func:`crystallite.microstructure.grain_neighbors`: no grain neighbors
itself).

Because every vertex now has 3 genuinely different labels, this can be
checked exactly like :mod:`triple_junction`'s convergence test -- except
here the RMS deviation from 120 degrees should already be at its numerical
floor at step 0 (up to pixel discretization), and stay there, rather than
decrease from a large initial value.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from _plotting import plt, save_all
from crystallite.grain_orientation import GrainOrientation
from crystallite.grid import Grid
from crystallite.microstructure import grain_neighbors, periodic_voronoi

LX = 3.0
LY = np.sqrt(3.0)
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
JUNCTION_RADIUS = 0.2  # physical units
# 6 positions of the index-6 (i-j)%3 proper 3-coloring (colors [0,1,2,0,1,2]
# in this order), with color 0 (the first and fourth position) split into
# labels 0 and 3 to give 4 distinct grains -- see module docstring. Both
# periodicity vectors (Lx and Ly) preserve (i-j)%3, unlike the first
# (buggy) attempt.
_FRACS = [(0.0, 0.0), (1.0 / 6, 0.5), (2.0 / 6, 0.0), (3.0 / 6, 0.5), (4.0 / 6, 0.0), (5.0 / 6, 0.5)]
_POSITION_LABELS = [0, 1, 2, 3, 1, 2]
SEEDS = np.array([[fx * LX, fy * LY] for fx, fy in _FRACS])
N_GRAINS = 4


def grain_labels(eta):
    return np.argmax(np.abs(np.asarray(eta)), axis=0)[:, :, 0]


def grain_count(eta):
    return len(np.unique(np.argmax(np.abs(np.asarray(eta)), axis=0)))


def find_junctions(labels, n_grains, merge_distance=10.0):
    presence = np.zeros((n_grains,) + labels.shape, dtype=bool)
    for k in range(n_grains):
        mask = labels == k
        touching = mask.copy()
        for axis in (0, 1):
            for shift in (1, -1):
                touching |= np.roll(mask, shift, axis=axis)
        presence[k] = touching
    distinct = presence.sum(axis=0) == 3
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


def angular_widths(labels, junction, radius_index, n_grains, n_samples=1440):
    jx, jy = junction
    nx, ny = labels.shape
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    xs = np.round(jx + radius_index * np.cos(theta)).astype(int) % nx
    ys = np.round(jy + radius_index * np.sin(theta)).astype(int) % ny
    owner = labels[xs, ys]
    return np.array([(owner == k).sum() for k in range(n_grains)]) / n_samples * 360.0


def plot_hexagonal_four_grain():
    grid = Grid(shape=(NX, NY, 1), lengths=(LX, LY, 1.0))
    raw_ids, _ = periodic_voronoi(grid, len(SEEDS), seeds=SEEDS)
    grain_ids = np.array(_POSITION_LABELS)[np.asarray(raw_ids)][:, :, 0]
    neighbors = grain_neighbors(grain_ids)
    print("neighbor sets (no grain should neighbor itself):", neighbors)
    assert all(g not in n for g, n in enumerate(neighbors)), "a grain neighbors itself"

    eta = np.zeros((N_GRAINS, NX, NY, 1), dtype=np.float32)
    for i in range(N_GRAINS):
        eta[i, :, :, 0] = grain_ids == i

    solver = GrainOrientation(
        grid, mobility=MOBILITY, gradient_energy=GRADIENT_ENERGY,
        barrier_coefficient=BARRIER_COEFFICIENT, quartic_coefficient=QUARTIC_COEFFICIENT,
        cross_coefficient=CROSS_COEFFICIENT,
    )

    radius_index = JUNCTION_RADIUS * NY
    steps_recorded, rms_deviation, n_junctions_history = [0], [], []

    def measure(snap):
        labels = grain_labels(snap)
        junctions = find_junctions(labels, N_GRAINS)
        widths = [angular_widths(labels, j, radius_index, N_GRAINS) for j in junctions]
        deviations = [x - 120.0 for w in widths for x in w if x > 1.0]
        rms_deviation.append(float(np.sqrt(np.mean(np.square(deviations)))) if deviations else float("nan"))
        n_junctions_history.append(len(junctions))

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

    fig, axes = plt.subplots(1, len(snapshots), figsize=(3.6 * len(snapshots), 3.6 * LY / LX),
                              constrained_layout=True)
    for ax, (step, snap) in zip(axes, snapshots):
        labels = grain_labels(snap)
        ax.imshow(np.swapaxes(labels, 0, 1), origin="lower", cmap="Set1", vmin=0, vmax=N_GRAINS - 1,
                   interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"step {step}, {grain_count(snap)} grains")
    save_all(fig, "grain_orientation.hexagonal_four_grain_microstructure")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    ax2.plot(steps_recorded, rms_deviation, "o-", color="C0", markersize=4)
    ax2.axhline(0.0, color="0.6", linestyle="--", linewidth=1)
    ax2.set_xlabel("step")
    ax2.set_ylabel(r"RMS deviation from $120^\circ$")
    save_all(fig2, "grain_orientation.hexagonal_four_grain_convergence")
    plt.close(fig2)

    print(f"final grain count = {final_grain_count}")
    print(f"number of triple junctions detected: step 0 = {n_junctions_history[0]}, "
          f"step {steps_recorded[-1]} = {n_junctions_history[-1]}")
    print(f"RMS deviation from 120deg: step 0 = {rms_deviation[0]:.3f}, "
          f"step {steps_recorded[-1]} = {rms_deviation[-1]:.3f}")


if __name__ == "__main__":
    plot_hexagonal_four_grain()
