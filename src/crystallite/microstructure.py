"""Polycrystal microstructures on a periodic grid: tessellations, crystal
orientations, and the orientation-dependent property fields the solvers take.

Everything here builds host (NumPy) arrays; the solvers move them to the
active backend. Property fields put the tensor axes first, ``tensor.shape +
grid.shape``, as :class:`crystallite.elastic_deformation.ElasticDeformation`,
:class:`crystallite.conduction.SteadyConduction` and
:class:`crystallite.mass_diffusion.MassDiffusion` expect.

**Orientation convention.** A rotation matrix ``R`` has the crystal axes as its
columns, expressed in the sample frame, so a crystal-frame tensor ``T`` becomes
``R T R^T`` (rank 2), ``R R R R : C`` (rank 4), and so on -- the same
convention as :meth:`crystallite.material.properties.Solid.rotate`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def rotation_from_quaternion(quaternion):
    """Rotation matrices from unit quaternions ``(w, x, y, z)``.

    Parameters
    ----------
    quaternion : array_like of shape (..., 4)
        Normalized internally.

    Returns
    -------
    ndarray of shape (..., 3, 3)
    """
    q = np.asarray(quaternion, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    rotation = np.empty(q.shape[:-1] + (3, 3))
    rotation[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotation[..., 0, 1] = 2.0 * (x * y - z * w)
    rotation[..., 0, 2] = 2.0 * (x * z + y * w)
    rotation[..., 1, 0] = 2.0 * (x * y + z * w)
    rotation[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotation[..., 1, 2] = 2.0 * (y * z - x * w)
    rotation[..., 2, 0] = 2.0 * (x * z - y * w)
    rotation[..., 2, 1] = 2.0 * (y * z + x * w)
    rotation[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rotation


def random_rotations(n, seed=None):
    """`n` rotation matrices uniformly distributed on SO(3) (unit quaternions
    from a 4D normal, which is uniform on the 3-sphere and so on SO(3)).

    Parameters
    ----------
    n : int
    seed : int or numpy.random.Generator, optional

    Returns
    -------
    ndarray of shape (n, 3, 3)
    """
    rng = np.random.default_rng(seed)
    return rotation_from_quaternion(rng.normal(size=(n, 4)))


def rotation_from_bunge_euler(phi1, big_phi, phi2, degrees=True):
    r"""Rotation matrix for Bunge (ZXZ) Euler angles, in this module's
    convention (crystal axes as columns, in the sample frame):
    :math:`R=R_z(\varphi_1)R_x(\Phi)R_z(\varphi_2)` with active rotations, the
    transpose of the usual sample-to-crystal matrix :math:`g`. Accepts arrays
    (broadcast together); returns shape ``(..., 3, 3)``.
    """
    phi1, big_phi, phi2 = np.broadcast_arrays(
        *(np.deg2rad(a) if degrees else np.asarray(a, dtype=float) for a in (phi1, big_phi, phi2))
    )
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c, s = np.cos(big_phi), np.sin(big_phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    rotation = np.empty(phi1.shape + (3, 3))
    rotation[..., 0, 0] = c1 * c2 - s1 * c * s2
    rotation[..., 0, 1] = -c1 * s2 - s1 * c * c2
    rotation[..., 0, 2] = s1 * s
    rotation[..., 1, 0] = s1 * c2 + c1 * c * s2
    rotation[..., 1, 1] = -s1 * s2 + c1 * c * c2
    rotation[..., 1, 2] = -c1 * s
    rotation[..., 2, 0] = s * s2
    rotation[..., 2, 1] = s * c2
    rotation[..., 2, 2] = c
    return rotation


def rotate_tensor(tensor, rotation):
    """Rotate a tensor of any rank by `rotation` (one matrix, or a stack
    ``(n, 3, 3)``, giving a stack of tensors): every index is transformed,
    ``T'_{ij...} = R_{ia} R_{jb} ... T_{ab...}``. Polar tensors only (proper
    rotations do not distinguish axial ones)."""
    tensor = np.asarray(tensor, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    if rotation.ndim == 3:
        return np.stack([rotate_tensor(tensor, r) for r in rotation])
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must be (3, 3) or (n, 3, 3), got {rotation.shape}")
    rotated = tensor
    for axis in range(tensor.ndim):
        rotated = np.moveaxis(np.tensordot(rotation, rotated, axes=([1], [axis])), 0, axis)
    return rotated


def periodic_voronoi(grid, n_grains, seeds=None, seed=None):
    """Grain ids of a periodic Voronoi tessellation on `grid`.

    Voxels take the id of the nearest seed under the minimum-image convention
    (a torus, so the tessellation is periodic), over the grid's active axes: a
    grid with one point along an axis gives a 2D tessellation.

    Parameters
    ----------
    grid : Grid
    n_grains : int
        Number of random seeds (ignored if `seeds` is given).
    seeds : array_like of shape (n, ndim_active), optional
        Seed positions in physical units, within ``[0, length)`` of each
        active axis.
    seed : int or numpy.random.Generator, optional
        For the random seeds.

    Returns
    -------
    grain_ids : ndarray of int, shape ``grid.shape``
    seeds : ndarray of shape (n, ndim_active)
        Seeds actually used. Grains too small to hold a voxel simply do not
        appear in `grain_ids`.
    """
    axes = [axis % 3 for axis in grid.fft_axes]
    lengths = np.array([grid.lengths[axis] for axis in axes], dtype=float)
    if seeds is None:
        rng = np.random.default_rng(seed)
        seeds = rng.random((n_grains, len(axes))) * lengths
    seeds = np.asarray(seeds, dtype=float)
    if seeds.ndim != 2 or seeds.shape[1] != len(axes):
        raise ValueError(
            f"seeds must have shape (n, {len(axes)}) for this grid, got {seeds.shape}"
        )
    if np.any(seeds < 0.0) or np.any(seeds >= lengths):
        raise ValueError("seeds must lie within [0, length) along each active axis")
    coordinates = [np.arange(grid.shape[axis]) * grid.lengths[axis] / grid.shape[axis] for axis in axes]
    points = np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1).reshape(-1, len(axes))
    _, nearest = cKDTree(seeds, boxsize=lengths).query(points)
    return nearest.reshape(grid.shape), seeds


def grain_neighbors(grain_ids, axes=None):
    """Which grains touch, across the periodic boundary too.

    Parameters
    ----------
    grain_ids : ndarray of int
        Grain-id field.
    axes : sequence of int, optional
        Axes to compare along. Default: every axis with more than one point.

    Returns
    -------
    list of set of int
        ``neighbors[g]`` is the set of grains touching grain ``g``.
    """
    grain_ids = np.asarray(grain_ids)
    if axes is None:
        axes = [axis for axis, n in enumerate(grain_ids.shape) if n > 1]
    neighbors = [set() for _ in range(int(grain_ids.max()) + 1)]
    for axis in axes:
        other = np.roll(grain_ids, -1, axis=axis)
        touching = grain_ids != other
        pairs = np.unique(np.stack([grain_ids[touching], other[touching]], axis=1), axis=0)
        for g, h in pairs:
            neighbors[g].add(int(h))
            neighbors[h].add(int(g))
    return neighbors


def _assign_slots(neighbors, n_slots, rng, attempts=50):
    """Random greedy colouring of the grain graph with `n_slots` colours, so
    touching grains differ; a colour is drawn uniformly among the free ones,
    which spreads the colours (orientations) evenly over the grains."""
    n_grains = len(neighbors)
    for _ in range(attempts):
        slots = np.full(n_grains, -1)
        complete = True
        for grain in rng.permutation(n_grains):
            used = {slots[h] for h in neighbors[grain] if slots[h] >= 0}
            free = [s for s in range(n_slots) if s not in used]
            if not free:
                complete = False
                break
            slots[grain] = rng.choice(free)
        if complete:
            return slots
    raise ValueError(
        f"could not give touching grains different orientations from a pool of "
        f"{n_slots}: the most-connected grain has {max(len(n) for n in neighbors)} "
        "neighbors; use a larger pool"
    )


@dataclass(frozen=True)
class Microstructure:
    r"""A periodic polycrystal: grain ids, and a crystal orientation per grain.

    Orientations live on *slots*, the orientation pool that the
    order-parameter fields of
    :class:`crystallite.grain_orientation.GrainOrientation` index: every grain
    in a slot shares its orientation. With one slot per grain each grain has
    its own orientation; with a smaller pool, touching grains are given
    different slots (a random graph colouring), so a large polycrystal needs
    few order parameters.

    Build one with :meth:`voronoi`.

    Attributes
    ----------
    grid : Grid
    grain_ids : ndarray of int, shape ``grid.shape``
    slots : ndarray of int, shape ``(n_grains,)``
        Slot (order-parameter index) of each grain.
    slot_orientations : ndarray, shape ``(n_slots, 3, 3)``
        Rotation matrix of each slot (crystal axes as columns).
    seeds : ndarray or None
        Seed positions of a Voronoi tessellation.
    """

    grid: object
    grain_ids: object
    slots: object
    slot_orientations: object
    seeds: object = None

    @classmethod
    def voronoi(cls, grid, n_grains, orientations=None, seed=None, seeds=None):
        """Random periodic Voronoi polycrystal.

        Parameters
        ----------
        grid : Grid
        n_grains : int
            Requested number of grains (grains too small to hold a voxel are
            dropped, so :attr:`n_grains` can be smaller).
        orientations : None, int or array_like of shape (m, 3, 3)
            ``None``: one uniformly random orientation per grain. An ``int``
            ``m``: a random pool of ``m`` orientations, assigned so that
            touching grains differ. An array: an explicit pool (e.g. from
            :func:`rotation_from_bunge_euler`, or a textured sample). A pool
            the size of the number of grains gives grain ``g`` orientation
            ``g``.
        seed : int, optional
            Seeds the seeds, the orientations and the assignment.
        seeds : array_like, optional
            Explicit seed positions (see :func:`periodic_voronoi`).
        """
        rng = np.random.default_rng(seed)
        raw_ids, seed_positions = periodic_voronoi(grid, n_grains, seeds=seeds, seed=rng)
        present, ids = np.unique(raw_ids, return_inverse=True)
        ids = ids.reshape(raw_ids.shape)
        n_present = len(present)

        if orientations is None:
            pool = random_rotations(n_present, seed=rng)
        elif isinstance(orientations, (int, np.integer)):
            pool = random_rotations(int(orientations), seed=rng)
        else:
            pool = np.asarray(orientations, dtype=float)
            if pool.ndim != 3 or pool.shape[1:] != (3, 3):
                raise ValueError(f"orientations must have shape (m, 3, 3), got {pool.shape}")
        if len(pool) == len(seed_positions) and len(pool) != n_present:
            pool = pool[present]

        if len(pool) == n_present:
            slots = np.arange(n_present)
        elif len(pool) < n_present:
            slots = _assign_slots(grain_neighbors(ids), len(pool), rng)
        else:
            raise ValueError(
                f"{len(pool)} orientations for {n_present} grains: give at most one per grain"
            )
        return cls(grid=grid, grain_ids=ids, slots=slots, slot_orientations=pool, seeds=seed_positions)

    @property
    def n_grains(self):
        return len(self.slots)

    @property
    def n_slots(self):
        return len(self.slot_orientations)

    @property
    def orientations(self):
        """Rotation matrix of each grain, shape ``(n_grains, 3, 3)``."""
        return self.slot_orientations[self.slots]

    def slot_field(self):
        """Slot index at every voxel, shape ``grid.shape``."""
        return self.slots[self.grain_ids]

    def rotation_field(self):
        """Rotation matrix at every voxel, shape ``(3, 3) + grid.shape``."""
        return np.moveaxis(self.orientations[self.grain_ids], (-2, -1), (0, 1)).astype(
            self.grid.real_dtype
        )

    def property_field(self, tensor):
        """A crystal-frame property placed grain by grain in the sample frame:
        each voxel holds `tensor` rotated by its grain's orientation.

        Parameters
        ----------
        tensor : array_like
            Any rank: a scalar (returned uniform), a ``(3, 3)`` conductivity
            or mobility, a ``(3, 3, 3, 3)`` stiffness, ...

        Returns
        -------
        ndarray of shape ``tensor.shape + grid.shape``, dtype ``grid.real_dtype``
            Ready for ``SteadyConduction(grid, field, ...)``,
            ``ElasticDeformation(..., stiffness=field, ...)``, etc. Sharp
            across grain boundaries.
        """
        tensor = np.asarray(tensor, dtype=float)
        if tensor.ndim == 0:
            return np.full(self.grid.shape, float(tensor), dtype=self.grid.real_dtype)
        per_grain = rotate_tensor(tensor, self.orientations)
        field = per_grain[self.grain_ids]
        rank = tensor.ndim
        return np.moveaxis(field, tuple(range(-rank, 0)), tuple(range(rank))).astype(
            self.grid.real_dtype
        )

    def average_property(self, tensor):
        """Volume average (Voigt, arithmetic) of a crystal-frame property over
        the grains: the natural homogeneous reference medium of a polycrystal,
        e.g. ``GreenOperator(grid, ms.average_property(stiffness))`` for
        :class:`crystallite.elastic_deformation.ElasticDeformation`.

        Returns
        -------
        ndarray of `tensor`'s shape
        """
        counts = np.bincount(self.grain_ids.ravel(), minlength=self.n_grains)
        rotated = rotate_tensor(np.asarray(tensor, dtype=float), self.orientations)
        return np.tensordot(counts / counts.sum(), rotated, axes=([0], [0]))

    def grain_interior(self, grain, margin=1):
        """Mask of grain `grain` with `margin` voxels shaved off, periodically:
        the points whose whole neighborhood of half-width `margin` (along every
        axis with more than one point) is in the grain. Shape ``grid.shape``."""
        mask = self.grain_ids == grain
        active = [n > 1 for n in mask.shape]
        pad = [(margin, margin) if a else (0, 0) for a in active]
        box = np.ones([2 * margin + 1 if a else 1 for a in active], dtype=bool)
        eroded = ndimage.binary_erosion(np.pad(mask, pad, mode="wrap"), structure=box)
        return eroded[tuple(slice(m, -m if m else None) if a else slice(None) for (m, _), a in zip(pad, active))]

    def order_parameters(self, amplitude=1.0, smooth=True):
        r"""Initial condition for
        :class:`crystallite.grain_orientation.GrainOrientation`: one order
        parameter per slot, equal to `amplitude` inside the grains of that slot
        and 0 elsewhere (``amplitude=sqrt(a/b)`` for its barrier and quartic
        coefficients). With `smooth` the sharp indicators are Lanczos-filtered
        (band-limited, mean preserved), which the semi-implicit scheme then
        relaxes to the diffuse equilibrium profile.

        Returns
        -------
        ndarray of shape ``(n_slots,) + grid.shape``, dtype ``grid.real_dtype``
            Order parameter ``i`` carries orientation ``slot_orientations[i]``.
        """
        slot_field = self.slot_field()
        eta = np.stack([amplitude * (slot_field == s) for s in range(self.n_slots)])
        eta = eta.astype(self.grid.real_dtype)
        if smooth:
            grid = self.grid
            hat = grid.fft(eta)
            eta = np.real(np.asarray(grid.ifft(hat * grid.lanczos_filter))).astype(grid.real_dtype)
        return eta


def orientation_weights(eta):
    r"""Partition of unity from order parameters, :math:`w_i=\eta_i^2/
    \sum_j\eta_j^2` (uniform where every :math:`\eta` vanishes): the share of
    each orientation at a point, shape of `eta`, summing to 1 over axis 0."""
    eta = np.asarray(eta, dtype=float)
    squared = eta**2
    total = squared.sum(axis=0, keepdims=True)
    uniform = 1.0 / eta.shape[0]
    return np.where(total > 0.0, squared / np.where(total > 0.0, total, 1.0), uniform)


def mixed_property_field(tensor, rotations, weights, dtype=np.float32):
    r"""A crystal-frame property mixed over orientations,
    :math:`P(x)=\sum_i w_i(x)\,R_i\cdot T` (arithmetic mixing of the rotated
    tensors) -- for order-parameter fields that evolve, including ones grown
    from noise with no tessellation: :math:`w=` :func:`orientation_weights`
    of ``eta`` and ``rotations`` the orientation attached to each order
    parameter. Other mixing rules replace this one function.

    Parameters
    ----------
    tensor : array_like
        Crystal-frame property of any rank.
    rotations : array_like of shape (n, 3, 3)
    weights : array_like of shape (n,) + grid.shape

    Returns
    -------
    ndarray of shape ``tensor.shape + grid.shape``
    """
    weights = np.asarray(weights, dtype=float)
    rotated = rotate_tensor(tensor, rotations)
    if rotated.shape[0] != weights.shape[0]:
        raise ValueError(
            f"{rotated.shape[0]} rotations but {weights.shape[0]} weight fields"
        )
    return np.tensordot(rotated, weights, axes=([0], [0])).astype(dtype)
