import numpy as np
import pytest

from crystallite.conduction import SteadyConduction
from crystallite.elastic_deformation import ElasticDeformation
from crystallite.grain_orientation import GrainOrientation
from crystallite.grid import Grid
from crystallite.microstructure import (
    Microstructure,
    grain_neighbors,
    mixed_property_field,
    orientation_weights,
    periodic_voronoi,
    random_rotations,
    rotate_tensor,
    rotation_from_bunge_euler,
    rotation_from_quaternion,
)
from crystallite.spectral.long_range import GreenOperator
from crystallite.verification.anisotropic_inclusion import cubic_from_zener, isotropic_tensor


def _is_rotation(rotation, atol=1e-12):
    rotation = np.asarray(rotation)
    identity = np.broadcast_to(np.eye(3), rotation.shape)
    np.testing.assert_allclose(rotation @ np.swapaxes(rotation, -1, -2), identity, atol=atol)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=atol)


# --- orientations -----------------------------------------------------------


def test_random_rotations_are_proper_reproducible_and_uniform():
    rotations = random_rotations(40000, seed=3)
    _is_rotation(rotations)
    np.testing.assert_array_equal(random_rotations(5, seed=3), random_rotations(5, seed=3))
    assert not np.allclose(random_rotations(5, seed=3), random_rotations(5, seed=4))
    # uniform on SO(3): E[R_ij R_kl] = delta_ik delta_jl / 3, E[R] = 0
    assert np.abs(rotations.mean(axis=0)).max() < 0.01
    assert (rotations[:, 0, 0] ** 2).mean() == pytest.approx(1.0 / 3.0, abs=0.01)
    assert (rotations[:, 0, 0] * rotations[:, 1, 1]).mean() == pytest.approx(0.0, abs=0.01)


def test_quaternion_rotations():
    np.testing.assert_allclose(rotation_from_quaternion([1.0, 0, 0, 0]), np.eye(3), atol=1e-15)
    quarter_turn_z = rotation_from_quaternion([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    np.testing.assert_allclose(quarter_turn_z, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-15)
    np.testing.assert_allclose(  # normalized internally
        rotation_from_quaternion([3.0, 0, 0, 3.0]), rotation_from_quaternion([1.0, 0, 0, 1.0]),
        atol=1e-14,
    )


def test_bunge_euler_rotations():
    np.testing.assert_allclose(rotation_from_bunge_euler(0, 0, 0), np.eye(3), atol=1e-15)
    # phi1 alone turns the crystal about z: its x axis (first column) ends up along sample y
    turned = rotation_from_bunge_euler(90.0, 0.0, 0.0)
    np.testing.assert_allclose(turned @ [1, 0, 0], [0, 1, 0], atol=1e-12)

    def about(axis, angle):
        c, s = np.cos(angle), np.sin(angle)
        return {
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
            "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        }[axis]

    phi1, big, phi2 = 0.4, 0.9, -1.3
    expected = about("z", phi1) @ about("x", big) @ about("z", phi2)
    np.testing.assert_allclose(
        rotation_from_bunge_euler(phi1, big, phi2, degrees=False), expected, atol=1e-14
    )
    stack = rotation_from_bunge_euler([10.0, 20.0, 30.0], 40.0, [1.0, 2.0, 3.0])
    assert stack.shape == (3, 3, 3)
    _is_rotation(stack)


def test_rotate_tensor_of_any_rank():
    rng = np.random.default_rng(0)
    rotation = random_rotations(1, seed=1)[0]
    vector, matrix = rng.normal(size=3), rng.normal(size=(3, 3))
    stiffness = rng.normal(size=(3, 3, 3, 3))
    np.testing.assert_allclose(rotate_tensor(vector, rotation), rotation @ vector, atol=1e-13)
    np.testing.assert_allclose(rotate_tensor(matrix, rotation), rotation @ matrix @ rotation.T, atol=1e-13)
    np.testing.assert_allclose(
        rotate_tensor(stiffness, rotation),
        np.einsum("ia,jb,kc,ld,abcd->ijkl", rotation, rotation, rotation, rotation, stiffness),
        atol=1e-13,
    )
    assert rotate_tensor(2.5, rotation) == 2.5
    assert rotate_tensor(matrix, random_rotations(4, seed=2)).shape == (4, 3, 3)
    with pytest.raises(ValueError, match="rotation"):
        rotate_tensor(matrix, np.eye(4))


def test_symmetry_is_respected_by_rotation():
    rotation = random_rotations(1, seed=5)[0]
    isotropic = isotropic_tensor(1.0, 0.7)
    np.testing.assert_allclose(rotate_tensor(isotropic, rotation), isotropic, atol=1e-13)
    cubic = cubic_from_zener(3.0)
    quarter_turn = rotation_from_bunge_euler(90.0, 0.0, 0.0)
    np.testing.assert_allclose(rotate_tensor(cubic, quarter_turn), cubic, atol=1e-13)
    assert not np.allclose(rotate_tensor(cubic, rotation), cubic)


# --- tessellation -----------------------------------------------------------


@pytest.mark.parametrize("shape", [(24, 20, 1), (12, 10, 8)])
def test_voronoi_is_the_nearest_seed_under_the_minimum_image(shape):
    grid = Grid(shape=shape, lengths=(1.0, 1.3, 0.9))
    ids, seeds = periodic_voronoi(grid, 7, seed=2)
    axes = [axis for axis in range(3) if shape[axis] > 1]
    assert ids.shape == shape and seeds.shape == (7, len(axes))
    lengths = np.array([grid.lengths[axis] for axis in axes])
    coordinates = [np.arange(shape[axis]) * grid.lengths[axis] / shape[axis] for axis in axes]
    points = np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1).reshape(-1, len(axes))
    delta = np.abs(points[:, None, :] - seeds[None, :, :])
    delta = np.minimum(delta, lengths - delta)
    brute = np.argmin((delta**2).sum(axis=-1), axis=1).reshape(shape)
    np.testing.assert_array_equal(ids, brute)


def test_voronoi_seed_validation():
    grid = Grid(shape=(16, 16, 1))
    with pytest.raises(ValueError, match="shape"):
        periodic_voronoi(grid, 3, seeds=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="within"):
        periodic_voronoi(grid, 3, seeds=np.array([[0.1, 0.2], [1.5, 0.2], [0.3, 0.3]]))


def test_grain_neighbors_includes_the_periodic_wrap():
    ids = np.array([0, 0, 1, 1, 2, 2]).reshape(6, 1, 1)
    assert grain_neighbors(ids) == [{1, 2}, {0, 2}, {0, 1}]  # 2 and 0 touch across the boundary
    assert grain_neighbors(np.array([0, 0, 1, 1]).reshape(4, 1, 1)) == [{1}, {0}]


# --- the microstructure -----------------------------------------------------


def test_one_orientation_per_grain_by_default():
    grid = Grid(shape=(48, 48, 1))
    ms = Microstructure.voronoi(grid, 15, seed=7)
    assert ms.n_grains == len(np.unique(ms.grain_ids)) == 15
    np.testing.assert_array_equal(np.unique(ms.grain_ids), np.arange(ms.n_grains))
    np.testing.assert_array_equal(ms.slots, np.arange(ms.n_grains))
    assert ms.n_slots == ms.n_grains
    _is_rotation(ms.orientations)
    again = Microstructure.voronoi(grid, 15, seed=7)
    np.testing.assert_array_equal(again.grain_ids, ms.grain_ids)
    np.testing.assert_array_equal(again.orientations, ms.orientations)


def test_a_smaller_pool_gives_touching_grains_different_orientations():
    grid = Grid(shape=(64, 64, 1))
    ms = Microstructure.voronoi(grid, 60, orientations=12, seed=4)
    assert ms.n_slots == 12 and ms.n_grains > 40
    for grain, others in enumerate(grain_neighbors(ms.grain_ids)):
        for other in others:
            assert ms.slots[grain] != ms.slots[other]
    assert len(np.unique(ms.slots)) >= 10  # the pool is actually used, not just its first few
    np.testing.assert_array_equal(ms.orientations, ms.slot_orientations[ms.slots])


def test_an_explicit_pool_is_used_as_given():
    grid = Grid(shape=(32, 32, 1))
    pool = rotation_from_bunge_euler(np.arange(6) * 30.0, 0.0, 0.0)
    ms = Microstructure.voronoi(grid, 6, orientations=pool, seed=1)
    assert ms.n_grains == 6
    np.testing.assert_allclose(ms.orientations, pool)


def test_orientation_pool_validation_and_infeasibility():
    grid = Grid(shape=(32, 32, 1))
    with pytest.raises(ValueError, match="at most one per grain"):
        Microstructure.voronoi(grid, 4, orientations=random_rotations(10, seed=0), seed=1)
    with pytest.raises(ValueError, match="shape"):
        Microstructure.voronoi(grid, 4, orientations=np.zeros((4, 3)), seed=1)
    with pytest.raises(ValueError, match="pool"):
        Microstructure.voronoi(grid, 30, orientations=2, seed=1)


def test_grains_too_small_for_a_voxel_are_dropped():
    grid = Grid(shape=(8, 8, 1))
    ms = Microstructure.voronoi(grid, 200, seed=0)
    assert ms.n_grains < 200
    np.testing.assert_array_equal(np.unique(ms.grain_ids), np.arange(ms.n_grains))
    explicit = Microstructure.voronoi(grid, 200, orientations=random_rotations(200, seed=1), seed=0)
    assert explicit.n_grains == ms.n_grains


def test_property_field_places_the_rotated_tensor_in_every_grain():
    grid = Grid(shape=(24, 24, 6))
    ms = Microstructure.voronoi(grid, 8, seed=2)
    stiffness = cubic_from_zener(3.0)
    field = ms.property_field(stiffness)
    assert field.shape == (3, 3, 3, 3) + grid.shape and field.dtype == np.float32
    for point in ((0, 0, 0), (5, 17, 2), (23, 3, 5), (11, 11, 3)):
        grain = ms.grain_ids[point]
        expected = rotate_tensor(stiffness, ms.orientations[grain])
        np.testing.assert_allclose(field[(Ellipsis,) + point], expected, atol=1e-6)
    conductivity = ms.property_field(np.diag([2.0, 1.0, 0.5]))
    assert conductivity.shape == (3, 3) + grid.shape
    np.testing.assert_allclose(  # rotation keeps the trace
        conductivity[0, 0] + conductivity[1, 1] + conductivity[2, 2], 3.5, atol=1e-5
    )
    isotropic = isotropic_tensor(1.0, 0.7)
    np.testing.assert_allclose(  # an isotropic tensor is the same in every grain
        ms.property_field(isotropic),
        np.broadcast_to(isotropic[..., None, None, None], (3, 3, 3, 3) + grid.shape),
        atol=1e-6,
    )
    np.testing.assert_allclose(ms.property_field(1.7), 1.7)
    assert ms.property_field(1.7).shape == grid.shape


def test_rotation_field_matches_the_orientations():
    grid = Grid(shape=(20, 20, 1))
    ms = Microstructure.voronoi(grid, 5, seed=3)
    field = ms.rotation_field()
    assert field.shape == (3, 3) + grid.shape
    voxel = (7, 13, 0)
    np.testing.assert_allclose(field[(Ellipsis,) + voxel], ms.orientations[ms.grain_ids[voxel]], atol=1e-6)


def test_average_property_is_the_volume_mean_of_the_property_field():
    grid = Grid(shape=(24, 24, 1))
    ms = Microstructure.voronoi(grid, 6, seed=3)
    stiffness = cubic_from_zener(3.0)
    np.testing.assert_allclose(
        ms.average_property(stiffness),
        ms.property_field(stiffness).astype(float).mean(axis=(4, 5, 6)), atol=1e-5,
    )
    conductivity = np.diag([2.0, 1.0, 0.5])
    np.testing.assert_allclose(
        ms.average_property(conductivity),
        ms.property_field(conductivity).astype(float).mean(axis=(2, 3, 4)), atol=1e-5,
    )
    assert ms.average_property(1.7) == pytest.approx(1.7)
    np.testing.assert_allclose(  # the same tensor in every grain averages to itself
        Microstructure.voronoi(grid, 5, orientations=rotation_from_bunge_euler(0.0, 0.0, 0.0)[None].repeat(5, 0), seed=1)
        .average_property(stiffness), stiffness, atol=1e-12,
    )


@pytest.mark.parametrize("margin", [0, 1, 3])
def test_grain_interior_keeps_only_points_whose_whole_neighborhood_is_in_the_grain(margin):
    grid = Grid(shape=(32, 32, 1))
    ms = Microstructure.voronoi(grid, 5, seed=2)
    for grain in range(ms.n_grains):
        interior = ms.grain_interior(grain, margin=margin)
        member = ms.grain_ids == grain
        assert interior.shape == grid.shape and not (interior & ~member).any()
        if margin == 0:
            np.testing.assert_array_equal(interior, member)
        for i, j in list(zip(*np.nonzero(interior[:, :, 0])))[:40]:
            for di in range(-margin, margin + 1):
                for dj in range(-margin, margin + 1):
                    assert member[(i + di) % 32, (j + dj) % 32, 0]
        if margin and interior.any():
            assert interior.sum() < member.sum()  # the boundary layer really is shaved off


# --- order parameters -------------------------------------------------------


def test_order_parameters_are_one_hot_indicators_of_the_slots():
    grid = Grid(shape=(32, 32, 1))
    ms = Microstructure.voronoi(grid, 20, orientations=6, seed=5)
    eta = ms.order_parameters(amplitude=1.3, smooth=False)
    assert eta.shape == (6,) + grid.shape and eta.dtype == np.float32
    np.testing.assert_allclose((eta**2).sum(axis=0), 1.3**2, atol=1e-6)
    assert ((eta != 0).sum(axis=0) == 1).all()
    np.testing.assert_array_equal(eta.argmax(axis=0), ms.slot_field())


def test_smoothed_order_parameters_are_band_limited_and_keep_their_means():
    grid = Grid(shape=(32, 32, 1))
    ms = Microstructure.voronoi(grid, 20, orientations=6, seed=5)
    sharp = ms.order_parameters(smooth=False)
    smooth = ms.order_parameters(smooth=True)
    np.testing.assert_allclose(smooth.mean(axis=(1, 2, 3)), sharp.mean(axis=(1, 2, 3)), atol=1e-6)
    assert not np.allclose(smooth, sharp)


def test_the_order_parameters_drive_the_grain_orientation_solver_downhill():
    grid = Grid(shape=(32, 32, 1))
    ms = Microstructure.voronoi(grid, 12, orientations=6, seed=9)
    solver = GrainOrientation(grid, gradient_energy=1.0e-3, cross_coefficient=1.5)
    eta = ms.order_parameters(amplitude=1.0)
    initial = float(solver.free_energy(eta))
    for _ in range(40):
        eta = solver.step(eta, 0.05)
    assert np.isfinite(np.asarray(eta)).all()
    assert float(solver.free_energy(eta)) < initial
    # grains are still there: each point is still dominated by one order parameter
    assert (np.abs(np.asarray(eta)).max(axis=0) > 0.5).mean() > 0.9


# --- mixing over orientations ------------------------------------------------


def test_orientation_weights_are_a_partition_of_unity():
    rng = np.random.default_rng(0)
    eta = rng.normal(size=(5, 6, 4, 1))
    weights = orientation_weights(eta)
    np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(orientation_weights(-eta), weights)
    one_hot = np.zeros((3, 2, 2, 1))
    one_hot[1] = 0.8
    np.testing.assert_allclose(orientation_weights(one_hot), (one_hot > 0).astype(float))
    np.testing.assert_allclose(orientation_weights(np.zeros((4, 2, 2, 1))), 0.25)


def test_mixing_one_hot_weights_reproduces_the_hard_property_field():
    grid = Grid(shape=(24, 24, 1))
    ms = Microstructure.voronoi(grid, 15, orientations=5, seed=6)
    weights = orientation_weights(ms.order_parameters(smooth=False))
    tensor = np.diag([2.0, 1.0, 0.5])
    mixed = mixed_property_field(tensor, ms.slot_orientations, weights)
    np.testing.assert_allclose(mixed, ms.property_field(tensor), atol=1e-6)


def test_mixing_equal_weights_averages_the_rotated_tensors():
    rotations = random_rotations(4, seed=1)
    weights = np.full((4, 3, 3, 1), 0.25)
    tensor = np.diag([2.0, 1.0, 0.5])
    mixed = mixed_property_field(tensor, rotations, weights)
    np.testing.assert_allclose(
        mixed[:, :, 1, 1, 0], rotate_tensor(tensor, rotations).mean(axis=0), atol=1e-6
    )
    with pytest.raises(ValueError, match="weight fields"):
        mixed_property_field(tensor, rotations, weights[:3])


# --- the polycrystal through the solvers -------------------------------------


def test_a_random_2d_polycrystal_conducts_as_the_geometric_mean():
    # Dykhne's exact result: uniaxial grains in random in-plane orientations
    # have effective conductivity sqrt(k_a k_b) (here 1, at geometric mean 1)
    ratio, values = 4.0, []
    for seed in range(4):
        grid = Grid(shape=(128, 128, 1))
        rng = np.random.default_rng(seed)
        rotations = rotation_from_bunge_euler(rng.uniform(0.0, 180.0, 100), 0.0, 0.0)
        ms = Microstructure.voronoi(grid, 100, orientations=rotations, seed=seed)
        kappa = ms.property_field(np.diag([np.sqrt(ratio), 1.0 / np.sqrt(ratio), 1.0]))
        solver = SteadyConduction(grid, kappa, reference_conductivity=np.eye(3))
        for field, i in (([1.0, 0.0, 0.0], 0), ([0.0, 1.0, 0.0], 1)):
            solution = solver.solve(field, tol=1e-6, max_iterations=4000)
            assert solution.converged
            values.append(float(np.asarray(solution.flux)[i].mean() / np.asarray(solution.driving_force)[i].mean()))
    assert np.mean(values) == pytest.approx(1.0, abs=0.01)


def _mandel(stiffness):
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    weight = [1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)]
    return np.array([
        [weight[a] * weight[b] * stiffness[i, j, k, l] for b, (k, l) in enumerate(pairs)]
        for a, (i, j) in enumerate(pairs)
    ])


def test_a_random_cubic_polycrystal_lies_between_the_voigt_and_reuss_bounds():
    grid = Grid(shape=(24, 24, 24))
    ms = Microstructure.voronoi(grid, 12, seed=1)
    crystal = cubic_from_zener(3.0)
    field = ms.property_field(crystal)
    voigt = field.astype(float).mean(axis=(4, 5, 6))
    solver = ElasticDeformation(
        grid, 1.0, 1.0, 1.0, 1.0, stiffness=field, reference_green=GreenOperator(grid, np.asarray(voigt, np.float32))
    )
    strain = np.zeros((3, 3))
    strain[0, 0] = 0.01
    solution = solver.solve(strain, tol=1e-5, max_iterations=2000)
    assert solution.converged
    energy = 0.5 * np.einsum("ij,ij", np.asarray(solution.stress).mean(axis=(2, 3, 4)), strain)
    volume = np.bincount(ms.grain_ids.ravel(), minlength=ms.n_grains) / ms.grain_ids.size
    reuss = np.linalg.inv(sum(v * np.linalg.inv(_mandel(rotate_tensor(crystal, r)))
                              for v, r in zip(volume, ms.orientations)))
    e = np.array([0.01, 0, 0, 0, 0, 0])
    assert 0.5 * e @ reuss @ e < energy < 0.5 * e @ _mandel(voigt) @ e


def test_a_polycrystal_grown_from_noise_maps_to_solvable_property_fields():
    # no tessellation and no assigned grains: each order parameter carries an orientation
    # from a pool sampled once, and the (evolving) order parameters weight them
    grid = Grid(shape=(64, 64, 1))
    n_orientations = 8
    eta = 1.0e-2 * np.random.default_rng(0).standard_normal((n_orientations,) + grid.shape)
    solver = GrainOrientation(grid, gradient_energy=5.0e-4, cross_coefficient=1.5)
    for _ in range(200):
        eta = solver.step(eta, 0.1)
    weights = orientation_weights(np.asarray(eta))
    # grains formed: most points are dominated by one order parameter, and several survive
    # (the eta^2 weights only saturate away from the diffuse boundaries, so test eta itself)
    assert (np.abs(np.asarray(eta)).max(axis=0) > 0.5).mean() > 0.9
    assert len(np.unique(np.asarray(eta).argmax(axis=0))) >= 4

    rotations = random_rotations(n_orientations, seed=1)
    kappa = mixed_property_field(np.diag([2.0, 1.0, 0.5]), rotations, weights)
    assert kappa.shape == (3, 3) + grid.shape and np.isfinite(kappa).all()
    eigenvalues = np.linalg.eigvalsh(np.moveaxis(kappa, (0, 1), (-2, -1)).reshape(-1, 3, 3))
    assert eigenvalues.min() > 0.4  # mixing positive-definite tensors stays positive definite
    reference = np.mean(rotate_tensor(np.diag([2.0, 1.0, 0.5]), rotations), axis=0)
    solution = SteadyConduction(grid, kappa, reference_conductivity=reference).solve(
        [1.0, 0.0, 0.0], tol=1e-6, max_iterations=3000
    )
    assert solution.converged
