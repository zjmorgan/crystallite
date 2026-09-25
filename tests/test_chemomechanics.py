import numpy as np
import pytest

from crystallite.chemomechanics import CoherentDiffusion
from crystallite.elastic_deformation import ElasticDeformation
from crystallite.grid import Grid
from crystallite.mass_diffusion import MassDiffusion
from crystallite.verification.anisotropic_inclusion import cubic_from_zener
from crystallite.microstructure import Microstructure, rotate_tensor, rotation_from_bunge_euler
from crystallite.verification.chemomechanics import (
    axis_diagonal_anisotropy,
    interface_orientation,
    khachaturyan_b,
)

EPS0 = 0.05
STIFFNESS_SCALE = 25.0
MODES = [(1, 0), (2, 0), (1, 1), (1, 2), (2, 3), (5, 4)]


def _coherent(zener, dealias=True, n=64, **diffusion):
    grid = Grid(shape=(n, n, 1))
    stiffness = STIFFNESS_SCALE * cubic_from_zener(zener, 1.0, 0.7)
    elasticity = ElasticDeformation(grid, 1.0, 0.7, 1.0, 0.7, stiffness=stiffness)
    diffusion.setdefault("gradient_energy", 1.0e-4)
    solver = MassDiffusion(grid, mobility=1.0, **diffusion)
    coherent = CoherentDiffusion(
        solver, elasticity, EPS0 * np.eye(3), reference_composition=0.5, dealias=dealias
    )
    return grid, stiffness, coherent


def _mode(grid, m, amplitude):
    x, y = np.asarray(grid.x[0]), np.asarray(grid.x[1])
    return 0.5 + amplitude * np.cos(2.0 * np.pi * (m[0] * x + m[1] * y))


# --- the extra chemical potential hook ---------------------------------------


def test_extra_chemical_potential_hook():
    grid = Grid(shape=(32, 32, 1))
    solver = MassDiffusion(grid, mobility=1.0, gradient_energy=0.0, left_well=0.0, right_well=1.0)
    rng = np.random.default_rng(0)
    c = (0.5 + 0.2 * rng.standard_normal(grid.shape)).astype(np.float32)
    plain = np.asarray(solver.step(c, 1e-4))
    np.testing.assert_array_equal(np.asarray(solver.step(c, 1e-4, extra_chemical_potential=None)), plain)
    # a uniform potential drives no flux
    np.testing.assert_allclose(
        np.asarray(solver.step(c, 1e-4, extra_chemical_potential=np.full(grid.shape, 3.0))), plain, atol=1e-6
    )
    # a potential cancelling f'(c) leaves a flat mu, so nothing moves (no gradient energy)
    cancel = -np.asarray(solver.free_energy_derivative(c))
    np.testing.assert_allclose(
        np.asarray(solver.step(c, 1e-3, extra_chemical_potential=cancel)), c, atol=1e-5
    )
    with pytest.raises(ValueError, match="extra_chemical_potential"):
        solver.step(c, 1e-4, extra_chemical_potential=np.ones((4, 4, 1)))


# --- the coupling against Khachaturyan's elastic energy ----------------------


@pytest.mark.parametrize("zener", [3.0, 1.0, 1.0 / 3.0])
@pytest.mark.parametrize("mode", MODES)
def test_the_elastic_potential_of_a_mode_is_khachaturyans_b(zener, mode):
    grid, stiffness, coupled = _coherent(zener, dealias=False)
    amplitude = 0.05
    c = _mode(grid, mode, amplitude)
    solution = coupled.solve_elastic(c, tol=1e-6)
    assert solution.converged
    mu = np.asarray(coupled.elastic_potential(solution))
    x, y = np.asarray(grid.x[0]), np.asarray(grid.x[1])
    projection = 2.0 * np.mean(mu * np.cos(2.0 * np.pi * (mode[0] * x + mode[1] * y))) / amplitude
    expected = khachaturyan_b(stiffness, EPS0 * np.eye(3), [mode[0], mode[1], 0.0])
    assert projection == pytest.approx(expected, rel=1e-5)


def test_the_filter_scales_each_mode_by_its_gain_squared():
    grid, stiffness, coupled = _coherent(3.0, dealias=True)
    mode, amplitude = (2, 1), 0.05
    solution = coupled.solve_elastic(_mode(grid, mode, amplitude), tol=1e-6)
    mu = np.asarray(coupled.elastic_potential(solution))
    x, y = np.asarray(grid.x[0]), np.asarray(grid.x[1])
    projection = 2.0 * np.mean(mu * np.cos(2.0 * np.pi * (mode[0] * x + mode[1] * y))) / amplitude
    sinc = lambda m: np.sin(2 * np.pi * m / 64) / (2 * np.pi * m / 64)
    gain = sinc(mode[0]) * sinc(mode[1])
    expected = khachaturyan_b(stiffness, EPS0 * np.eye(3), [2.0, 1.0, 0.0]) * gain**2
    assert projection == pytest.approx(expected, rel=1e-5)


def test_elastic_soft_directions_follow_the_zener_ratio():
    # dilatational misfit: B is smallest along <100> for A_Z > 1, along <110> in plane for A_Z < 1,
    # and does not depend on direction for an isotropic crystal
    def b(zener, direction):
        return khachaturyan_b(
            STIFFNESS_SCALE * cubic_from_zener(zener), EPS0 * np.eye(3), direction
        )

    assert b(3.0, [1, 0, 0]) < b(3.0, [1, 1, 0])
    assert b(1.0 / 3.0, [1, 1, 0]) < b(1.0 / 3.0, [1, 0, 0])
    assert b(1.0, [1, 0, 0]) == pytest.approx(b(1.0, [1, 1, 0]), rel=1e-9)
    assert b(1.0, [1, 0, 0]) == pytest.approx(b(1.0, [3, 2, 0]), rel=1e-9)


@pytest.mark.parametrize("dealias", [False, True])
def test_the_elastic_potential_is_the_functional_derivative_of_the_elastic_energy(dealias):
    grid, _, coupled = _coherent(3.0, dealias=dealias)
    rng = np.random.default_rng(1)
    n = grid.shape[0]
    i, j = np.arange(n)[:, None], np.arange(n)[None, :]

    def smooth():
        field = sum(
            rng.normal() * np.cos(2 * np.pi * (kx * i + ky * j) / n + rng.uniform(0, 6.28))
            for kx in range(-5, 6) for ky in range(-5, 6)
        )
        return (field / np.abs(field).max())[:, :, None].astype(np.float32)

    c, h = (0.5 + 0.1 * smooth()).astype(np.float32), smooth()

    def energy(composition):
        return float(coupled.elastic_energy(composition, coupled.solve_elastic(composition, tol=1e-7)))

    step = 2.0e-2
    derivative = (energy(c + step * h) - energy(c - step * h)) / (2.0 * step)
    potential = np.asarray(coupled.elastic_potential(coupled.solve_elastic(c, tol=1e-7)))
    assert derivative == pytest.approx(float(np.mean(potential * h)), rel=1e-4)


def test_a_modes_elastic_energy_is_a_quarter_b_amplitude_squared():
    grid, stiffness, coupled = _coherent(3.0, dealias=False)
    amplitude = 0.05
    c = _mode(grid, (1, 1), amplitude)
    energy = float(coupled.elastic_energy(c, coupled.solve_elastic(c, tol=1e-6)))
    b = khachaturyan_b(stiffness, EPS0 * np.eye(3), [1.0, 1.0, 0.0])
    assert energy == pytest.approx(0.25 * b * amplitude**2, rel=1e-5)


def test_a_homogeneous_stiffness_solves_in_about_one_iteration():
    grid, _, coupled = _coherent(3.0)
    solution = coupled.solve_elastic(_mode(grid, (2, 1), 0.05), tol=1e-6)
    assert solution.converged and solution.iterations <= 3


# --- validation -----------------------------------------------------------------


def test_construction_validation():
    grid = Grid(shape=(16, 16, 1))
    elasticity = ElasticDeformation(grid, 1.0, 0.7, 1.0, 0.7)
    solver = MassDiffusion(grid)
    with pytest.raises(ValueError, match="misfit_strain"):
        CoherentDiffusion(solver, elasticity, np.ones(3))
    other = MassDiffusion(Grid(shape=(8, 8, 1)))
    with pytest.raises(ValueError, match="grid"):
        CoherentDiffusion(other, elasticity, np.eye(3))


# --- the morphology --------------------------------------------------------------


def _anisotropy(zener, seed, n=64, steps=300, dt=2.0e-4):
    """(axis - diagonal) / (axis + diagonal) weight of the structure factor of a
    spinodal decomposition driven by the misfit alone."""
    grid, _, coupled = _coherent(
        zener, n=n, gradient_energy=2.0e-4, left_well=0.0, right_well=1.0,
        scheme="semi_implicit", reference_curvature=2.0, dealias=True,
    )
    rng = np.random.default_rng(seed)
    c = (0.5 + 0.01 * rng.standard_normal(grid.shape)).astype(np.float32)
    displacement = None
    for _ in range(steps):
        c, solution = coupled.step(c, dt, displacement=displacement, tol=1e-5, max_iterations=200)
        displacement = solution.displacement
    c = np.asarray(c)[:, :, 0]
    return axis_diagonal_anisotropy(c), c


def test_misfit_alone_aligns_the_decomposition_with_the_elastically_soft_directions():
    means = {}
    for zener in (3.0, 1.0, 1.0 / 3.0):
        values = []
        for seed in range(4):
            value, composition = _anisotropy(zener, seed)
            values.append(value)
            assert composition.min() < 0.2 and composition.max() > 0.8  # it did decompose
            assert composition.mean() == pytest.approx(0.5, abs=1e-3)  # and conserved mass
        means[zener] = np.mean(values)
    assert means[3.0] > 0.7  # modulations along <100>
    assert means[1.0 / 3.0] < -0.5  # along the in-plane <110>
    assert abs(means[1.0]) < 0.2  # nothing to choose between for an isotropic crystal


# --- a misfit field, and the polycrystal -------------------------------------


def _angle_error(measured, expected):
    return (measured - expected + 45.0) % 90.0 - 45.0


def test_interface_orientation_of_known_stripes():
    n = 64
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    for m in ((3, 1), (1, 3), (3, 0), (2, 2), (5, 2)):
        stripes = np.cos(2.0 * np.pi * (m[0] * i + m[1] * j) / n)
        angle, strength = interface_orientation(stripes)
        assert abs(_angle_error(angle, np.degrees(np.arctan2(m[1], m[0])))) < 1e-6
        assert strength == pytest.approx(1.0, abs=1e-9)
    stripes = np.cos(2.0 * np.pi * (3 * i + 1 * j) / n)
    half = (i < n // 2).astype(float)  # a weight window picks the same orientation
    assert abs(_angle_error(interface_orientation(stripes, weight=half)[0], np.degrees(np.arctan2(1, 3)))) < 1e-6
    noise = np.random.default_rng(0).standard_normal((n, n))
    assert interface_orientation(noise)[1] < 0.1


def test_a_misfit_field_equals_the_constant_tensor_when_uniform():
    grid, _, constant = _coherent(3.0, dealias=False)
    field = CoherentDiffusion(
        constant.diffusion, constant.elasticity,
        np.broadcast_to((EPS0 * np.eye(3))[:, :, None, None, None], (3, 3) + grid.shape),
        reference_composition=0.5, dealias=False,
    )
    c = _mode(grid, (2, 1), 0.05)
    a, b = constant.solve_elastic(c, tol=1e-6), field.solve_elastic(c, tol=1e-6)
    np.testing.assert_allclose(
        np.asarray(constant.elastic_potential(a)), np.asarray(field.elastic_potential(b)), atol=1e-6
    )
    with pytest.raises(ValueError, match="misfit_strain"):
        CoherentDiffusion(constant.diffusion, constant.elasticity, np.ones((3, 3, 4, 4, 1)))


def _polycrystal(zener, angles, n, seed=0, misfit=None):
    """A periodic polycrystal of cubic crystals turned in the plane by `angles`
    (one per grain), and its coupled misfit-driven solver."""
    grid = Grid(shape=(n, n, 1))
    rotations = rotation_from_bunge_euler(np.asarray(angles, dtype=float), 0.0, 0.0)
    microstructure = Microstructure.voronoi(grid, len(angles), orientations=rotations, seed=seed)
    crystal = STIFFNESS_SCALE * cubic_from_zener(zener, 1.0, 0.7)
    diffusion = MassDiffusion(
        grid, mobility=1.0, gradient_energy=2.0e-4, left_well=0.0, right_well=1.0,
        scheme="semi_implicit", reference_curvature=2.0, dealias=True,
    )
    coupled = CoherentDiffusion.polycrystal(
        diffusion, microstructure, crystal, EPS0 * np.eye(3) if misfit is None else misfit,
        reference_composition=0.5,
    )
    return grid, microstructure, crystal, coupled


def test_a_rotated_crystal_with_a_rotated_misfit_has_the_rotated_b():
    misfit = EPS0 * np.diag([1.0, -0.6, 0.2])  # crystal frame, not dilatational
    grid, microstructure, crystal, coupled = _polycrystal(3.0, [25.0], 64, misfit=misfit)
    coupled = CoherentDiffusion(
        coupled.diffusion, coupled.elasticity, coupled.misfit_strain,
        reference_composition=0.5, dealias=False,
    )
    rotation = microstructure.orientations[0]
    mode, amplitude = (2, 1), 0.05
    solution = coupled.solve_elastic(_mode(grid, mode, amplitude), tol=1e-6)
    assert solution.converged
    x, y = np.asarray(grid.x[0]), np.asarray(grid.x[1])
    wave = np.cos(2.0 * np.pi * (mode[0] * x + mode[1] * y))
    projection = 2.0 * np.mean(np.asarray(coupled.elastic_potential(solution)) * wave) / amplitude
    expected = khachaturyan_b(
        rotate_tensor(crystal, rotation), rotate_tensor(misfit, rotation), [2.0, 1.0, 0.0]
    )
    assert projection == pytest.approx(expected, rel=2e-4)


@pytest.mark.parametrize("zener, shift", [(3.0, 0.0), (1.0 / 3.0, 45.0)])
def test_each_grain_of_a_polycrystal_decomposes_along_its_own_soft_direction(zener, shift):
    # two grains whose crystals are turned 40 degrees apart: the modulations follow the crystal,
    # <100> for A_Z > 1 and the diagonal for A_Z < 1, grain by grain
    angles = [10.0, 50.0]
    grid, microstructure, _, coupled = _polycrystal(zener, angles, 64)
    rng = np.random.default_rng(1)
    c = (0.5 + 0.01 * rng.standard_normal(grid.shape)).astype(np.float32)
    displacement = None
    for _ in range(300):
        c, solution = coupled.step(c, 2.0e-4, displacement=displacement, tol=1e-4, max_iterations=300)
        displacement = solution.displacement
    c = np.asarray(c)[:, :, 0]
    assert c.min() < 0.2 and c.max() > 0.8 and c.mean() == pytest.approx(0.5, abs=1e-3)
    for grain in range(microstructure.n_grains):
        rotation = microstructure.orientations[grain]
        crystal_angle = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])) % 90.0
        mask = microstructure.grain_interior(grain, margin=5)[:, :, 0]
        assert mask.sum() > 300
        angle, strength = interface_orientation(c, weight=mask)
        assert strength > 0.25
        assert abs(_angle_error(angle, crystal_angle + shift)) < 10.0


def test_an_off_critical_polycrystal_forms_aligned_precipitates_of_the_minority_phase():
    # mean composition 0.35: the minority phase occupies about a third of the cell (lever rule)
    # and still follows each grain's <100> soft direction, though more weakly than stripes
    grid, microstructure, _, coupled = _polycrystal(3.0, [10.0, 50.0], 64)
    rng = np.random.default_rng(1)
    c = (0.35 + 0.01 * rng.standard_normal(grid.shape)).astype(np.float32)
    displacement = None
    for _ in range(900):
        c, solution = coupled.step(c, 2.0e-4, displacement=displacement, tol=1e-4, max_iterations=300)
        displacement = solution.displacement
    c = np.asarray(c)[:, :, 0]
    assert c.min() < 0.2 and c.max() > 0.8 and c.mean() == pytest.approx(0.35, abs=1e-3)
    assert 0.2 < np.mean(c > 0.5) < 0.4
    for grain in range(microstructure.n_grains):
        rotation = microstructure.orientations[grain]
        crystal_angle = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])) % 90.0
        mask = microstructure.grain_interior(grain, margin=5)[:, :, 0]
        angle, strength = interface_orientation(c, weight=mask)
        assert strength > 0.2
        assert abs(_angle_error(angle, crystal_angle)) < 10.0


def test_polycrystal_constructor_validation():
    grid = Grid(shape=(16, 16, 1))
    ms = Microstructure.voronoi(Grid(shape=(8, 8, 1)), 2, seed=0)
    with pytest.raises(ValueError, match="grid"):
        CoherentDiffusion.polycrystal(
            MassDiffusion(grid), ms, cubic_from_zener(3.0), np.eye(3)
        )
