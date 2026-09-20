import numpy as np
import pytest

from crystallite.grid import Grid
from crystallite.verification.anisotropic_inclusion import (
    AnisotropicInclusionCase,
    antiplane_orthotropic_stiffness,
    cubic_from_zener,
    interior_stress,
    isolated_eshelby_tensor,
    isotropic_tensor,
    periodic_eshelby_tensor,
    rotate_stiffness,
)
from crystallite.verification.elastic_deformation import _lattice_eshelby_tensor

LAM, MU = 1.0, 0.7
NU = LAM / (2.0 * (LAM + MU))


def _isotropic_ellipse_tensor(a, b):
    d = 2.0 * (1.0 - NU)
    return {
        (0, 0, 0, 0): ((b**2 + 2 * a * b) / (a + b) ** 2 + (1 - 2 * NU) * b / (a + b)) / d,
        (1, 1, 1, 1): ((a**2 + 2 * a * b) / (a + b) ** 2 + (1 - 2 * NU) * a / (a + b)) / d,
        (0, 0, 1, 1): (b**2 / (a + b) ** 2 - (1 - 2 * NU) * b / (a + b)) / d,
        (1, 1, 0, 0): (a**2 / (a + b) ** 2 - (1 - 2 * NU) * a / (a + b)) / d,
        (0, 1, 0, 1): ((a**2 + b**2) / (2 * (a + b) ** 2) + (1 - 2 * NU) / 2) / d,
        (0, 2, 0, 2): b / (2 * (a + b)),
        (1, 2, 1, 2): a / (2 * (a + b)),
    }


@pytest.mark.parametrize("a, b", [(0.1, 0.1), (0.06, 0.12), (0.12, 0.06), (0.02, 0.2)])
def test_isolated_tensor_reproduces_the_isotropic_ellipse_closed_forms(a, b):
    tensor = isolated_eshelby_tensor(isotropic_tensor(LAM, MU), a, b)
    for index, value in _isotropic_ellipse_tensor(a, b).items():
        assert tensor[index] == pytest.approx(value, abs=1e-12)


def test_cubic_with_unit_zener_ratio_is_the_isotropic_medium():
    np.testing.assert_allclose(cubic_from_zener(1.0, LAM, MU), isotropic_tensor(LAM, MU), atol=1e-13)


def test_periodic_tensor_matches_the_independent_isotropic_lattice_sum():
    a = 0.1
    general = periodic_eshelby_tensor(isotropic_tensor(LAM, MU), a, a, 1.0, 1.0, n_modes=512)
    isotropic = _lattice_eshelby_tensor(a, a, 1.0, 1.0, LAM, MU)
    assert general[0, 0, 0, 0] == pytest.approx(isotropic["s1111"], abs=2e-5)
    assert general[1, 1, 0, 0] == pytest.approx(isotropic["s2211"], abs=2e-5)
    assert general[0, 0, 1, 1] == pytest.approx(isotropic["s1122"], abs=2e-5)
    assert general[0, 1, 0, 1] == pytest.approx(isotropic["s1212"], abs=2e-5)
    assert general[0, 2, 0, 2] == pytest.approx(isotropic["s1313"], abs=2e-5)
    assert general[1, 2, 1, 2] == pytest.approx(isotropic["s2323"], abs=2e-5)


@pytest.mark.parametrize("zener", [0.125, 8.0])
def test_isolated_circle_tensor_is_covariant_under_rotation_of_the_matrix(zener):
    # a circle is isotropic, so rotating the matrix rotates the Eshelby tensor
    angle = 0.7
    c = cubic_from_zener(zener)
    rotated_stiffness = rotate_stiffness(c, angle)
    direct = isolated_eshelby_tensor(rotated_stiffness, 0.1, 0.1)
    cos, sin = np.cos(angle), np.sin(angle)
    r = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    rotated_tensor = np.einsum("ia,jb,kc,ld,abcd->ijkl", r, r, r, r, isolated_eshelby_tensor(c, 0.1, 0.1))
    np.testing.assert_allclose(direct, rotated_tensor, atol=1e-11)


@pytest.mark.parametrize("zener", [0.125, 8.0])
def test_cubic_periodic_tensor_is_unchanged_by_a_quarter_turn_of_the_matrix(zener):
    # cubic symmetry and the square cell are both 4-fold, so 90 degrees is a symmetry
    c = cubic_from_zener(zener)
    base = periodic_eshelby_tensor(c, 0.1, 0.1, 1.0, 1.0, n_modes=256)
    turned = periodic_eshelby_tensor(rotate_stiffness(c, np.pi / 2), 0.1, 0.1, 1.0, 1.0, n_modes=256)
    np.testing.assert_allclose(turned, base, atol=1e-10)


@pytest.mark.parametrize("angle", [0.0, 0.3, 1.1])
@pytest.mark.parametrize("zener", [0.125, 8.0])
def test_stiffness_times_eshelby_tensor_has_the_energy_symmetry(zener, angle):
    # C:S is symmetric under exchanging the two index pairs (a property of the Green operator)
    c = rotate_stiffness(cubic_from_zener(zener), angle)
    s = isolated_eshelby_tensor(c, 0.08, 0.12)
    cs = np.einsum("mnij,ijkl->mnkl", c, s)
    np.testing.assert_allclose(cs, np.einsum("mnkl->klmn", cs), atol=1e-11)


@pytest.mark.parametrize("ratio", [0.125, 0.5, 2.0, 8.0])
def test_antiplane_interior_stress_matches_the_coordinate_mapping_oracle(ratio):
    # c55 u,11 + c44 u,22 = 0 becomes the isotropic Laplace equation under
    # y' = y sqrt(c55/c44), turning the circle into an ellipse: sigma_13 =
    # -2 c55 e / (1 + sqrt(c55/c44)) and sigma_23 = -2 c44 e / (1 + sqrt(c44/c55))
    c = antiplane_orthotropic_stiffness(ratio, LAM, MU)
    c44, c55 = c[1, 2, 1, 2], c[0, 2, 0, 2]
    e = 0.01
    tensor = isolated_eshelby_tensor(c, 0.1, 0.1)
    eps13 = np.zeros((3, 3)); eps13[0, 2] = eps13[2, 0] = e
    eps23 = np.zeros((3, 3)); eps23[1, 2] = eps23[2, 1] = e
    assert interior_stress(c, tensor, eps13)[0, 2] == pytest.approx(
        -2.0 * c55 * e / (1.0 + np.sqrt(c55 / c44)), rel=1e-10
    )
    assert interior_stress(c, tensor, eps23)[1, 2] == pytest.approx(
        -2.0 * c44 * e / (1.0 + np.sqrt(c44 / c55)), rel=1e-10
    )


@pytest.mark.parametrize("zener", [0.125, 8.0])
def test_periodic_tensor_approaches_the_isolated_one_for_a_small_cubic_inclusion(zener):
    c = cubic_from_zener(zener)
    a = 0.01  # area fraction ~3e-4
    isolated = isolated_eshelby_tensor(c, a, a)
    periodic = periodic_eshelby_tensor(c, a, a, 1.0, 1.0, n_modes=1024)
    np.testing.assert_allclose(periodic, isolated, atol=2e-3)


def test_cubic_eigenstrain_stress_depends_on_orientation_with_a_quarter_turn_period():
    a = 0.1
    eps = np.zeros((3, 3)); eps[0, 0] = 0.01
    c = cubic_from_zener(8.0)
    grid = Grid(shape=(16, 16, 1), lengths=(1.0, 1.0, 1.0))
    stresses = []
    for angle in (0.0, np.pi / 4, np.pi / 2):
        case = AnisotropicInclusionCase(grid, rotate_stiffness(c, angle), a, a)
        stresses.append(case.periodic_interior_stress(eps, n_modes=256))
    np.testing.assert_allclose(stresses[2], stresses[0], atol=1e-10)
    assert abs(stresses[1][0, 0] - stresses[0][0, 0]) > 1e-3  # the anisotropy is real


@pytest.mark.parametrize("zener, load", [(8.0, (0, 0)), (0.125, (0, 1)), (8.0, (0, 1))])
def test_periodic_interior_stress_agrees_with_an_independent_numeric_solve(zener, load):
    # 192^2 CG solve with the anisotropic constitutive law, sampled at the centre
    grid = Grid(shape=(192, 192, 1), lengths=(1.0, 1.0, 1.0))
    case = AnisotropicInclusionCase(grid, cubic_from_zener(zener), 0.1, 0.1)
    eps = np.zeros((3, 3)); eps[load] = eps[load[::-1]] = 0.01
    solution = case.solver().solve(
        np.zeros((3, 3)), eigenstrain=case.eigenstrain_field(eps), tol=1e-6, max_iterations=200
    )
    assert solution.converged
    inside = np.asarray(case.elliptical_radius) < 0.5
    stress = np.asarray(solution.stress)
    analytic = case.periodic_interior_stress(eps)
    scale = max(np.abs(analytic).max(), 1e-12)
    for i, j in ((0, 0), (1, 1), (0, 1)):
        assert float(np.mean(stress[i, j][inside])) == pytest.approx(analytic[i, j], abs=0.02 * scale)
