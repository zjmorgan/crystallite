"""The conduction references are the antiplane (out-of-plane) components of the
elastic Eshelby solution:

    kappa_il = C_{i3l3},  X*_i = 2 eps*_{i3},  q_i = sigma_{i3},  S_il = 2 S_{i3l3}

(i, l in-plane; kappa = mu isotropic, kappa_11 = c55 and kappa_22 = c44 for an
orthotropic matrix). These tests check that our conduction depolarization
tensors, interior and exterior fields equal the elastic ones under that map.
"""

import numpy as np
import pytest

from crystallite.grid import Grid
from crystallite.verification.anisotropic_inclusion import (
    antiplane_orthotropic_stiffness,
    isolated_eshelby_tensor,
    isotropic_tensor,
    periodic_eshelby_tensor,
    rotate_stiffness,
)
from crystallite.verification.conduction import (
    circular_inhomogeneity_driving_force,
    ConductionInclusionCase,
    elliptical_perturbation,
    equivalent_driving_force,
    interior_driving_force,
    isolated_depolarization_tensor,
    periodic_depolarization_tensor,
)
from crystallite.verification.elastic_deformation import (
    EllipticalHoleInPlateCase,
    HoleInPlateCase,
    _antiplane_inhomogeneity_polar_stress,
    _ellipse_inhomogeneity_antiplane_exterior_stress,
    _lattice_eshelby_tensor,
)

LAM, MU = 1.0, 0.7


def _conductivity(stiffness):
    """kappa_il = C_{i3l3} as a (3, 3) tensor (the x3-x3 entry is arbitrary)."""
    kappa = np.eye(3)
    for i in range(2):
        for l in range(2):
            kappa[i, l] = stiffness[i, 2, l, 2]
    return kappa


def _depolarization(eshelby):
    """S_il = 2 S_{i3l3}, in-plane block."""
    return 2.0 * np.array([[eshelby[i, 2, l, 2] for l in range(2)] for i in range(2)])


def _matrices():
    isotropic = isotropic_tensor(LAM, MU)
    orthotropic = antiplane_orthotropic_stiffness(4.0, LAM, MU)
    return {
        "isotropic": isotropic,
        "orthotropic": orthotropic,
        "rotated": rotate_stiffness(orthotropic, 0.6),
    }


@pytest.mark.parametrize("name", ["isotropic", "orthotropic", "rotated"])
@pytest.mark.parametrize("a, b", [(0.1, 0.1), (0.06, 0.12), (0.12, 0.05)])
def test_isolated_depolarization_is_the_elastic_antiplane_eshelby_tensor(name, a, b):
    stiffness = _matrices()[name]
    conduction = isolated_depolarization_tensor(_conductivity(stiffness), a, b)[:2, :2]
    elastic = _depolarization(isolated_eshelby_tensor(stiffness, a, b))
    np.testing.assert_allclose(conduction, elastic, atol=1e-10)


@pytest.mark.parametrize("name", ["isotropic", "orthotropic", "rotated"])
@pytest.mark.parametrize("a, b", [(0.1, 0.1), (0.06, 0.12)])
def test_periodic_depolarization_is_the_elastic_antiplane_lattice_sum(name, a, b):
    stiffness = _matrices()[name]
    conduction = periodic_depolarization_tensor(
        _conductivity(stiffness), a, b, 1.0, 1.0, n_modes=256
    )[:2, :2]
    elastic = _depolarization(periodic_eshelby_tensor(stiffness, a, b, 1.0, 1.0, n_modes=256))
    np.testing.assert_allclose(conduction, elastic, atol=1e-9)


def test_periodic_depolarization_matches_the_isotropic_elastic_s1313_and_s2323():
    a, b = 0.06, 0.12
    lattice = _lattice_eshelby_tensor(a, b, 1.0, 1.0, LAM, MU, n_modes=512)
    conduction = periodic_depolarization_tensor(MU, a, b, 1.0, 1.0, n_modes=512)
    assert conduction[0, 0] == pytest.approx(2.0 * lattice["s1313"], abs=1e-8)
    assert conduction[1, 1] == pytest.approx(2.0 * lattice["s2323"], abs=1e-8)


@pytest.mark.parametrize("contrast", [0.0, 0.125, 0.5, 1.5, 8.0, float("inf")])
def test_circle_periodic_interior_flux_is_the_elastic_antiplane_interior_stress(contrast):
    # remote sigma_13 = tau  <->  mean driving force X = tau / mu along x1
    grid = Grid(shape=(16, 16, 1))
    tau, a = 0.013, 0.1
    elastic = HoleInPlateCase(
        grid, matrix_lame_lambda=LAM, matrix_lame_mu=MU, hole_radius=a, contrast=contrast
    ).periodic_antiplane_interior_stress(tau)
    flux = ConductionInclusionCase(grid, MU, a, a, contrast=contrast).periodic_interior_flux(
        [tau / MU, 0.0, 0.0], n_modes=1024
    )
    assert flux[0] == pytest.approx(float(elastic[0, 2]), rel=2e-4, abs=1e-9)
    assert flux[1] == pytest.approx(float(elastic[1, 2]), abs=1e-9)


@pytest.mark.parametrize("contrast", [0.125, 0.5, 1.5, 8.0])
@pytest.mark.parametrize("a, b", [(0.06, 0.12), (0.12, 0.05)])
def test_ellipse_periodic_interior_flux_is_the_elastic_antiplane_interior_stress(contrast, a, b):
    grid = Grid(shape=(16, 16, 1))
    tau = 0.013
    elastic = EllipticalHoleInPlateCase(
        grid, matrix_lame_lambda=LAM, matrix_lame_mu=MU, semi_axis_a=a, semi_axis_b=b,
        contrast=contrast,
    ).periodic_antiplane_interior_stress(tau)
    flux = ConductionInclusionCase(grid, MU, a, b, contrast=contrast).periodic_interior_flux(
        [tau / MU, 0.0, 0.0], n_modes=1024
    )
    assert flux[0] == pytest.approx(float(elastic[0, 2]), rel=2e-4)


@pytest.mark.parametrize("contrast", [0.0, 0.5, 8.0, float("inf")])
def test_isolated_circle_field_is_the_elastic_antiplane_polar_stress(contrast):
    tau, a = 0.02, 0.1
    r = np.array([0.11, 0.15, 0.3, 0.6, 0.09, 0.05])
    theta = np.array([0.2, 1.1, 2.0, 3.5, 0.7, 5.0])
    x, y = r * np.cos(theta), r * np.sin(theta)
    x1, x2 = circular_inhomogeneity_driving_force(x, y, a, contrast, (tau / MU, 0.0))
    q_r = MU * (x1 * np.cos(theta) + x2 * np.sin(theta))
    q_theta = MU * (-x1 * np.sin(theta) + x2 * np.cos(theta))
    sigma_r3, sigma_theta3 = _antiplane_inhomogeneity_polar_stress(r, theta, a, tau, contrast)
    outside = r >= a
    np.testing.assert_allclose(q_r[outside], np.asarray(sigma_r3)[outside], atol=1e-12)
    np.testing.assert_allclose(q_theta[outside], np.asarray(sigma_theta3)[outside], atol=1e-12)


@pytest.mark.parametrize("contrast", [0.0, 0.125, 8.0, float("inf")])
@pytest.mark.parametrize("a, b", [(0.12, 0.05), (0.05, 0.12), (0.1, 0.07)])
def test_ellipse_exterior_is_the_elastic_antiplane_eigenstrain_field(contrast, a, b):
    # eigen X* of the equivalent inclusion  <->  eps*_{i3} = X*_i / 2; flux perturbation mu dX = sigma_{i3}
    exciting = np.array([0.9, -0.4, 0.0])
    s = isolated_depolarization_tensor(1.0, a, b)
    interior = interior_driving_force(contrast, s, exciting)
    star = equivalent_driving_force(contrast, s, exciting, interior)
    eigenstrain = np.zeros((3, 3))
    eigenstrain[0, 2], eigenstrain[1, 2] = 0.5 * star[0], 0.5 * star[1]
    rng = np.random.default_rng(3)
    angle = rng.uniform(0.0, 2.0 * np.pi, 8)
    scale = rng.uniform(1.3, 4.0, 8)
    x, y = scale * a * np.cos(angle), scale * b * np.sin(angle)
    dx, dy = elliptical_perturbation(x, y, a, b, contrast, exciting[:2])
    sigma_13, sigma_23 = _ellipse_inhomogeneity_antiplane_exterior_stress(
        x, y, eigenstrain, a, b, MU
    )
    np.testing.assert_allclose(MU * dx, np.asarray(sigma_13), atol=1e-9)
    np.testing.assert_allclose(MU * dy, np.asarray(sigma_23), atol=1e-9)
