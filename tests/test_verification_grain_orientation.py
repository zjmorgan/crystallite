import numpy as np
import pytest

from crystallite.grain_orientation import GrainOrientation
from crystallite.grid import Grid
from crystallite.verification import PlanarGrainBoundaryCase


def test_eta0_and_width_match_analytic_relations():
    case = PlanarGrainBoundaryCase(
        Grid(shape=(64, 1, 1)),
        barrier_coefficient=2.0,
        quartic_coefficient=3.0,
        gradient_energy=1.0e-3,
    )
    assert case.eta0 == pytest.approx(np.sqrt(2.0 / 3.0))
    assert case.width > 0


def test_inactive_orientations_are_zero():
    grid = Grid(shape=(64, 1, 1))
    case = PlanarGrainBoundaryCase(
        grid, gradient_energy=1.0e-3, n_orientations=4, active_orientation=2
    )
    eta = np.asarray(case.profile())

    for i in range(4):
        if i != 2:
            np.testing.assert_allclose(eta[i], 0.0)
    assert np.max(np.abs(eta[2])) == pytest.approx(case.eta0, rel=1e-3)


def test_planar_profile_is_a_driving_force_equilibrium():
    # Unlike a single tanh on a periodic domain, the exact kink-antikink
    # profile should zero the driving force everywhere, no margin needed
    # (same property established for the analogous Cahn-Hilliard case).
    grid = Grid(shape=(256, 1, 1), lengths=(1.0, 1.0, 1.0))
    case = PlanarGrainBoundaryCase(
        grid, barrier_coefficient=1.0, quartic_coefficient=1.0,
        gradient_energy=2.0e-3, center=0.5,
    )
    eta = case.profile()

    model = GrainOrientation(
        grid, mobility=1.0, gradient_energy=case.gradient_energy,
        barrier_coefficient=case.barrier_coefficient,
        quartic_coefficient=case.quartic_coefficient, cross_coefficient=1.0,
    )
    driving = np.asarray(model.driving_force(eta))

    assert np.max(np.abs(driving)) < 1e-6


def test_planar_profile_is_stationary_under_time_stepping():
    grid = Grid(shape=(256, 1, 1), lengths=(1.0, 1.0, 1.0))
    case = PlanarGrainBoundaryCase(
        grid, barrier_coefficient=1.0, quartic_coefficient=1.0,
        gradient_energy=2.0e-3, center=0.5,
    )
    eta = case.profile()

    model = GrainOrientation(
        grid, mobility=1.0, gradient_energy=case.gradient_energy,
        barrier_coefficient=case.barrier_coefficient,
        quartic_coefficient=case.quartic_coefficient,
    )
    updated = eta
    for _ in range(50):
        updated = model.step(updated, 1.0e-3)

    np.testing.assert_allclose(np.asarray(updated), np.asarray(eta), atol=1e-4)
