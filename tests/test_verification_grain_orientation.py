import numpy as np
import pytest

from crystallite.grain_orientation import GrainOrientation
from crystallite.grid import Grid
from crystallite.verification import CircularGrainCase, PlanarGrainBoundaryCase


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


def _zero_crossing_radius(eta, grid, center):
    """Measure a circular grain's radius along the horizontal line through
    its center by linearly interpolating the sign change -- robust to the
    diffuse interface in a way a naive area count is not."""
    row = np.asarray(eta)[0, :, grid.shape[1] // 2, 0]
    xs = np.asarray(grid.x[0])[:, 0, 0]
    center_index = np.argmin(np.abs(xs - center[0]))
    for i in range(center_index, len(xs) - 1):
        if row[i] > 0 and row[i + 1] <= 0:
            fraction = row[i] / (row[i] - row[i + 1])
            return xs[i] + fraction * (xs[i + 1] - xs[i]) - center[0]
    return None


def test_circular_grain_eta0_width_and_surface_energy_match_analytic_relations():
    case = CircularGrainCase(
        Grid(shape=(64, 64, 1)),
        barrier_coefficient=2.0,
        quartic_coefficient=3.0,
        gradient_energy=1.0e-3,
    )
    assert case.eta0 == pytest.approx(np.sqrt(2.0 / 3.0))
    assert case.width > 0
    assert case.surface_energy == pytest.approx(
        (4.0 / 3.0) * 1.0e-3 * case.eta0**2 / case.width
    )


def test_circular_grain_inactive_orientations_are_zero():
    grid = Grid(shape=(64, 64, 1))
    case = CircularGrainCase(
        grid, gradient_energy=1.0e-3, n_orientations=4, active_orientation=2,
        center=(0.5, 0.5),
    )
    eta = np.asarray(case.profile(0.2))

    for i in range(4):
        if i != 2:
            np.testing.assert_allclose(eta[i], 0.0)
    assert np.max(np.abs(eta[2])) == pytest.approx(case.eta0, rel=1e-2)


def test_circular_grain_static_energy_matches_perimeter_times_surface_energy():
    # F = 2*pi*R*sigma (2D), verified directly against the solver's own
    # free_energy() rather than trusting the closed-form surface_energy in
    # isolation -- the two must agree, since sigma is *defined* as the
    # static excess energy per unit boundary length.
    a, b = 1.0, 1.0
    grid = Grid(shape=(200, 200, 1), lengths=(1.0, 1.0, 1.0))
    case = CircularGrainCase(
        grid, barrier_coefficient=a, quartic_coefficient=b,
        gradient_energy=2.0e-3, mobility=1.0, center=(0.5, 0.5),
    )
    radius = 0.3
    eta = case.profile(radius)

    model = GrainOrientation(
        grid, mobility=case.mobility, gradient_energy=case.gradient_energy,
        barrier_coefficient=a, quartic_coefficient=b, cross_coefficient=1.5,
    )
    domain_area = grid.lengths[0] * grid.lengths[1]
    background = -(a**2) / (4.0 * b)
    excess_energy = float(model.free_energy(eta)) * domain_area - background * domain_area

    assert excess_energy == pytest.approx(2.0 * np.pi * radius * case.surface_energy, rel=1e-2)


def test_circular_grain_radius_shrinks_according_to_the_curvature_flow_law():
    # The rate is set by gradient_energy, *not* surface_energy -- see
    # CircularGrainCase's docstring; this is the regression test for that
    # distinction (using surface_energy here would predict a shrink rate
    # off by a large, non-order-1 factor).
    a, b, kappa, mobility = 1.0, 1.0, 2.0e-3, 1.0
    grid = Grid(shape=(200, 200, 1), lengths=(1.0, 1.0, 1.0))
    case = CircularGrainCase(
        grid, barrier_coefficient=a, quartic_coefficient=b,
        gradient_energy=kappa, mobility=mobility, center=(0.5, 0.5),
    )
    initial_radius = 0.3
    eta = case.profile(initial_radius)

    model = GrainOrientation(
        grid, mobility=mobility, gradient_energy=kappa,
        barrier_coefficient=a, quartic_coefficient=b, cross_coefficient=1.5,
    )

    time_step = 0.02
    times = []
    radii_squared = []
    t = 0.0
    for _ in range(100):
        radius = _zero_crossing_radius(eta, grid, case.center)
        assert radius is not None
        times.append(t)
        radii_squared.append(radius**2)
        eta = model.step(eta, time_step)
        t += time_step

    times = np.array(times)
    radii_squared = np.array(radii_squared)
    analytic = np.array(
        [float(case.radius(tt, initial_radius)) ** 2 for tt in times]
    )

    np.testing.assert_allclose(radii_squared, analytic, atol=2e-3)

    slope = np.polyfit(times, radii_squared, 1)[0]
    assert slope == pytest.approx(-2.0 * mobility * kappa, rel=0.05)
