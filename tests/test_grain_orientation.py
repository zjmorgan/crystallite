import numpy as np
import pytest

from crystallite.grain_orientation import GrainOrientation
from crystallite.grid import Grid


def test_step_rejects_wrong_field_shape():
    grid = Grid(shape=(8, 8, 1))
    model = GrainOrientation(grid)
    with pytest.raises(ValueError, match="grid.shape"):
        model.step(np.zeros((2, 4, 4, 1)), 1.0e-2)


def test_step_rejects_negative_time_step():
    grid = Grid(shape=(8, 8, 1))
    model = GrainOrientation(grid)
    with pytest.raises(ValueError, match="time_step"):
        model.step(np.zeros((2,) + grid.shape), -1.0)


def test_local_term_matches_finite_difference_of_free_energy():
    # A spatially uniform shift leaves the gradient term unchanged (the
    # gradient of a constant is zero), isolating the local (bulk) term's
    # functional derivative: d(mean free energy)/dh at a uniform shift h
    # is exactly the spatial mean of the local nonlinear term.
    grid = Grid(shape=(16, 16, 1), lengths=(1.0, 1.0, 1.0))
    model = GrainOrientation(
        grid, mobility=1.0, gradient_energy=0.05,
        barrier_coefficient=1.0, quartic_coefficient=1.0, cross_coefficient=1.5,
    )
    rng = np.random.default_rng(2)
    eta = 0.3 * rng.standard_normal((2,) + grid.shape)

    h = 1.0e-5
    orientation = 1
    plus = eta.copy()
    plus[orientation] += h
    minus = eta.copy()
    minus[orientation] -= h
    finite_difference = (
        float(model.free_energy(plus)) - float(model.free_energy(minus))
    ) / (2 * h)

    local_mean = float(np.mean(np.asarray(model._nonlinear_term(eta))[orientation]))

    assert local_mean == pytest.approx(finite_difference, rel=1e-4)


def test_driving_force_gradient_term_matches_spectral_laplacian():
    # -kappa * Laplacian(eta) on a smooth, exactly-known field, checked
    # against the analytic Laplacian directly (avoids the single-grid-cell
    # perturbation subtlety of relating a spectral functional derivative
    # to a finite-volume cell perturbation).
    grid = Grid(shape=(16, 16, 1), lengths=(1.0, 1.0, 1.0))
    kappa = 0.05
    model = GrainOrientation(grid, gradient_energy=kappa, barrier_coefficient=0.0, quartic_coefficient=0.0)

    x = np.asarray(grid.x[0])
    y = np.asarray(grid.x[1])
    smooth = np.broadcast_to(
        np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y), grid.shape
    ).astype(np.float64)
    eta = smooth.reshape((1,) + grid.shape)

    driving = np.asarray(model.driving_force(eta))[0]
    exact_laplacian = -((2 * np.pi) ** 2) * 2.0 * smooth
    expected_gradient_term = -kappa * exact_laplacian

    np.testing.assert_allclose(driving, expected_gradient_term, atol=5e-3)


def test_semi_implicit_matches_the_exact_linear_solution():
    # With barrier_coefficient = quartic_coefficient = cross_coefficient = 0,
    # the nonlinear remainder N is identically zero and the equation is the
    # exactly-linear d(eta_hat)/dt = -mobility*gradient_energy*k^2*eta_hat.
    # Unlike MassDiffusion's semi-implicit scheme (bug fixed earlier this
    # session), GrainOrientation's step divides the *whole* update by
    # (1 + h*mobility*L(k)) -- correct here because delta F/delta eta_i is
    # itself the evolution rate, with no extra spatial-derivative layer the
    # way Cahn-Hilliard's div(M grad(mu)) has. With N=0 this must match
    # backward Euler (and hence the true exponential decay) almost exactly,
    # even at a time step far too large for any explicit scheme.
    grid = Grid(shape=(32, 32, 1))
    mobility, kappa = 1.0, 0.05
    model = GrainOrientation(
        grid, mobility=mobility, gradient_energy=kappa,
        barrier_coefficient=0.0, quartic_coefficient=0.0, cross_coefficient=0.0,
    )

    rng = np.random.default_rng(0)
    eta0 = 0.01 * rng.standard_normal((1,) + grid.shape)

    dt, steps = 1.0e-2, 2000
    eta = eta0.copy()
    for _ in range(steps):
        eta = model.step(eta, dt)
    eta = np.asarray(eta)

    rate = mobility * kappa * np.asarray(grid.k2)
    exact_hat = np.asarray(grid.fft(eta0[0])) * np.exp(-rate * steps * dt)
    exact = np.asarray(grid.ifft(exact_hat))

    np.testing.assert_allclose(eta[0], exact, atol=1e-10)


def test_free_energy_decreases_monotonically():
    # Allen-Cahn's defining property: F is a Lyapunov functional.
    grid = Grid(shape=(32, 32, 1))
    model = GrainOrientation(
        grid, mobility=1.0, gradient_energy=0.05,
        barrier_coefficient=1.0, quartic_coefficient=1.0, cross_coefficient=1.5,
    )
    rng = np.random.default_rng(1)
    eta = 0.1 * rng.standard_normal((3,) + grid.shape)

    energy = float(model.free_energy(eta))
    for _ in range(200):
        eta = model.step(eta, 1.0e-2)
        next_energy = float(model.free_energy(eta))
        assert next_energy <= energy + 1e-10
        energy = next_energy


def test_single_grain_states_are_equilibria():
    # eta = (0, ..., +-eta0, ..., 0) for eta0 = sqrt(a/b) is a single-grain
    # state -- a genuine equilibrium of the local dynamics regardless of a,
    # b: the nonlinear term eta0*(-a + b*eta0**2) = eta0*(-a + a) = 0
    # exactly, at any permutation/sign. (The free-energy *value* there is
    # -a**2/(4b), not 0 in general -- the reference slide's "=0" holds only
    # for its own specific normalization, not the general a, b case.)
    a, b, gamma = 2.0, 3.0, 1.5
    grid = Grid(shape=(4, 4, 1))
    model = GrainOrientation(
        grid, barrier_coefficient=a, quartic_coefficient=b, cross_coefficient=gamma
    )
    eta0 = np.sqrt(a / b)

    for sign in (1.0, -1.0):
        eta = np.zeros((3,) + grid.shape)
        eta[1] = sign * eta0
        local_term = np.asarray(model._nonlinear_term(eta))
        np.testing.assert_allclose(local_term, 0.0, atol=1e-10)
