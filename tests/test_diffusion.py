import numpy as np
import pytest

from crystallite.diffusion import (
    LinearSpectralDiffusion,
    MassDiffusion,
    double_well_curvature,
    double_well_derivative,
    double_well_equilibrium_width,
)
from crystallite.phase import Grid
from crystallite.verification import InterfaceCase, SinusoidalCase


def test_linear_diffusion_decays_sinusoidal_mode():
    case = SinusoidalCase()
    solver = LinearSpectralDiffusion(
        case.grid, mobility=2.0, free_energy_curvature=3.0
    )
    field = solver.step(case.initial_field(), 0.25)
    rate = 2.0 * case.wave_number**2 * 3.0

    assert case.amplitude_of(field) == pytest.approx(
        case.expected_amplitude(0.25, rate), rel=1e-5
    )


def test_gradient_energy_gives_fourth_order_decay():
    case = SinusoidalCase()
    solver = LinearSpectralDiffusion(
        case.grid,
        mobility=2.0,
        gradient_energy=0.5,
        free_energy_curvature=3.0,
    )
    rate = 2.0 * case.wave_number**2 * (
        3.0 + 0.5 * case.wave_number**2
    )

    assert solver.decay_rate[case.mode, 0, 0] == pytest.approx(
        rate, rel=1e-5
    )


def test_anisotropic_mobility_and_gradient_energy():
    case = SinusoidalCase()
    solver = LinearSpectralDiffusion(
        case.grid,
        mobility=np.diag([2.0, 4.0, 8.0]),
        gradient_energy=np.diag([0.5, 1.0, 1.5]),
        free_energy_curvature=3.0,
    )
    rate = 2.0 * case.wave_number**2 * (
        3.0 + 0.5 * case.wave_number**2
    )

    assert solver.decay_rate[case.mode, 0, 0] == pytest.approx(
        rate, rel=1e-5
    )


def test_run_returns_fields_at_requested_times():
    case = SinusoidalCase()
    solver = LinearSpectralDiffusion(case.grid, mobility=1.0)
    fields = solver.run(case.initial_field(), [0.0, 0.1, 0.2])

    assert fields.shape == (3,) + case.grid.shape
    np.testing.assert_allclose(fields[0], case.initial_field(), atol=1e-6)


def test_zero_mode_is_conserved():
    case = SinusoidalCase()
    solver = LinearSpectralDiffusion(case.grid, mobility=1.0)
    field = np.ones(case.grid.shape) + case.initial_field()

    np.testing.assert_allclose(
        np.mean(solver.step(field, 0.1)), np.mean(field), atol=1e-6
    )


def test_mass_diffusion_step_reduces_a_sinusoidal_mode():
    case = SinusoidalCase()
    solver = MassDiffusion(
        case.grid,
        mobility=1.0,
        gradient_energy=5.0e-4,
        free_energy_derivative=lambda c: np.zeros_like(c),
    )

    field = case.initial_field()
    updated = solver.step(field, 1.0e-3)

    assert np.mean(updated) == pytest.approx(np.mean(field), abs=1e-8)
    assert case.amplitude_of(updated) < case.amplitude_of(field)


def test_interface_profile_relaxes_toward_tanh():
    case = InterfaceCase(
        center=0.5,
        width=0.08,
        left_state=-1.0,
        right_state=1.0,
    )
    solver = MassDiffusion(
        case.grid,
        mobility=1.0,
        gradient_energy=2.0e-3,
    )

    field = case.initial_field()
    relaxed = field.copy()
    for _ in range(100):
        relaxed = solver.step(relaxed, 1.0e-9)

    assert case.profile_error(relaxed) < case.profile_error(field)


def test_invalid_tensor_shape_raises():
    case = SinusoidalCase()
    with pytest.raises(ValueError, match="3 by 3"):
        LinearSpectralDiffusion(case.grid, mobility=np.ones((2, 2)))


def test_double_well_curvature_matches_finite_difference():
    c = np.array([-0.6, -0.1, 0.0, 0.3, 0.9])
    h = 1e-5
    finite_difference = (
        double_well_derivative(c + h, a=1.7, c_alpha=-0.8, c_beta=1.2)
        - double_well_derivative(c - h, a=1.7, c_alpha=-0.8, c_beta=1.2)
    ) / (2 * h)

    np.testing.assert_allclose(
        double_well_curvature(c, a=1.7, c_alpha=-0.8, c_beta=1.2),
        finite_difference,
        rtol=1e-4,
    )


def test_double_well_curvature_at_midpoint_is_minus_a_times_span_squared():
    # f''(midpoint) = -a * (c_beta - c_alpha)^2, the value used to
    # linearize the spinodal decay tests below.
    assert double_well_curvature(0.2, a=1.7, c_alpha=-0.8, c_beta=1.2) == pytest.approx(
        -1.7 * (1.2 - (-0.8)) ** 2
    )


def test_double_well_equilibrium_width_matches_first_integral():
    # For a=1, c_alpha=-1, c_beta=1 (half-span w=1): delta = sqrt(kappa/2).
    assert double_well_equilibrium_width(1.0, -1.0, 1.0, 0.02) == pytest.approx(
        np.sqrt(0.01)
    )


def test_negative_curvature_is_a_growing_spinodal_mode():
    # fick.pdf, "Sinusoidal interface": a short-wavelength perturbation
    # about the unstable midpoint of a double well still decays, because
    # the gradient-energy term stabilizes it, even though the bulk
    # curvature there is negative (spinodal).
    case = SinusoidalCase()
    solver = LinearSpectralDiffusion(
        case.grid, mobility=1.0, gradient_energy=0.5,
        free_energy_curvature=-4.0,
    )
    rate = case.wave_number**2 * (-4.0 + 0.5 * case.wave_number**2)
    assert rate > 0

    field = solver.step(case.initial_field(), 0.1)
    assert case.amplitude_of(field) == pytest.approx(
        case.expected_amplitude(0.1, rate), rel=1e-5
    )

    # with a weaker gradient term the same mode and curvature instead
    # grow (still spinodal, but no longer stabilized at this
    # wavelength), which the exact spectral stepper must also get right.
    unstable = LinearSpectralDiffusion(
        case.grid, mobility=1.0, gradient_energy=0.01,
        free_energy_curvature=-4.0,
    )
    unstable_rate = case.wave_number**2 * (-4.0 + 0.01 * case.wave_number**2)
    assert unstable_rate < 0

    grown = unstable.step(case.initial_field(), 0.01)
    assert case.amplitude_of(grown) == pytest.approx(
        case.expected_amplitude(0.01, unstable_rate), rel=1e-4
    )


def test_mass_diffusion_matches_linear_spinodal_decay_for_small_amplitude():
    # Cross-validates the nonlinear MassDiffusion stepper against the
    # exact LinearSpectralDiffusion solution in the small-amplitude
    # limit about the unstable midpoint (c=0) of the default double
    # well (c_alpha=-1, c_beta=1), where f''(0) = -4 * a (see
    # double_well_curvature). This is the same "sinusoidal interface
    # decaying amplitude" case as fick.pdf, generalized to the
    # nonlinear solver that actually produced it historically.
    grid = Grid(shape=(32, 32, 1))
    case = SinusoidalCase(grid=grid, amplitude=0.01)
    kappa = 0.05
    curvature = -4.0

    linear = LinearSpectralDiffusion(
        grid, mobility=1.0, gradient_energy=kappa,
        free_energy_curvature=curvature,
    )
    nonlinear = MassDiffusion(
        grid, mobility=1.0, gradient_energy=kappa,
        bulk_free_energy_coefficient=1.0, left_well=-1.0, right_well=1.0,
    )

    time_step, steps = 2.0e-7, 3000
    linear_field = linear.step(case.initial_field(), time_step * steps)

    nonlinear_field = case.initial_field()
    for _ in range(steps):
        nonlinear_field = nonlinear.step(nonlinear_field, time_step)

    assert case.amplitude_of(nonlinear_field) == pytest.approx(
        case.amplitude_of(linear_field), rel=5e-3
    )


def test_tanh_profile_is_a_chemical_potential_equilibrium():
    # fick.pdf, "Planar interface": the tanh profile width follows from
    # the first integral of kappa*c'' = f'(c). double_well_equilibrium_width
    # derives that width analytically; this checks the resulting profile
    # actually zeroes the solver's own chemical potential, independent of
    # any time integration.
    a, kappa = 1.0, 2.0e-3
    width = double_well_equilibrium_width(a, -1.0, 1.0, kappa)
    case = InterfaceCase(width=width, left_state=-1.0, right_state=1.0)
    solver = MassDiffusion(
        case.grid,
        mobility=1.0,
        gradient_energy=kappa,
        bulk_free_energy_coefficient=a,
        left_well=-1.0,
        right_well=1.0,
    )

    mu = np.asarray(solver.chemical_potential(case.tanh_profile())).reshape(-1)

    # A single interface on a periodic domain is not itself periodic (the
    # profile runs from -1 to +1 once, then must jump back), so exclude a
    # margin around the domain wrap where that mismatch, not the interface
    # model, dominates the residual.
    x = np.asarray(case.grid.x[0]).reshape(-1)
    margin = 10.0 * width
    interior = (x > margin) & (x < case.grid.lengths[0] - margin)

    assert np.max(np.abs(mu[interior])) < 0.1

    # an interface twice as wide as the analytic prediction should be a
    # much worse equilibrium, confirming the check has discriminating power
    wrong = InterfaceCase(width=2.0 * width, left_state=-1.0, right_state=1.0)
    wrong_mu = np.asarray(
        solver.chemical_potential(wrong.tanh_profile())
    ).reshape(-1)
    assert np.max(np.abs(wrong_mu[interior])) > 5.0 * np.max(np.abs(mu[interior]))