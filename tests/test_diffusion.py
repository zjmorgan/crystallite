import numpy as np
import pytest

from crystallite.diffusion import LinearSpectralDiffusion, MassDiffusion
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