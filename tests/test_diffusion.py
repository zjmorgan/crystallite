import numpy as np
import pytest

from crystallite.diffusion import LinearSpectralDiffusion
from crystallite.verification import SinusoidalCase


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


def test_invalid_tensor_shape_raises():
    case = SinusoidalCase()
    with pytest.raises(ValueError, match="3 by 3"):
        LinearSpectralDiffusion(case.grid, mobility=np.ones((2, 2)))