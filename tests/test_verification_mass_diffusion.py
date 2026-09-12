import numpy as np
import pytest

from crystallite.grid import Grid
from crystallite.verification import SinusoidalCase


def test_sinusoidal_case_matches_historical_default_profile():
    case = SinusoidalCase()

    np.testing.assert_allclose(
        np.asarray(case.initial_field()),
        np.broadcast_to(
            0.125 * np.sin(4 * np.pi * np.asarray(case.grid.x[0])),
            case.grid.shape,
        ),
        atol=1e-6,
    )


def test_sinusoidal_case_recovers_amplitude():
    case = SinusoidalCase()
    field = np.broadcast_to(
        0.75 * np.sin(case.wave_number * np.asarray(case.grid.x[0])),
        case.grid.shape,
    )

    assert case.amplitude_of(field) == pytest.approx(0.75, abs=1e-6)


def test_sinusoidal_case_supports_256_by_256_grid():
    case = SinusoidalCase(grid=Grid(shape=(256, 256, 1)))

    assert case.initial_field().shape == (256, 256, 1)


def test_sinusoidal_case_expected_decay():
    case = SinusoidalCase()

    np.testing.assert_allclose(
        np.asarray(case.expected_amplitude([0.0, 2.0], 0.5)),
        [0.125, 0.125 * np.exp(-1.0)],
    )


def test_sinusoidal_case_rejects_three_dimensional_grid():
    with pytest.raises(ValueError, match="two-dimensional"):
        SinusoidalCase(grid=Grid(shape=(8, 8, 8)))