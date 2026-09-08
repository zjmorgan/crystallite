import numpy as np
import pytest

from crystallite.mixture import arithmetic, harmonic


def test_arithmetic_mixes_scalar_properties():
    assert arithmetic(2.0, 10.0, 0.25) == pytest.approx(8.0)


def test_arithmetic_mixes_anisotropic_properties_elementwise():
    a = np.diag([2.0, 4.0, 8.0])
    b = np.diag([10.0, 20.0, 40.0])

    np.testing.assert_allclose(arithmetic(a, b, 0.25), np.diag([8.0, 16.0, 32.0]))


def test_harmonic_mixes_scalar_properties():
    assert harmonic(2.0, 10.0, 0.25) == pytest.approx(5.0)


def test_harmonic_mixes_anisotropic_properties_by_matrix_inverse():
    a = np.diag([2.0, 4.0, 8.0])
    b = np.diag([10.0, 20.0, 40.0])
    expected = np.diag([1.0 / (0.25 / 2.0 + 0.75 / 10.0),
                        1.0 / (0.25 / 4.0 + 0.75 / 20.0),
                        1.0 / (0.25 / 8.0 + 0.75 / 40.0)])

    np.testing.assert_allclose(harmonic(a, b, 0.25), expected)


def test_harmonic_rejects_nonsquare_tensor_properties():
    with pytest.raises(ValueError, match="square matrices"):
        harmonic(np.ones((2, 3)), np.ones((2, 3)), 0.5)