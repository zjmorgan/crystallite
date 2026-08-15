import numpy as np
import pytest

from crystallite.backend import xp
from crystallite.phase import Grid


def test_default_construction():
    grid = Grid()
    assert grid.fft_axes == (-3, -2)
    assert grid.fft_shape == (256, 256)


def test_spacing():
    grid = Grid(shape=(4, 5, 6), lengths=(2.0, 5.0, 6.0))
    assert grid.spacing == pytest.approx((0.5, 1.0, 1.0))


def test_invalid_shape_length_raises():
    with pytest.raises(AssertionError):
        Grid(shape=(4, 4))


def test_invalid_lengths_length_raises():
    with pytest.raises(AssertionError):
        Grid(lengths=(1.0, 1.0))


def test_nonpositive_shape_raises():
    with pytest.raises(AssertionError):
        Grid(shape=(4, 0, 4))


def test_nonpositive_lengths_raises():
    with pytest.raises(AssertionError):
        Grid(lengths=(1.0, 0.0, 1.0))


def test_inactive_axis_is_excluded_from_fft():
    grid = Grid(shape=(8, 8, 1), lengths=(1.0, 1.0, 1.0))
    assert grid.fft_axes == (-3, -2)
    assert grid.fft_shape == (8, 8)
    assert grid.k[2].shape == (1, 1, 1)
    assert np.all(np.asarray(grid.k[2]) == 0)


def test_x_coordinates():
    grid = Grid(shape=(4, 3, 1), lengths=(2.0, 3.0, 1.0))
    expected_x0 = np.arange(4) * 0.5
    np.testing.assert_allclose(
        np.asarray(grid.x[0]).ravel(), expected_x0, atol=1e-6
    )


def test_k_frequencies_match_numpy_fft():
    shape = (4, 6, 8)
    lengths = (1.0, 1.0, 1.0)
    grid = Grid(shape=shape, lengths=lengths)

    dx = tuple(l / n for n, l in zip(shape, lengths))

    expected_k0 = 2 * np.pi * np.fft.fftfreq(shape[0], d=dx[0])
    expected_k1 = 2 * np.pi * np.fft.fftfreq(shape[1], d=dx[1])
    expected_k2 = 2 * np.pi * np.fft.rfftfreq(shape[2], d=dx[2])

    np.testing.assert_allclose(
        np.asarray(grid.k[0]).ravel(), expected_k0, atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(grid.k[1]).ravel(), expected_k1, atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(grid.k[2]).ravel(), expected_k2, atol=1e-4
    )


def test_k2_is_sum_of_squares():
    grid = Grid(shape=(4, 6, 8), lengths=(1.0, 1.0, 1.0))
    expected = grid.k[0] ** 2 + grid.k[1] ** 2 + grid.k[2] ** 2
    np.testing.assert_allclose(
        np.asarray(grid.k2), np.asarray(expected), atol=1e-6
    )


def test_inv_k2_is_reciprocal_away_from_dc():
    grid = Grid(shape=(4, 6, 8), lengths=(1.0, 1.0, 1.0))
    k2 = np.asarray(grid.k2)
    inv_k2 = np.asarray(grid.inv_k2)

    nonzero = k2 != 0
    np.testing.assert_allclose(
        inv_k2[nonzero], 1.0 / k2[nonzero], atol=1e-6
    )


def test_inv_k2_is_zero_at_dc():
    grid = Grid(shape=(4, 6, 8), lengths=(1.0, 1.0, 1.0))
    assert np.asarray(grid.k2)[0, 0, 0] == 0
    assert np.asarray(grid.inv_k2)[0, 0, 0] == 0
    assert not np.any(np.isnan(np.asarray(grid.inv_k2)))


def test_fft_ifft_roundtrip():
    grid = Grid(shape=(8, 8, 8), lengths=(1.0, 1.0, 1.0))
    real = xp.sin(2 * xp.pi * grid.x[0]) * xp.ones(
        grid.fft_shape, dtype=grid.real_dtype
    )

    recovered = grid.ifft(grid.fft(real))

    np.testing.assert_allclose(
        np.asarray(recovered), np.asarray(real), atol=1e-4
    )
