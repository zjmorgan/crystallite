import numpy as np
import pytest

from crystallite.backend import xp
from crystallite.phase import Grid
from crystallite.spectral.short_range import DifferentialOperators


@pytest.fixture
def grid():
    return Grid(shape=(8, 8, 8), lengths=(1.0, 1.0, 1.0))


@pytest.fixture
def ops(grid):
    return DifferentialOperators(grid)


@pytest.fixture
def fourier_shape(grid):
    """Shape of an array actually produced by `grid.fft`."""
    return tuple(k.shape[axis] for axis, k in enumerate(grid.k))


def _random_field(shape, rng):
    return (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ).astype(np.complex128)


def test_grad_inserts_length_3_axis(ops, fourier_shape):
    f = _random_field(fourier_shape, np.random.default_rng(0))
    grad_f = ops.grad(f)
    assert grad_f.shape == (3,) + fourier_shape


def test_grad_matches_analytic_derivative(grid, ops):
    ones = xp.ones(grid.fft_shape, dtype=grid.real_dtype)
    x0 = grid.x[0].astype(grid.real_dtype) * ones
    field = np.sin(2 * np.pi * np.asarray(x0))

    f_hat = grid.fft(field)
    grad_hat = ops.grad(f_hat)
    grad_real = grid.ifft(grad_hat)

    expected_dx = 2 * np.pi * np.cos(2 * np.pi * np.asarray(x0))

    np.testing.assert_allclose(
        np.asarray(grad_real[0]), expected_dx, atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(grad_real[1]), np.zeros_like(expected_dx), atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(grad_real[2]), np.zeros_like(expected_dx), atol=1e-4
    )


def test_curl_of_gradient_is_zero(ops, fourier_shape):
    rng = np.random.default_rng(1)
    f = _random_field(fourier_shape, rng)

    grad_f = ops.grad(f)
    curl_grad_f = ops.curl(grad_f)

    np.testing.assert_allclose(
        np.asarray(curl_grad_f), 0.0, atol=1e-6
    )


def test_div_of_curl_is_zero(ops, fourier_shape):
    rng = np.random.default_rng(2)
    w = _random_field((3,) + fourier_shape, rng)

    curl_w = ops.curl(w)
    div_curl_w = ops.div(curl_w)

    np.testing.assert_allclose(
        np.asarray(div_curl_w), 0.0, atol=1e-6
    )


def test_div_grad_equals_laplacian(ops, fourier_shape):
    rng = np.random.default_rng(3)
    f = _random_field(fourier_shape, rng)

    div_grad_f = ops.div(ops.grad(f))
    lap_f = ops.laplacian(f)

    np.testing.assert_allclose(
        np.asarray(div_grad_f), np.asarray(lap_f), atol=1e-6
    )


def test_div_raises_without_tensor_axis(ops, fourier_shape):
    rng = np.random.default_rng(4)
    f = _random_field(fourier_shape, rng)

    with pytest.raises(ValueError):
        ops.div(f)


def test_curl_raises_without_tensor_axis(ops, fourier_shape):
    rng = np.random.default_rng(5)
    f = _random_field(fourier_shape, rng)

    with pytest.raises(ValueError):
        ops.curl(f)


def test_div_raises_for_wrong_tensor_length(ops, fourier_shape):
    rng = np.random.default_rng(6)
    w = _random_field((2,) + fourier_shape, rng)

    with pytest.raises(ValueError):
        ops.div(w)
