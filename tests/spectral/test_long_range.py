import numpy as np
import pytest

from crystallite.backend import xp
from crystallite.phase import Grid
from crystallite.spectral.long_range import CoulombOperator, GreenOperator
from crystallite.spectral.short_range import DifferentialOperators


@pytest.fixture
def grid():
    return Grid(shape=(8, 8, 8), lengths=(1.0, 1.0, 1.0))


@pytest.fixture
def ops(grid):
    return DifferentialOperators(grid)


@pytest.fixture
def coulomb(grid):
    return CoulombOperator(grid)


@pytest.fixture
def fourier_shape(grid):
    """Shape of an array actually produced by `grid.fft`."""
    return tuple(k.shape[axis] for axis, k in enumerate(grid.k))


def _random_field(shape, rng):
    return (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ).astype(np.complex128)


def _manual_potential(grid, source):
    k2mag = np.asarray(grid.k2)
    safe_k2mag = np.where(k2mag == 0, 1.0, k2mag)
    return np.where(k2mag == 0, 0.0, -np.asarray(source) / safe_k2mag)


# -- generic potential()/field() behavior, independent of what the
#    source term physically represents --


def test_potential_shape(coulomb, fourier_shape):
    source = _random_field(fourier_shape, np.random.default_rng(0))
    phi = coulomb.potential(source)
    assert phi.shape == source.shape


def test_potential_solves_poisson_away_from_dc(coulomb, ops, fourier_shape):
    source = _random_field(fourier_shape, np.random.default_rng(1))
    phi = coulomb.potential(source)
    lap = np.asarray(ops.laplacian(phi))

    # The k=0 mode of the Poisson equation is degenerate (phi is
    # defined to be 0 there regardless of source), so only check
    # away from DC.
    lap[0, 0, 0] = 0
    source = np.asarray(source).copy()
    source[0, 0, 0] = 0

    np.testing.assert_allclose(lap, source, atol=1e-6)


def test_potential_dc_mode_has_no_nan(coulomb, fourier_shape):
    source = _random_field(fourier_shape, np.random.default_rng(2))
    phi = coulomb.potential(source)

    assert not np.any(np.isnan(np.asarray(phi)))
    assert np.asarray(phi)[0, 0, 0] == 0


def test_field_shape(coulomb, fourier_shape):
    source = _random_field(fourier_shape, np.random.default_rng(3))
    field = coulomb.field(source)
    assert field.shape == (3,) + fourier_shape


def test_field_is_curl_free(coulomb, ops, fourier_shape):
    source = _random_field(fourier_shape, np.random.default_rng(4))
    field = coulomb.field(source)

    np.testing.assert_allclose(np.asarray(ops.curl(field)), 0.0, atol=1e-6)


def test_field_has_no_nan_at_dc(coulomb, fourier_shape):
    source = _random_field(fourier_shape, np.random.default_rng(5))
    field = coulomb.field(source)

    assert not np.any(np.isnan(np.asarray(field)))
    np.testing.assert_array_equal(np.asarray(field[:, 0, 0, 0]), 0.0)


def test_field_matches_manual_formula(coulomb, grid, fourier_shape):
    source = _random_field(fourier_shape, np.random.default_rng(6))

    k0, k1, k2 = grid.k
    phi = _manual_potential(grid, source)
    expected = -np.stack(
        (1j * k0 * phi, 1j * k1 * phi, 1j * k2 * phi), axis=-4
    )

    np.testing.assert_allclose(
        np.asarray(coulomb.field(source)), expected, atol=1e-6
    )


def test_external_field_recovers_uniform_value_in_real_space(
    coulomb, grid, fourier_shape
):
    # A zero source, so the only contribution is the external field.
    source = xp.zeros(fourier_shape, dtype=grid.complex_dtype)
    external = [0.3, -1.2, 2.5]

    field_hat = coulomb.field(source, uniform_field=external)
    field_real = grid.ifft(field_hat)

    for axis, value in enumerate(external):
        np.testing.assert_allclose(
            np.asarray(field_real[axis]),
            np.full(grid.fft_shape, value),
            atol=1e-4,
        )


def test_external_field_superposes_on_source_field(
    coulomb, grid, fourier_shape
):
    source = _random_field(fourier_shape, np.random.default_rng(13))
    external = [1.0, 0.0, -0.5]

    n_points = 1
    for n in grid.fft_shape:
        n_points *= n

    plain = np.asarray(coulomb.field(source))
    with_external = np.asarray(coulomb.field(source, uniform_field=external))

    diff = with_external - plain
    expected_diff = np.zeros_like(diff)
    expected_diff[:, 0, 0, 0] = np.asarray(external) * n_points

    np.testing.assert_allclose(diff, expected_diff, atol=1e-6)


def test_field_without_external_is_unchanged(coulomb, fourier_shape):
    source = _random_field(fourier_shape, np.random.default_rng(14))

    np.testing.assert_array_equal(
        np.asarray(coulomb.field(source)),
        np.asarray(coulomb.field(source, uniform_field=None)),
    )


def test_uniform_field_gradient_recovers_linear_field_in_real_space(
    coulomb, grid, fourier_shape
):
    # A zero source, so the only contribution is the imposed gradient,
    # built directly in reciprocal space (see _add_uniform_gradient).
    source = xp.zeros(fourier_shape, dtype=grid.complex_dtype)
    gradient = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    )

    field_hat = coulomb.field(source, uniform_field_gradient=gradient)
    field_real = np.asarray(grid.ifft(field_hat))

    x0, x1, x2 = (np.asarray(x) for x in grid.x)
    for i in range(3):
        expected = np.broadcast_to(
            gradient[i, 0] * x0 + gradient[i, 1] * x1 + gradient[i, 2] * x2,
            grid.fft_shape,
        )
        np.testing.assert_allclose(field_real[i], expected, atol=1e-4)


# -- callers build the source term; these reconstruct the two known
#    physical cases (charge-charge and dipole-dipole) to confirm the
#    generalized class still reproduces the expected physics --


def test_charge_charge_via_negated_density(grid, coulomb):
    ones = xp.ones(grid.fft_shape, dtype=grid.real_dtype)
    x0 = grid.x[0].astype(grid.real_dtype) * ones
    rho = np.cos(2 * np.pi * np.asarray(x0))

    rho_hat = grid.fft(rho)
    e_hat = coulomb.field(-rho_hat)
    e_real = grid.ifft(e_hat)

    expected_ex = np.sin(2 * np.pi * np.asarray(x0)) / (2 * np.pi)

    np.testing.assert_allclose(
        np.asarray(e_real[0]), expected_ex, atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(e_real[1]), np.zeros_like(expected_ex), atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(e_real[2]), np.zeros_like(expected_ex), atol=1e-4
    )


def test_dipole_dipole_via_divergence_source(coulomb, ops, fourier_shape):
    m = _random_field((3,) + fourier_shape, np.random.default_rng(7))

    h = coulomb.field(ops.div(m))

    # Independent closed form: h_hat = -k (k . m_hat) / k^2.
    k0, k1, k2 = coulomb.grid.k
    k_dot_m = np.asarray(k0 * m[0] + k1 * m[1] + k2 * m[2])

    k2mag = np.asarray(coulomb.grid.k2)
    safe_k2mag = np.where(k2mag == 0, 1.0, k2mag)
    inv_k2mag = np.where(k2mag == 0, 0.0, 1.0 / safe_k2mag)

    expected = -np.stack(
        (
            np.asarray(k0) * k_dot_m * inv_k2mag,
            np.asarray(k1) * k_dot_m * inv_k2mag,
            np.asarray(k2) * k_dot_m * inv_k2mag,
        ),
        axis=-4,
    )

    np.testing.assert_allclose(np.asarray(h), expected, atol=1e-6)


# -- GreenOperator: conduction (rank-2 medium) and elasticity
#    (rank-4 medium) share the same acoustic-tensor-inversion core --


def _isotropic_stiffness(lam, mu):
    delta = np.eye(3)
    return lam * np.einsum("ij,kl->ijkl", delta, delta) + mu * (
        np.einsum("ik,jl->ijkl", delta, delta)
        + np.einsum("il,jk->ijkl", delta, delta)
    )


def _isotropic_green_manual(grid, lam, mu):
    k0, k1, k2 = grid.k
    shape = grid.k2.shape
    k = np.stack(
        [np.broadcast_to(np.asarray(x), shape) for x in (k0, k1, k2)],
        axis=0,
    )

    k2mag = np.asarray(grid.k2)
    safe_k2mag = np.where(k2mag == 0, 1.0, k2mag)
    inv_k2mag = np.where(k2mag == 0, 0.0, 1.0 / safe_k2mag)

    delta = np.eye(3).reshape(3, 3, 1, 1, 1)
    kk = np.einsum("i...,j...->ij...", k, k)
    coeff = (lam + mu) / (lam + 2 * mu)

    return (delta - coeff * kk * inv_k2mag) * inv_k2mag / mu


@pytest.fixture
def isotropic_medium():
    return _isotropic_stiffness(lam=1.5, mu=0.7)


@pytest.fixture
def elastic_green(grid, isotropic_medium):
    return GreenOperator(grid, isotropic_medium)


def test_green_operator_rejects_bad_medium_shape(grid):
    with pytest.raises(ValueError):
        GreenOperator(grid, xp.zeros((3, 4)))


def test_conduction_identity_medium_matches_grid_inv_k2(grid):
    green = GreenOperator(grid, xp.eye(3, dtype=grid.real_dtype))
    np.testing.assert_allclose(
        np.asarray(green.tensor()), np.asarray(grid.inv_k2), atol=1e-6
    )


def test_conduction_matches_manual_anisotropic_formula(grid, fourier_shape):
    kappa = xp.asarray([[2.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    green = GreenOperator(grid, kappa)

    k0, k1, k2 = grid.k
    a = 2 * np.asarray(k0) ** 2 + np.asarray(k1) ** 2 + np.asarray(k2) ** 2
    safe_a = np.where(a == 0, 1.0, a)
    expected_g = np.where(a == 0, 0.0, 1.0 / safe_a)

    np.testing.assert_allclose(
        np.asarray(green.tensor()), expected_g, atol=1e-6
    )

    source = _random_field(fourier_shape, np.random.default_rng(8))
    np.testing.assert_allclose(
        np.asarray(green.potential(source)),
        expected_g * np.asarray(source),
        atol=1e-6,
    )


def test_conduction_dc_mode_has_no_nan(grid, fourier_shape):
    green = GreenOperator(grid, xp.eye(3, dtype=grid.real_dtype))
    source = _random_field(fourier_shape, np.random.default_rng(9))
    result = green.potential(source)

    assert not np.any(np.isnan(np.asarray(result)))
    assert np.asarray(result)[0, 0, 0] == 0


def test_elastic_tensor_shape(elastic_green, fourier_shape):
    assert elastic_green.tensor().shape == (3, 3) + fourier_shape


def test_elastic_tensor_matches_isotropic_closed_form(
    grid, elastic_green
):
    expected = _isotropic_green_manual(grid, lam=1.5, mu=0.7)
    np.testing.assert_allclose(
        np.asarray(elastic_green.tensor()), expected, atol=1e-5
    )


def test_elastic_tensor_is_inverse_of_acoustic_tensor(
    grid, isotropic_medium, elastic_green
):
    k = np.stack(
        [
            np.broadcast_to(np.asarray(x), grid.k2.shape)
            for x in grid.k
        ],
        axis=0,
    )
    a = np.einsum("ijkl,j...,l...->ik...", isotropic_medium, k, k)
    g = np.asarray(elastic_green.tensor())

    identity = np.einsum("ik...,kj...->ij...", a, g)

    k2mag = np.asarray(grid.k2)
    identity[..., k2mag == 0] = np.eye(3).reshape(3, 3, 1)

    expected = np.broadcast_to(
        np.eye(3).reshape(3, 3, 1, 1, 1), identity.shape
    )
    np.testing.assert_allclose(identity, expected, atol=1e-5)


def test_elastic_potential_shape(elastic_green, fourier_shape):
    f = _random_field((3,) + fourier_shape, np.random.default_rng(10))
    u = elastic_green.potential(f)
    assert u.shape == f.shape


def test_elastic_potential_solves_acoustic_equation_away_from_dc(
    grid, isotropic_medium, elastic_green, fourier_shape
):
    f = _random_field((3,) + fourier_shape, np.random.default_rng(11))
    u = np.asarray(elastic_green.potential(f))

    k = np.stack(
        [
            np.broadcast_to(np.asarray(x), grid.k2.shape)
            for x in grid.k
        ],
        axis=0,
    )
    a = np.einsum("ijkl,j...,l...->ik...", isotropic_medium, k, k)
    recovered = np.einsum("ik...,k...->i...", a, u)

    recovered[:, 0, 0, 0] = 0
    f = np.asarray(f).copy()
    f[:, 0, 0, 0] = 0

    np.testing.assert_allclose(recovered, f, atol=1e-4)


def test_elastic_potential_has_no_nan_at_dc(elastic_green, fourier_shape):
    f = _random_field((3,) + fourier_shape, np.random.default_rng(12))
    u = elastic_green.potential(f)

    assert not np.any(np.isnan(np.asarray(u)))
    np.testing.assert_array_equal(np.asarray(u[:, 0, 0, 0]), 0.0)


def _random_symmetric_tensor_field(fourier_shape, rng):
    a = _random_field((3, 3) + fourier_shape, rng)
    return 0.5 * (a + np.swapaxes(a, 0, 1))


# -- field(): strain response to a body force, parallel to
#    CoulombOperator.potential()/field(), including uniform
#    far-field/gradient superposition --


def test_elastic_field_shape(elastic_green, fourier_shape):
    f = _random_field((3,) + fourier_shape, np.random.default_rng(19))
    eps = elastic_green.field(f)
    assert eps.shape == (3, 3) + fourier_shape


def test_elastic_field_is_symmetric(elastic_green, fourier_shape):
    f = _random_field((3,) + fourier_shape, np.random.default_rng(20))
    eps = np.asarray(elastic_green.field(f))

    np.testing.assert_allclose(eps, np.swapaxes(eps, 0, 1), atol=1e-6)


def test_elastic_field_matches_manual_symmetrized_gradient(
    elastic_green, grid, fourier_shape
):
    f = _random_field((3,) + fourier_shape, np.random.default_rng(21))

    k = np.stack(
        [
            np.broadcast_to(np.asarray(x), grid.k2.shape)
            for x in grid.k
        ],
        axis=0,
    )
    u = np.asarray(elastic_green.potential(f))
    # eps_kl = 0.5 * (i k_l u_k + i k_k u_l)
    grad_u = 1j * np.einsum("l...,k...->lk...", k, u)
    expected = 0.5 * (grad_u + np.swapaxes(grad_u, 0, 1))

    np.testing.assert_allclose(
        np.asarray(elastic_green.field(f)), expected, atol=1e-6
    )


def test_elastic_field_dc_mode_is_zero_without_uniform_field(
    elastic_green, fourier_shape
):
    f = _random_field((3,) + fourier_shape, np.random.default_rng(22))
    eps = elastic_green.field(f)

    assert not np.any(np.isnan(np.asarray(eps)))
    np.testing.assert_array_equal(np.asarray(eps[:, :, 0, 0, 0]), 0.0)


def test_elastic_field_uniform_field_recovers_uniform_value_in_real_space(
    elastic_green, grid, fourier_shape
):
    f = xp.zeros((3,) + fourier_shape, dtype=grid.complex_dtype)
    uniform_field = [
        [0.1, 0.2, 0.0], [0.2, -0.3, 0.05], [0.0, 0.05, 0.15]
    ]

    eps_hat = elastic_green.field(f, uniform_field=uniform_field)
    eps_real = grid.ifft(eps_hat)

    for i in range(3):
        for j in range(3):
            np.testing.assert_allclose(
                np.asarray(eps_real[i, j]),
                np.full(grid.fft_shape, uniform_field[i][j]),
                atol=1e-4,
            )


def test_elastic_field_uniform_field_superposes(
    elastic_green, grid, fourier_shape
):
    f = _random_field((3,) + fourier_shape, np.random.default_rng(23))
    uniform_field = [[0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, 0.0]]

    n_points = 1
    for n in grid.fft_shape:
        n_points *= n

    plain = np.asarray(elastic_green.field(f))
    with_uniform = np.asarray(
        elastic_green.field(f, uniform_field=uniform_field)
    )

    diff = with_uniform - plain
    expected_diff = np.zeros_like(diff)
    expected_diff[:, :, 0, 0, 0] = np.asarray(uniform_field) * n_points

    np.testing.assert_allclose(diff, expected_diff, atol=1e-6)


def test_elastic_field_uniform_field_gradient_recovers_linear_field(
    elastic_green, grid, fourier_shape
):
    f = xp.zeros((3,) + fourier_shape, dtype=grid.complex_dtype)
    gradient = np.random.default_rng(28).normal(size=(3, 3, 3))

    eps_hat = elastic_green.field(f, uniform_field_gradient=gradient)
    eps_real = np.asarray(grid.ifft(eps_hat))

    x0, x1, x2 = (np.asarray(x) for x in grid.x)
    for i in range(3):
        for j in range(3):
            expected = np.broadcast_to(
                gradient[i, j, 0] * x0
                + gradient[i, j, 1] * x1
                + gradient[i, j, 2] * x2,
                grid.fft_shape,
            )
            np.testing.assert_allclose(
                eps_real[i, j], expected, atol=1e-3
            )


def test_elastic_field_uniform_field_from_stress_via_compliance(
    elastic_green, isotropic_medium, grid, fourier_shape
):
    # Stress-controlled far field: convert an applied macroscopic
    # stress to the equivalent strain via apply_compliance, then feed
    # it to field() as uniform_field. Checked by recovering the
    # applied stress via a manual C:strain (Hooke's law) contraction.
    f = xp.zeros((3,) + fourier_shape, dtype=grid.complex_dtype)
    sigma_appl = np.asarray(
        [[1.0, 0.2, 0.0], [0.2, -0.5, 0.0], [0.0, 0.0, 0.3]]
    )

    eps_bar = elastic_green.apply_compliance(sigma_appl)
    eps_hat = elastic_green.field(f, uniform_field=eps_bar)
    eps_real = np.asarray(grid.ifft(eps_hat))

    recovered_stress_real = np.asarray(
        grid.ifft(
            np.einsum("ijkl,kl...->ij...", isotropic_medium, eps_hat)
        )
    )
    for i in range(3):
        for j in range(3):
            np.testing.assert_allclose(
                recovered_stress_real[i, j],
                np.full(grid.fft_shape, sigma_appl[i, j]),
                atol=1e-4,
            )


# -- compliance() / apply_compliance(): the reference medium's
#    functional inverse, converting a far-field stress/flux into the
#    equivalent far-field strain/gradient --


def _isotropic_compliance_manual(lam, mu):
    delta = np.eye(3)
    E = mu * (3 * lam + 2 * mu) / (lam + mu)
    nu = lam / (2 * (lam + mu))

    return ((1 + nu) / E) * 0.5 * (
        np.einsum("ik,jl->ijkl", delta, delta)
        + np.einsum("il,jk->ijkl", delta, delta)
    ) - (nu / E) * np.einsum("ij,kl->ijkl", delta, delta)


def _cubic_stiffness(c11, c12, c44):
    stiffness = _isotropic_stiffness(lam=c12, mu=c44)
    correction = c11 - c12 - 2 * c44
    for m in range(3):
        stiffness[m, m, m, m] += correction
    return stiffness


def test_conduction_compliance_is_matrix_inverse(grid):
    kappa = xp.asarray([[2.0, 0, 0], [0, 1.0, 0], [0, 0, 4.0]])
    green = GreenOperator(grid, kappa)

    np.testing.assert_allclose(
        np.asarray(green.compliance()),
        np.linalg.inv(np.asarray(kappa)),
        atol=1e-6,
    )


def test_elastic_compliance_matches_isotropic_closed_form(elastic_green):
    expected = _isotropic_compliance_manual(lam=1.5, mu=0.7)
    np.testing.assert_allclose(
        np.asarray(elastic_green.compliance()), expected, atol=1e-6
    )


@pytest.mark.parametrize(
    "medium",
    [
        _isotropic_stiffness(lam=1.5, mu=0.7),
        _cubic_stiffness(c11=3.0, c12=1.2, c44=0.8),
    ],
    ids=["isotropic", "cubic"],
)
def test_elastic_compliance_inverts_on_symmetric_subspace(grid, medium):
    green = GreenOperator(grid, medium)
    s = np.asarray(green.compliance())

    # S : C is the identity operator on symmetric tensors, i.e. the
    # symmetrizer 0.5 * (delta_im delta_jn + delta_in delta_jm), not
    # the full rank-4 identity delta_im delta_jn.
    product = np.einsum("ijkl,klmn->ijmn", s, medium)

    delta = np.eye(3)
    expected = 0.5 * (
        np.einsum("im,jn->ijmn", delta, delta)
        + np.einsum("in,jm->ijmn", delta, delta)
    )

    np.testing.assert_allclose(product, expected, atol=1e-6)


def test_conduction_apply_compliance_matches_manual(grid):
    kappa = xp.asarray([[2.0, 0, 0], [0, 1.0, 0], [0, 0, 4.0]])
    green = GreenOperator(grid, kappa)
    flux = np.asarray([1.0, -2.0, 0.5])

    expected = np.linalg.inv(np.asarray(kappa)) @ flux

    np.testing.assert_allclose(
        np.asarray(green.apply_compliance(flux)), expected, atol=1e-6
    )


def test_elastic_apply_compliance_matches_manual(
    elastic_green, isotropic_medium
):
    stress = np.asarray(
        [[1.0, 0.2, 0.0], [0.2, -0.5, 0.1], [0.0, 0.1, 0.3]]
    )
    expected = _isotropic_compliance_manual(lam=1.5, mu=0.7)
    expected = np.einsum("ijkl,kl->ij", expected, stress)

    np.testing.assert_allclose(
        np.asarray(elastic_green.apply_compliance(stress)),
        expected,
        atol=1e-6,
    )


def test_elastic_apply_compliance_round_trips_through_stress(
    elastic_green, isotropic_medium
):
    stress = np.asarray(
        [[1.0, 0.2, 0.0], [0.2, -0.5, 0.1], [0.0, 0.1, 0.3]]
    )
    strain = elastic_green.apply_compliance(stress)
    recovered = np.einsum("ijkl,kl->ij", isotropic_medium, np.asarray(strain))

    np.testing.assert_allclose(recovered, stress, atol=1e-6)


def test_conduction_apply_compliance_round_trips_through_kappa(grid):
    kappa = xp.asarray([[2.0, 0, 0], [0, 1.0, 0], [0, 0, 4.0]])
    green = GreenOperator(grid, kappa)
    flux = np.asarray([1.0, -2.0, 0.5])

    gradient = green.apply_compliance(flux)
    recovered = np.asarray(kappa) @ np.asarray(gradient)

    np.testing.assert_allclose(recovered, flux, atol=1e-6)


def test_elastic_apply_compliance_broadcasts_over_gradient_direction(
    elastic_green, isotropic_medium
):
    # A far-field *gradient* of stress has the same (i, j) tensor
    # shape as a uniform one, plus a trailing spatial-direction axis --
    # apply_compliance should apply compliance independently to each
    # direction slice, matching apply_compliance called per slice.
    s = _isotropic_compliance_manual(lam=1.5, mu=0.7)
    stress_gradient = np.random.default_rng(30).normal(size=(3, 3, 3))

    result = np.asarray(elastic_green.apply_compliance(stress_gradient))
    for d in range(3):
        expected_d = np.einsum("ijkl,kl->ij", s, stress_gradient[..., d])
        np.testing.assert_allclose(
            result[..., d], expected_d, atol=1e-6
        )


