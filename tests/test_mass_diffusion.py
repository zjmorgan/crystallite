import numpy as np
import pytest

from crystallite.mass_diffusion import (
    ChemicalFreeEnergy,
    GradientEnergy,
    LinearSpectralDiffusion,
    MassDiffusion,
    cahn_hilliard_period_range,
    cahn_hilliard_periodic_profile,
    double_well_curvature,
    double_well_derivative,
    double_well_equilibrium_width,
    chemical_free_energy,
    chemical_free_energy_derivative,
)
from crystallite.mobility import BinaryMobility
from crystallite.mobility import mobility as composition_mobility
from crystallite.grid import Grid
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


def test_cahn_hilliard_periodic_profile_satisfies_the_ode():
    # A single tanh interface is only a valid equilibrium on an infinite
    # domain -- a periodic domain needs a kink-antikink pair, whose exact
    # equilibrium is a Jacobi elliptic sn profile. Check it directly
    # against kappa*c'' = f'(c) via finite differences (periodic BCs).
    a, c_alpha, c_beta, kappa = 1.0, -1.0, 1.0, 2.0e-3
    length = 1.0
    x = np.linspace(0.0, length, 4000, endpoint=False)
    dx = x[1] - x[0]

    profile = np.asarray(
        cahn_hilliard_periodic_profile(
            x, length, a, c_alpha, c_beta, kappa, center=0.5 * length
        )
    )
    second_derivative = (np.roll(profile, -1) - 2 * profile + np.roll(profile, 1)) / dx**2
    residual = kappa * second_derivative - double_well_derivative(
        profile, a=a, c_alpha=c_alpha, c_beta=c_beta
    )

    assert np.max(np.abs(residual)) < 1e-3


def test_cahn_hilliard_periodic_profile_is_exactly_periodic():
    a, c_alpha, c_beta, kappa = 1.0, -1.0, 1.0, 2.0e-3
    length = 1.0
    x = np.array([0.1, 0.37, 0.6])

    profile = np.asarray(
        cahn_hilliard_periodic_profile(x, length, a, c_alpha, c_beta, kappa)
    )
    wrapped = np.asarray(
        cahn_hilliard_periodic_profile(x + length, length, a, c_alpha, c_beta, kappa)
    )

    np.testing.assert_allclose(profile, wrapped, atol=1e-10)


def test_cahn_hilliard_periodic_profile_reduces_to_two_tanh_interfaces():
    # Well past the minimum period, the kink and antikink are far apart
    # and each looks locally like the infinite-domain tanh solution. (Not
    # pushed further: representing the elliptic modulus this close to 1
    # -- e.g. for a >~100x separation -- needs more precision than
    # float64 offers for 1 - m directly.)
    a, c_alpha, c_beta, kappa = 1.0, -1.0, 1.0, 2.0e-3
    width = double_well_equilibrium_width(a, c_alpha, c_beta, kappa)
    length = 40.0 * width

    x = np.linspace(-0.5 * length, 0.5 * length, 4001)
    profile = np.asarray(
        cahn_hilliard_periodic_profile(x, length, a, c_alpha, c_beta, kappa, center=0.0)
    )
    near_kink = np.abs(x) < 10 * width
    tanh_reference = np.tanh(x[near_kink] / width)

    np.testing.assert_allclose(profile[near_kink], tanh_reference, atol=1e-3)


def test_cahn_hilliard_periodic_profile_rejects_too_short_a_period():
    a, c_alpha, c_beta, kappa = 1.0, -1.0, 1.0, 2.0e-3
    minimum = cahn_hilliard_period_range(a, c_alpha, c_beta, kappa)
    with pytest.raises(ValueError, match="length"):
        cahn_hilliard_periodic_profile(
            np.array([0.0]), 0.5 * minimum, a, c_alpha, c_beta, kappa
        )


def test_cahn_hilliard_periodic_profile_is_a_chemical_potential_equilibrium():
    # Unlike a single tanh evaluated on a periodic domain (which needs a
    # margin excluded around the domain wrap to hide the mismatch there),
    # the exact periodic profile should zero the chemical potential
    # everywhere, with no exclusion needed.
    a, c_alpha, c_beta, kappa = 1.0, -1.0, 1.0, 2.0e-3
    grid = Grid(shape=(256, 1, 1), lengths=(1.0, 1.0, 1.0))
    x = np.asarray(grid.x[0]).reshape(-1)
    length = grid.lengths[0]

    profile = cahn_hilliard_periodic_profile(
        x, length, a, c_alpha, c_beta, kappa, center=0.5 * length
    )
    solver = MassDiffusion(
        grid, mobility=1.0, gradient_energy=kappa,
        bulk_free_energy_coefficient=a, left_well=c_alpha, right_well=c_beta,
    )
    mu = np.asarray(solver.chemical_potential(np.asarray(profile).reshape(grid.shape)))

    assert np.max(np.abs(mu)) < 1e-4


def test_negative_curvature_is_a_growing_spinodal_mode():
    # A short-wavelength perturbation about the unstable midpoint of a
    # double well still decays, because the gradient-energy term
    # stabilizes it, even though the bulk curvature there is negative
    # (spinodal).
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
    # double_well_curvature). This is the same sinusoidal-interface
    # decaying-amplitude case, generalized to the nonlinear solver that
    # actually produced it historically.
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
    # The tanh profile width follows from the first integral of
    # kappa*c'' = f'(c). double_well_equilibrium_width
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


def test_callable_mobility_matches_equivalent_constant():
    # A callable that returns a spatially uniform field must reproduce the
    # plain-constant fast path, up to the extra FFT/IFFT round trip it
    # unavoidably takes to evaluate itself in real space: an odd-order
    # spectral derivative leaves an ill-defined (implicitly imaginary)
    # Nyquist-frequency component that a real-space round trip resolves
    # differently than staying purely in Fourier space -- an intrinsic,
    # negligible-amplitude spectral-differentiation artifact, not a
    # mobility-model difference, hence the small absolute tolerance.
    case = SinusoidalCase()
    field = case.initial_field()

    constant = MassDiffusion(case.grid, mobility=2.0, gradient_energy=5.0e-4)
    as_callable = MassDiffusion(
        case.grid,
        mobility=lambda c: np.full_like(c, 2.0),
        gradient_energy=5.0e-4,
    )

    np.testing.assert_allclose(
        as_callable.step(field, 1.0e-3),
        constant.step(field, 1.0e-3),
        rtol=1e-3,
        atol=2e-4,
    )


def test_callable_gradient_energy_matches_equivalent_constant():
    case = SinusoidalCase()
    field = case.initial_field()

    constant = MassDiffusion(case.grid, mobility=1.0, gradient_energy=5.0e-4)
    as_callable = MassDiffusion(
        case.grid,
        mobility=1.0,
        gradient_energy=lambda c: np.full_like(c, 5.0e-4),
    )

    np.testing.assert_allclose(
        as_callable.step(field, 1.0e-3),
        constant.step(field, 1.0e-3),
        rtol=1e-3,
        atol=2e-4,
    )


def test_callable_composition_dependent_mobility_conserves_mass():
    # The composition-dependent interdiffusion mobility
    # M = X(1-X)(X Mj + (1-X) Mk) --
    # crystallite.mobility.mobility(..., degeneracy=True) -- wired in as a
    # genuinely composition-dependent MassDiffusion.mobility. Flux
    # divergence integrates to zero on a periodic grid regardless of the
    # (possibly spatially varying) mobility, so mass conservation is a
    # mobility-model-independent invariant.
    case = SinusoidalCase()
    solver = MassDiffusion(
        case.grid,
        mobility=lambda c: composition_mobility(
            0.5 + c, d_a=1.0, d_b=2.0, temperature=1.0, degeneracy=True
        ),
        gradient_energy=5.0e-4,
    )

    field = case.initial_field()
    updated = solver.step(field, 1.0e-3)

    assert np.mean(updated) == pytest.approx(np.mean(field), abs=1e-6)


def test_tensor_field_mobility_and_gradient_energy_match_uniform_tensor():
    # A callable returning a (3, 3) + grid.shape tensor field that happens
    # to be spatially uniform must reproduce the plain-(3, 3)-tensor path
    # (same Nyquist-mode caveat as the scalar callable tests above).
    case = SinusoidalCase()
    field = case.initial_field()

    mobility_tensor = np.diag([2.0, 4.0, 8.0])
    gradient_tensor = np.diag([5.0e-4, 1.0e-3, 1.5e-3])
    mobility_field = np.broadcast_to(
        mobility_tensor.reshape(3, 3, 1, 1, 1), (3, 3) + case.grid.shape
    )
    gradient_field = np.broadcast_to(
        gradient_tensor.reshape(3, 3, 1, 1, 1), (3, 3) + case.grid.shape
    )

    constant = MassDiffusion(
        case.grid, mobility=mobility_tensor, gradient_energy=gradient_tensor
    )
    as_callable = MassDiffusion(
        case.grid,
        mobility=lambda c: mobility_field,
        gradient_energy=lambda c: gradient_field,
    )

    np.testing.assert_allclose(
        as_callable.step(field, 1.0e-3),
        constant.step(field, 1.0e-3),
        rtol=1e-3,
        atol=2e-4,
    )


def test_invalid_callable_property_shape_raises():
    case = SinusoidalCase()
    solver = MassDiffusion(case.grid, mobility=lambda c: np.ones((2, 2)))
    with pytest.raises(ValueError, match="mobility must be"):
        solver.step(case.initial_field(), 1.0e-3)


def test_chemical_free_energy_derivative_matches_finite_difference():
    c = np.array([-0.6, -0.1, 0.0, 0.3, 0.9])
    barrier_height, c_alpha, c_beta = 0.5, -1.0, 1.0
    h = 1e-5
    finite_difference = (
        chemical_free_energy(c + h, barrier_height, c_alpha, c_beta)
        - chemical_free_energy(c - h, barrier_height, c_alpha, c_beta)
    ) / (2 * h)

    np.testing.assert_allclose(
        chemical_free_energy_derivative(c, barrier_height, c_alpha, c_beta),
        finite_difference,
        rtol=1e-4,
    )


def test_chemical_free_energy_derivative_matches_double_well_derivative():
    # The fourth-order barrier-height parametrization is the same quartic
    # double well, just reparametrized by f0(X0) instead of `a`.
    c = np.array([-0.6, -0.1, 0.0, 0.3, 0.9])
    barrier_height, c_alpha, c_beta = 0.5, -1.0, 1.0
    a = 16.0 * barrier_height / (c_beta - c_alpha) ** 4

    np.testing.assert_allclose(
        chemical_free_energy_derivative(c, barrier_height, c_alpha, c_beta),
        double_well_derivative(c, a=a, c_alpha=c_alpha, c_beta=c_beta),
    )


def test_chemical_free_energy_class_matches_functions():
    c = np.array([-0.6, -0.1, 0.0, 0.3, 0.9])
    model = ChemicalFreeEnergy(barrier_height=0.5, c_alpha=-1.0, c_beta=1.0)

    np.testing.assert_allclose(
        model.value(c), chemical_free_energy(c, 0.5, -1.0, 1.0)
    )
    np.testing.assert_allclose(
        model.derivative(c), chemical_free_energy_derivative(c, 0.5, -1.0, 1.0)
    )


def test_gradient_energy_class_mixes_by_composition():
    model = GradientEnergy(parent=1.0e-4, product=2.0e-4)
    c = np.array([0.0, 0.25, 1.0])

    np.testing.assert_allclose(model.gradient_energy(c), 1.0e-4 + c * 1.0e-4)


def test_gradient_energy_class_supports_tensors_and_harmonic_mixing():
    parent = np.diag([1.0, 2.0, 3.0])
    product = np.diag([4.0, 5.0, 6.0])
    model = GradientEnergy(parent=parent, product=product, model="harmonic")

    expected = 1.0 / (0.25 / np.diag(product) + 0.75 / np.diag(parent))
    np.testing.assert_allclose(np.diag(model.gradient_energy(0.25)), expected)


def test_gradient_energy_class_rejects_unknown_model():
    with pytest.raises(ValueError, match="model"):
        GradientEnergy(parent=1.0, product=2.0, model="bogus")


def test_mass_diffusion_accepts_model_objects_directly_with_no_lambdas():
    # BinaryMobility/GradientEnergy/ChemicalFreeEnergy each expose a plain
    # bound method with the f(field) -> array signature MassDiffusion
    # expects, so none of mobility/gradient_energy/free_energy_derivative
    # needs a wrapper lambda.
    case = SinusoidalCase()
    mobility_model = BinaryMobility(
        d_a_in_a=1.0, d_a_in_b=2.0, d_b_in_a=2.0, d_b_in_b=3.0, temperature=1.0,
    )
    gradient_model = GradientEnergy(parent=1.0e-4, product=2.0e-4)
    free_energy_model = ChemicalFreeEnergy(barrier_height=0.25, c_alpha=-1.0, c_beta=1.0)

    solver = MassDiffusion(
        case.grid,
        mobility=mobility_model.mobility,
        gradient_energy=gradient_model.gradient_energy,
        free_energy_derivative=free_energy_model.derivative,
    )

    field = 0.5 + case.initial_field()
    updated = solver.step(field, 1.0e-6)

    assert np.mean(updated) == pytest.approx(np.mean(field), abs=1e-6)


def test_invalid_scheme_raises():
    case = SinusoidalCase()
    with pytest.raises(ValueError, match="scheme"):
        MassDiffusion(case.grid, scheme="bogus")


def test_semi_implicit_requires_reference_mobility_for_callable_mobility():
    case = SinusoidalCase()
    with pytest.raises(ValueError, match="reference_mobility"):
        MassDiffusion(
            case.grid,
            mobility=lambda c: np.full_like(c, 1.0),
            scheme="semi_implicit",
            reference_gradient_energy=0.01,
            reference_curvature=-1.0,
        )


def test_semi_implicit_requires_reference_gradient_energy_for_callable_gradient_energy():
    case = SinusoidalCase()
    with pytest.raises(ValueError, match="reference_gradient_energy"):
        MassDiffusion(
            case.grid,
            gradient_energy=lambda c: np.full_like(c, 0.01),
            scheme="semi_implicit",
            reference_mobility=1.0,
            reference_curvature=-1.0,
        )


def test_semi_implicit_requires_reference_curvature():
    case = SinusoidalCase()
    with pytest.raises(ValueError, match="reference_curvature"):
        MassDiffusion(
            case.grid,
            scheme="semi_implicit",
            reference_mobility=1.0,
            reference_gradient_energy=0.01,
        )


def test_semi_implicit_defaults_reference_values_from_constants():
    # When mobility/gradient_energy are already constants, they don't need
    # to be repeated as reference_mobility/reference_gradient_energy.
    case = SinusoidalCase()
    solver = MassDiffusion(
        case.grid,
        mobility=2.0,
        gradient_energy=0.01,
        scheme="semi_implicit",
        reference_curvature=-1.0,
    )
    # Should not raise, and should actually run.
    solver.step(case.initial_field(), 1.0e-3)


def test_semi_implicit_matches_explicit_at_a_safe_time_step():
    # At a time step small enough that dt * L(k_max) << 1 even for this
    # grid's shortest wavelength -- so the semi_implicit denominator is
    # close to 1 everywhere -- the two schemes should agree closely. (A
    # "small" dt is only safe in this sense relative to k_max**4, which is
    # why this uses a small grid rather than just a small dt.)
    grid = Grid(shape=(16, 16, 1), lengths=(1.0, 1.0, 1.0))
    case = SinusoidalCase(grid=grid)
    field = 0.5 + case.initial_field()
    dt = 1.0e-6

    explicit = MassDiffusion(
        grid, mobility=1.0, gradient_energy=0.01,
        bulk_free_energy_coefficient=1.0, left_well=-1.0, right_well=1.0,
    )
    semi_implicit = MassDiffusion(
        grid, mobility=1.0, gradient_energy=0.01,
        bulk_free_energy_coefficient=1.0, left_well=-1.0, right_well=1.0,
        scheme="semi_implicit", reference_mobility=1.0,
        reference_gradient_energy=0.01,
        reference_curvature=double_well_curvature(0.5, a=1.0, c_alpha=-1.0, c_beta=1.0),
    )

    np.testing.assert_allclose(
        semi_implicit.step(field, dt), explicit.step(field, dt), rtol=1e-3, atol=1e-8
    )


def test_semi_implicit_reduces_exactly_to_linear_spectral_diffusion():
    # For a genuinely linear PDE -- mobility, gradient_energy, and the
    # free-energy curvature all constant and exactly equal to their
    # "reference" values, so there is no nonlinear remainder anywhere --
    # semi_implicit must reduce to LinearSpectralDiffusion's own exact
    # solution, at a time step much larger than explicit stability would
    # allow. This is the scheme's core correctness property: dividing only
    # the flux-divergence term by (1 + alpha*dt*L(k)), not the whole
    # update (which would double-count the implicit damping and decay
    # roughly twice too fast -- a real bug this test would have caught).
    mobility, kappa, curvature = 1.0, 0.05, -4.0
    grid = Grid(shape=(32, 32, 1))
    case = SinusoidalCase(grid=grid, amplitude=0.01)
    dt, steps = 2.0e-7, 3000

    semi_implicit = MassDiffusion(
        grid, mobility=mobility, gradient_energy=kappa,
        free_energy_derivative=lambda c: curvature * c,
        scheme="semi_implicit", reference_mobility=mobility,
        reference_gradient_energy=kappa, reference_curvature=curvature,
    )
    field = case.initial_field()
    for _ in range(steps):
        field = semi_implicit.step(field, dt)

    linear = LinearSpectralDiffusion(
        grid, mobility=mobility, gradient_energy=kappa, free_energy_curvature=curvature
    )
    rate = float(np.asarray(linear.decay_rate)[case.mode, 0, 0])
    expected = case.expected_amplitude(steps * dt, rate)

    assert case.amplitude_of(field) == pytest.approx(expected, rel=1e-3)


def test_semi_implicit_stays_stable_where_explicit_blows_up():
    # A deliberately aggressive time step relative to this small grid's
    # Nyquist mode: the plain explicit scheme diverges within a couple of
    # steps, while semi_implicit -- treating the same homogeneous linear
    # operator implicitly -- stays bounded over many more steps.
    grid = Grid(shape=(16, 16, 1), lengths=(1.0, 1.0, 1.0))
    rng = np.random.default_rng(0)
    field0 = 0.5 + 0.05 * rng.standard_normal(grid.shape)
    dt = 1.0e-3

    explicit = MassDiffusion(
        grid, mobility=1.0, gradient_energy=0.01,
        bulk_free_energy_coefficient=1.0, left_well=-1.0, right_well=1.0,
    )
    field = field0.copy()
    for _ in range(3):
        field = explicit.step(field, dt)
    assert not np.all(np.isfinite(field)) or np.max(np.abs(field)) > 50

    reference_curvature = double_well_curvature(0.5, a=1.0, c_alpha=-1.0, c_beta=1.0)
    semi_implicit = MassDiffusion(
        grid, mobility=1.0, gradient_energy=0.01,
        bulk_free_energy_coefficient=1.0, left_well=-1.0, right_well=1.0,
        scheme="semi_implicit", reference_mobility=1.0,
        reference_gradient_energy=0.01, reference_curvature=reference_curvature,
    )
    field = field0.copy()
    for _ in range(30):
        field = semi_implicit.step(field, dt)
    assert np.all(np.isfinite(field))
    assert np.max(np.abs(field)) < 50


def test_semi_implicit_conserves_mass():
    case = SinusoidalCase()
    field = 0.5 + case.initial_field()
    reference_curvature = double_well_curvature(0.5, a=1.0, c_alpha=-1.0, c_beta=1.0)
    solver = MassDiffusion(
        case.grid, mobility=1.0, gradient_energy=0.01,
        bulk_free_energy_coefficient=1.0, left_well=-1.0, right_well=1.0,
        scheme="semi_implicit", reference_mobility=1.0,
        reference_gradient_energy=0.01, reference_curvature=reference_curvature,
    )
    updated = solver.step(field, 1.0e-2)
    assert np.mean(updated) == pytest.approx(np.mean(field), abs=1e-8)


def test_dealias_multiplies_chemical_potential_by_the_lanczos_filter():
    case = SinusoidalCase()
    field = 0.5 + case.initial_field()

    plain = MassDiffusion(case.grid, mobility=1.0, gradient_energy=5.0e-4)
    filtered = MassDiffusion(
        case.grid, mobility=1.0, gradient_energy=5.0e-4, dealias=True
    )

    plain_hat = np.asarray(plain._chemical_potential(field))
    filtered_hat = np.asarray(filtered._chemical_potential(field))
    expected = plain_hat * np.asarray(case.grid.lanczos_filter)

    np.testing.assert_allclose(filtered_hat, expected, rtol=1e-5, atol=1e-8)
    # A no-op filter would make this check vacuous.
    assert np.min(np.asarray(case.grid.lanczos_filter)) < 0.5


def test_dealias_still_conserves_mass():
    # The filter is 1 at k=0 (DC), so it must not perturb mass conservation.
    case = SinusoidalCase()
    field = 0.5 + case.initial_field()
    solver = MassDiffusion(
        case.grid, mobility=1.0, gradient_energy=5.0e-4, dealias=True
    )

    updated = solver.step(field, 1.0e-3)

    assert np.mean(updated) == pytest.approx(np.mean(field), abs=1e-8)