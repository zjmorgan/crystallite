"""``mass.diffusion.1.pgf``: analytical vs. numerical sinusoidal-interface
decaying amplitude, plus ``mass.diffusion.weight.pgf`` (varying the
semi-implicit stabilization weight ``alpha``) and ``mass.diffusion.step.pgf``
(varying the time step).

The analytical curve is :class:`crystallite.mass_diffusion.LinearSpectralDiffusion`'s
exact spectral solution, linearized about the unstable midpoint of the
default double well. The numerical curve is the full nonlinear
:class:`crystallite.mass_diffusion.MassDiffusion`, cross-validated against it in
the small-amplitude limit -- the same setup as
tests/test_diffusion.py's
``test_mass_diffusion_matches_linear_spinodal_decay_for_small_amplitude``.
"""

from __future__ import annotations

import numpy as np

from _plotting import plt, save_all
from crystallite.mass_diffusion import (
    LinearSpectralDiffusion,
    MassDiffusion,
    double_well_curvature,
)
from crystallite.mobility import mobility as composition_mobility
from crystallite.grid import Grid
from crystallite.verification import SinusoidalCase

KAPPA = 0.05
CURVATURE = -4.0
GRID = Grid(shape=(32, 32, 1))
CASE = SinusoidalCase(grid=GRID, amplitude=0.01)
TIME_STEP = 2.0e-7
STEPS = 3000


def _run_nonlinear(mobility=1.0, alpha=1.0, time_step=TIME_STEP, steps=STEPS):
    solver = MassDiffusion(
        GRID, mobility=mobility, gradient_energy=KAPPA,
        bulk_free_energy_coefficient=1.0, left_well=-1.0, right_well=1.0,
        scheme="semi_implicit", reference_mobility=mobility,
        reference_gradient_energy=KAPPA, reference_curvature=CURVATURE,
        alpha=alpha,
    )
    field = CASE.initial_field()
    times = [0.0]
    amplitudes = [CASE.amplitude_of(field)]
    for step in range(steps):
        field = solver.step(field, time_step)
        times.append((step + 1) * time_step)
        amplitudes.append(CASE.amplitude_of(field))
    return np.asarray(times), np.asarray(amplitudes)


def _decay_rate(mobility=1.0, curvature=CURVATURE):
    linear = LinearSpectralDiffusion(
        GRID, mobility=mobility, gradient_energy=KAPPA, free_energy_curvature=curvature
    )
    return float(np.asarray(linear.decay_rate)[CASE.mode, 0, 0])


def _linear_amplitude(times, mobility=1.0, curvature=CURVATURE):
    rate = _decay_rate(mobility=mobility, curvature=curvature)
    return CASE.expected_amplitude(times, rate)


def plot_amplitude_decay():
    """mass.diffusion.1.pgf: nonlinear solver vs. exact linear solution."""
    times, nonlinear_amplitude = _run_nonlinear()
    linear_amplitude = _linear_amplitude(times)

    # Nondimensionalize time by the linear decay rate and amplitude by its
    # initial value, so the curve is a universal exp(-t*rate) regardless of
    # the chosen mobility/gradient-energy/curvature.
    rate = abs(_decay_rate())
    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    ax.plot(
        times * rate, linear_amplitude / CASE.amplitude, color="C0",
        label="analytical (linear)",
    )
    ax.plot(
        times[::150] * rate, nonlinear_amplitude[::150] / CASE.amplitude, "o",
        color="C1", markersize=4, label="numerical (nonlinear)",
    )
    ax.set_xlabel(r"$t\,|\lambda|$")
    ax.set_ylabel(r"$A / A_0$")
    ax.legend()
    save_all(fig, "mass.diffusion.1")
    plt.close(fig)


def plot_weight_sensitivity():
    """mass.diffusion.weight.pgf: accuracy vs. stability trade-off in the
    semi-implicit stabilization weight alpha.

    At this grid's own explicit stability limit (dt ~ 4e-7 here), alpha
    barely matters -- the implicit correction is small regardless. Using
    a time step ~100x past that limit (where plain explicit stepping
    diverges outright) exposes the real trade-off: alpha=1 is standard
    backward Euler on the reference-linear part; alpha<1 recovers more
    accuracy at a fixed dt (the linear part is under-damped, closer to
    the true dynamics) at the cost of the very stability margin that lets
    dt be this large in the first place, while alpha>1 sacrifices
    accuracy (over-damping) for extra stability margin.
    """
    weight_time_step, weight_steps = 5.0e-5, 12
    rate = abs(_decay_rate())
    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    times = np.linspace(0, weight_steps * weight_time_step, weight_steps + 1)
    ax.plot(
        times * rate, _linear_amplitude(times) / CASE.amplitude, color="k",
        label="analytical",
    )
    for alpha in (0.5, 1.0, 2.0, 4.0):
        t, amplitude = _run_nonlinear(
            alpha=alpha, time_step=weight_time_step, steps=weight_steps
        )
        ax.plot(
            t * rate, amplitude / CASE.amplitude, "o-", markersize=4,
            label=rf"$\alpha = {alpha:g}$",
        )
    ax.set_xlabel(r"$t\,|\lambda|$")
    ax.set_ylabel(r"$A / A_0$")
    ax.legend()
    save_all(fig, "mass.diffusion.weight")
    plt.close(fig)


def plot_step_sensitivity():
    """mass.diffusion.step.pgf: time-step discretization error.

    Comparing against the exact linear solution conflates two different
    error sources -- the nonlinear solver's genuine deviation from its own
    linearization (amplitude-dependent, not dt-dependent) dominates over
    any plausible dt range and swamps the discretization error. Isolating
    the latter means comparing each dt against a fine-dt run of the same
    nonlinear solver instead.
    """
    total_time = STEPS * TIME_STEP
    fine_time_step = TIME_STEP / 16
    _, fine_amplitude = _run_nonlinear(
        time_step=fine_time_step, steps=int(round(total_time / fine_time_step))
    )
    reference_final = fine_amplitude[-1]

    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    time_steps = np.array([2.0e-6, 1.0e-6, 5.0e-7, 2.5e-7, TIME_STEP])
    errors = []
    for time_step in time_steps:
        steps = int(round(total_time / time_step))
        _, amplitude = _run_nonlinear(time_step=time_step, steps=steps)
        errors.append(abs(amplitude[-1] - reference_final))
    errors = np.asarray(errors)

    # Nondimensionalize the time step by the linear decay rate (a
    # dimensionless step-size-to-rate ratio) and the error by the reference
    # amplitude's own scale, so both axes are scale-free.
    rate = abs(_decay_rate())
    dimensionless_steps = time_steps * rate
    relative_errors = errors / abs(reference_final)
    ax.loglog(dimensionless_steps, relative_errors, "o-", color="C0", label="MassDiffusion")
    ax.loglog(
        dimensionless_steps, relative_errors[-1] * (dimensionless_steps / dimensionless_steps[-1]),
        "--", color="0.5", label=r"$O(\Delta t)$",
    )
    ax.set_xlabel(r"$\Delta t\,|\lambda|$")
    ax.set_ylabel(r"$|\Delta A| / |A_{\mathrm{ref}}|$")
    ax.legend()
    save_all(fig, "mass.diffusion.step")
    plt.close(fig)


def _run_with_composition_dependent_mobility(degeneracy, d_a=1.0, d_b=1.0):
    """Sinusoidal decay under a composition-dependent mobility about the
    mean composition 0.5 -- ``degeneracy=True`` is the surface-dominated
    mechanism (mobility confined near X=0.5, vanishing at the pure end
    members), ``degeneracy=False`` the volume-dominated one (mobility set
    by the Darken-mixed diffusivity, not vanishing at the end members).
    """
    # The field is centered on 0.5 (not the default double well's own
    # midpoint 0), so the reference curvature for both the semi-implicit
    # stabilizer and the linear comparison must be f''(0.5), not the
    # module-level CURVATURE = f''(0) used by the constant-mobility cases
    # above.
    reference_curvature = float(
        double_well_curvature(0.5, a=1.0, c_alpha=-1.0, c_beta=1.0)
    )
    mobility_fn = lambda c: composition_mobility(  # noqa: E731
        c, d_a, d_b, temperature=1.0, degeneracy=degeneracy
    )
    reference_mobility = float(
        composition_mobility(0.5, d_a, d_b, temperature=1.0, degeneracy=degeneracy)
    )
    solver = MassDiffusion(
        GRID, mobility=mobility_fn, gradient_energy=KAPPA,
        bulk_free_energy_coefficient=1.0, left_well=-1.0, right_well=1.0,
        scheme="semi_implicit", reference_mobility=reference_mobility,
        reference_gradient_energy=KAPPA, reference_curvature=reference_curvature,
    )
    field = 0.5 + CASE.initial_field()
    times = [0.0]
    amplitudes = [CASE.amplitude_of(field)]
    for step in range(STEPS):
        field = solver.step(field, TIME_STEP)
        times.append((step + 1) * TIME_STEP)
        amplitudes.append(CASE.amplitude_of(field))
    return np.asarray(times), np.asarray(amplitudes), reference_mobility, reference_curvature


def plot_volume_and_surface_mechanisms():
    """mass.diffusion.volume.1.pgf / mass.diffusion.surface.1.pgf."""
    for degeneracy, name in [(False, "volume"), (True, "surface")]:
        times, amplitude, reference_mobility, reference_curvature = (
            _run_with_composition_dependent_mobility(degeneracy)
        )
        linear = _linear_amplitude(
            times, mobility=reference_mobility, curvature=reference_curvature
        )

        rate = abs(_decay_rate(mobility=reference_mobility, curvature=reference_curvature))
        fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
        ax.plot(times * rate, linear / CASE.amplitude, color="C0", label="analytical (linear)")
        ax.plot(
            times[::150] * rate, amplitude[::150] / CASE.amplitude, "o", color="C1",
            markersize=4, label="numerical (nonlinear)",
        )
        ax.set_xlabel(r"$t\,|\lambda|$")
        ax.set_ylabel(r"$A / A_0$")
        ax.legend()
        save_all(fig, f"mass.diffusion.{name}.1")
        plt.close(fig)


if __name__ == "__main__":
    plot_amplitude_decay()
    plot_weight_sensitivity()
    plot_step_sensitivity()
    plot_volume_and_surface_mechanisms()
