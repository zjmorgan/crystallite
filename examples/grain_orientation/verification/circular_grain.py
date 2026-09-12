"""Analytical vs. numerical circular grain shrinkage.

A single order parameter is initialized as a disk of radius ``R0``
(:class:`crystallite.verification.CircularGrainCase`) and evolved under
:class:`crystallite.grain_orientation.GrainOrientation`. Curvature-driven
shrinkage follows ``R(t)^2 = R0^2 - 2*mobility*gradient_energy*t`` --
matched asymptotics on the radial Laplacian's curvature term shows the
rate is set by the gradient-energy coefficient itself, not the
interfacial (line-tension) energy ``surface_energy`` one might reach for
first; the two agree only up to a factor that is not of order 1 here (see
``CircularGrainCase``'s docstring for the derivation). The radius is
measured the same way as any other diffuse-interface position in this
project: linear interpolation of the zero crossing along a line through
the center, not an area count.
"""

from __future__ import annotations

import numpy as np

from _plotting import plt, save_all
from crystallite.grain_orientation import GrainOrientation
from crystallite.grid import Grid
from crystallite.verification import CircularGrainCase

A, B, GAMMA = 1.0, 1.0, 1.5
GRADIENT_ENERGY = 2.0e-3
MOBILITY = 1.0
INITIAL_RADIUS = 0.3
CENTER = (0.5, 0.5)


def _zero_crossing_radius(eta, grid, center):
    row = np.asarray(eta)[0, :, grid.shape[1] // 2, 0]
    xs = np.asarray(grid.x[0])[:, 0, 0]
    center_index = np.argmin(np.abs(xs - center[0]))
    for i in range(center_index, len(xs) - 1):
        if row[i] > 0 and row[i + 1] <= 0:
            fraction = row[i] / (row[i] - row[i + 1])
            return xs[i] + fraction * (xs[i + 1] - xs[i]) - center[0]
    return None


def plot_circular_grain():
    grid = Grid(shape=(200, 200, 1), lengths=(1.0, 1.0, 1.0))
    case = CircularGrainCase(
        grid, barrier_coefficient=A, quartic_coefficient=B,
        gradient_energy=GRADIENT_ENERGY, mobility=MOBILITY, center=CENTER,
    )
    model = GrainOrientation(
        grid, mobility=MOBILITY, gradient_energy=GRADIENT_ENERGY,
        barrier_coefficient=A, quartic_coefficient=B, cross_coefficient=GAMMA,
    )

    eta = case.profile(INITIAL_RADIUS)
    time_step = 0.03
    # Run comfortably past the analytic vanishing time so the disk actually
    # disappears on-screen, rather than stopping at some arbitrary early cutoff.
    vanishing_time = float(case.vanishing_time(INITIAL_RADIUS))
    steps = int(1.3 * vanishing_time / time_step) + 1

    times = []
    radii = []
    t = 0.0
    for _ in range(steps):
        radius = _zero_crossing_radius(eta, grid, CENTER)
        if radius is None:
            break
        times.append(t)
        radii.append(radius)
        eta = model.step(eta, time_step)
        t += time_step

    times = np.array(times)
    radii = np.array(radii)
    analytic_times = np.linspace(0.0, max(times[-1], vanishing_time), 200)
    analytic_radii = np.array(
        [float(case.radius(tt, INITIAL_RADIUS)) for tt in analytic_times]
    )

    # Nondimensionalize by the two scales the problem itself sets: the
    # initial radius and the analytic vanishing time -- so the curve always
    # runs from (0, 1) to (1, 0) regardless of the chosen parameters.
    marker_stride = max(1, len(times) // 40)
    t_scale = vanishing_time
    r_scale = INITIAL_RADIUS

    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    ax.plot(
        analytic_times / t_scale, analytic_radii / r_scale, color="C0",
        label="analytical",
    )
    ax.plot(
        times[::marker_stride] / t_scale, radii[::marker_stride] / r_scale,
        "o", color="C1", markersize=4, label="numerical",
    )
    ax.axvline(1.0, color="0.6", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$t / t_0$")
    ax.set_ylabel(r"$R / R_0$")
    ax.legend()
    save_all(fig, "grain_orientation.circular_grain")
    plt.close(fig)

    # Fit only the early, small-curvature regime -- the finite interface
    # width is a growing fraction of R as the disk shrinks, so the constant-
    # rate law is only expected to hold well away from vanishing.
    early = times < 0.5 * vanishing_time
    slope = np.polyfit(times[early], (radii**2)[early], 1)[0]
    predicted_slope = -2.0 * MOBILITY * GRADIENT_ENERGY
    print(
        f"measured d(R^2)/dt (early regime) = {slope:.6g}, "
        f"predicted -2*mobility*gradient_energy = {predicted_slope:.6g}"
    )
    print(
        f"numerical vanishing step = {len(times)} (t={times[-1]:.3g}), "
        f"analytic vanishing_time = {vanishing_time:.3g}"
    )


if __name__ == "__main__":
    plot_circular_grain()
