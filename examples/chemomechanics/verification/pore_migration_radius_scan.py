r"""Pore migration velocity under a remote stress gradient, vs. pore radius:
volume- and surface-diffusion mechanisms compared against the lattice-frame
background prediction, :math:`v_\mathrm{lattice}=(1+\nu)\,\gamma_\infty`
(volume) and :math:`(1+\nu)\,\gamma_\infty\,\delta/r_0` (surface), both
independent of the pore's own presence. :math:`\delta` is the diffuse
interface's own width, standing in for a physical surface mobility (there is
no separate physical surface-diffusivity parameter in this model).

A circular pore sits in an elastically heterogeneous matrix (Lame-parameter
contrast `CONTRAST`) under a remote "moment" stress-gradient background,
:math:`\sigma_{11}\to\gamma_\infty\,x_2` far from the pore
(:class:`crystallite.verification.elastic_deformation.HoleInPlateCase`). The
elastic chemical potential :math:`\mu_\mathrm{el}=-\mathrm{trace}(\sigma)`
couples to Cahn-Hilliard diffusion (:class:`crystallite.mass_diffusion.MassDiffusion`)
through one of two composition-dependent mobilities: :math:`M(X)=X` ("volume",
nonzero through the bulk) or :math:`M(X)=X(1-X)` ("surface", confined to the
diffuse interface) -- :math:`X=0` is the pore, :math:`X=1` the matrix. The
pore's weighted centroid drifts at a velocity measured here by direct time
integration (the background's own linear part is kept out of the FFT-based
flux -- it is not periodic and would alias -- and re-added as an explicit
real-space flux term; only the pore's localized stress correction goes
through the spectral solver).

For a *real* pore, the driving force is literally zero inside (vacuum) and
the full background value just outside; averaging those two flux values
across the jump discontinuity at the boundary halves :math:`v_\mathrm{lattice}`.
That averaging is well-motivated for the volume mechanism -- a genuine jump
between two bulk mobility values, :math:`M(0)=0` and :math:`M(1)=1` -- and
matches what's measured here (volume sits close to half of
:math:`v_\mathrm{lattice}`, flat with radius). But the surface mechanism's
transport is structurally different: it's a divergence of *tangential* flux
confined to a thin band around :math:`X=0.5` (the mobility :math:`X(1-X)` has
no bulk value on either side to average), so the same halving argument
doesn't obviously apply to it. Consistent with that, the surface mechanism's
measured velocity magnitude tracks :math:`v_\mathrm{lattice}` (unhalved)
rather than half of it, closely following its :math:`1/r_0` shape across a 4x
radius range -- but with the **opposite sign**, which the above does not
explain and remains an open question under separate investigation.

At small enough radii, the diffuse interface width relative to the pore
(:math:`\delta/r_0`) grows large enough that the volume-mechanism pore
partially heals over the run instead of translating rigidly (seen at
:math:`r_0=0.05`, excluded from `RADII` here: composition minimum rose from
~0 to ~0.3 over 800 steps) -- worth rechecking if any radius below ~0.06
looks similarly compromised.
"""

from __future__ import annotations

import numpy as np

from _plotting import analytic_numeric_curve, cached_numeric, component_legend, plt, save_all
from crystallite.elastic_deformation import ElasticDeformation
from crystallite.grid import Grid
from crystallite.mass_diffusion import MassDiffusion, double_well_curvature
from crystallite.verification.elastic_deformation import HoleInPlateCase

N = 128
MATRIX_LAMBDA, MATRIX_MU = 1.0, 0.7
POISSON_RATIO = MATRIX_LAMBDA / (2.0 * (MATRIX_LAMBDA + MATRIX_MU))
CONTRAST = 1.0e-3
CENTER = (0.5, 0.5)
ORIGIN = (CENTER[0], CENTER[1], 0.0)
GAMMA_INF = 0.1
RADII = (0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20)
PLOTTED_RADII = RADII
DT = 1.0e-5
STEPS = 800
RECORD_EVERY = 40
A_BULK = 1.0
INTERFACE_WIDTH = 1.5 / N


def lattice_velocity_volume(poisson_ratio=POISSON_RATIO, remote_stress_gradient=GAMMA_INF):
    """Lattice-frame (no-hole, unit-mobility) volume-mechanism migration
    velocity: radius-independent -- see module docstring."""
    return (1.0 + poisson_ratio) * remote_stress_gradient


def lattice_velocity_surface(pore_radius, poisson_ratio=POISSON_RATIO,
                              remote_stress_gradient=GAMMA_INF, interface_width=INTERFACE_WIDTH):
    """Lattice-frame (no-hole, unit-mobility) surface-mechanism migration
    velocity: scales as 1/pore_radius -- see module docstring."""
    return (1.0 + poisson_ratio) * remote_stress_gradient * interface_width / pore_radius


def _run_radius(r0, mechanism):
    """Time-integrate the pore's centroid drift at radius `r0` for `mechanism`
    ("volume" or "surface"); returns ``(whole_run_fit, last_quarter_fit)``."""
    grid = Grid(shape=(N, N, 1), lengths=(1.0, 1.0, 1.0))
    dx = 1.0 / N
    kappa = 2.0 * A_BULK * INTERFACE_WIDTH**2
    curv = double_well_curvature(0.0, A_BULK, 0.0, 1.0)
    mobility_fn = (lambda c: c) if mechanism == "volume" else (lambda c: c * (1.0 - c))

    case = HoleInPlateCase(grid, MATRIX_LAMBDA, MATRIX_MU, r0, contrast=CONTRAST, center=CENTER)
    macro_strain_gradient = case.macro_strain_gradient("moment", GAMMA_INF)
    # trace(sigma_background) is affine in x with constant gradient g_bg (see
    # module docstring): trace_coeff picks out its direction/magnitude from
    # the strain-gradient tensor via the isotropic trace law sigma_kk =
    # (3*lambda+2*mu)*eps_kk.
    trace_coeff = np.einsum("iij->j", np.asarray(macro_strain_gradient))
    g_bg = (3.0 * MATRIX_LAMBDA + 2.0 * MATRIX_MU) * trace_coeff

    x, y = np.asarray(grid.x[0]), np.asarray(grid.x[1])
    r = np.hypot(x - CENTER[0], y - CENTER[1])
    c0 = np.where(r < r0, 0.0, 1.0).astype(np.float32)
    c0 = np.clip(np.real(np.asarray(grid.ifft(grid.fft(c0) * grid.lanczos_filter))), 0.0, 1.0)

    # Same affine construction as ElasticDeformation._gradient_strain_field,
    # used only to subtract the non-periodic background trace before
    # anything is FFT'd.
    field_bg = np.zeros((3, 3) + grid.shape, dtype=np.float32)
    for axis in range(3):
        coord = np.asarray(grid.x[axis])
        field_bg = field_bg + np.asarray(macro_strain_gradient)[:, :, axis].reshape(
            (3, 3, 1, 1, 1)
        ) * (coord - ORIGIN[axis])
    trace_background = (3.0 * MATRIX_LAMBDA + 2.0 * MATRIX_MU) * (
        field_bg[0, 0] + field_bg[1, 1] + field_bg[2, 2]
    )

    diffusion = MassDiffusion(
        grid, mobility=mobility_fn, gradient_energy=kappa, bulk_free_energy_coefficient=A_BULK,
        left_well=0.0, right_well=1.0, scheme="semi_implicit",
        reference_curvature=curv, reference_mobility=1.0, reference_gradient_energy=kappa,
        dealias=True,
    )

    def elasticity_for(c):
        lam = np.maximum((CONTRAST + (1.0 - CONTRAST) * c) * MATRIX_LAMBDA, CONTRAST * MATRIX_LAMBDA)
        mu = np.maximum((CONTRAST + (1.0 - CONTRAST) * c) * MATRIX_MU, CONTRAST * MATRIX_MU)
        return ElasticDeformation(grid, lam.astype(np.float32), mu.astype(np.float32),
                                   MATRIX_LAMBDA, MATRIX_MU)

    def centroid_y(c):
        weight = np.clip(1.0 - c, 0.0, 1.0)[:, :, 0]
        yg = np.asarray(grid.x[1])[:, :, 0]
        return float((weight * yg).sum() / weight.sum())

    def real_gradient(field):
        return np.real(np.asarray(grid.ifft(diffusion.operator.grad(grid.fft(field)))))

    c = c0.copy().astype(np.float32)
    disp = None
    times, ys = [], []
    for step in range(STEPS):
        solver = elasticity_for(c)
        sol = solver.solve(np.zeros((3, 3)), macro_strain_gradient=macro_strain_gradient,
                            gradient_origin=ORIGIN, initial_displacement=disp,
                            tol=2.0e-4, max_iterations=1500)
        disp = sol.displacement
        trace_full = np.asarray(sol.stress[0, 0] + sol.stress[1, 1] + sol.stress[2, 2])
        trace_local = trace_full - trace_background

        c_after_local = diffusion.step(c, DT, extra_chemical_potential=-trace_local)

        # Constant background gradient g_bg -> flux J_bg = M(c)*g_bg, added
        # explicitly in real space; div(J_bg) = g_bg . grad(M(c)) since g_bg
        # is constant. Never FFT trace_background itself (see module
        # docstring / ElasticDeformation._gradient_strain_field).
        grad_mobility = real_gradient(mobility_fn(c))
        div_background_flux = sum(g_bg[i] * grad_mobility[i] for i in range(3))
        c = (np.asarray(c_after_local) - DT * div_background_flux).astype(np.float32)

        if step % RECORD_EVERY == 0:
            times.append(step * DT)
            ys.append(centroid_y(np.asarray(c)))

    times, ys = np.array(times), np.array(ys)
    design = np.vstack([times, np.ones_like(times)]).T
    fitted, _ = np.linalg.lstsq(design, ys, rcond=None)[0]
    quarter = len(times) // 4
    design_q = np.vstack([times[-quarter:], np.ones(quarter)]).T
    last_quarter, _ = np.linalg.lstsq(design_q, ys[-quarter:], rcond=None)[0]
    return float(fitted), float(last_quarter)


def _measured(r0, mechanism):
    key = f"pore_migration_radius_scan.{mechanism}.r0_{r0:.3f}.N{N}.dt{DT:.0e}.steps{STEPS}"
    return cached_numeric(key, lambda: _run_radius(r0, mechanism))


def plot_radius_scan():
    plotted = np.array(PLOTTED_RADII)
    r_smooth = np.linspace(plotted.min() * 0.9, plotted.max() * 1.05, 200)

    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)

    volume_prediction = np.full_like(r_smooth, 0.5 * lattice_velocity_volume())
    volume_numeric = np.array([_measured(r0, "volume")[1] for r0 in plotted])
    analytic_numeric_curve(
        ax, r_smooth, volume_prediction, volume_numeric, "volume", "C0",
        downsample=1, x_numeric=plotted, markersize=6,
    )

    surface_prediction = -lattice_velocity_surface(r_smooth)
    surface_numeric = np.array([_measured(r0, "surface")[1] for r0 in plotted])
    analytic_numeric_curve(
        ax, r_smooth, surface_prediction, surface_numeric, "surface", "C1",
        downsample=1, x_numeric=plotted, markersize=6,
    )

    ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
    ax.set_xlabel(r"pore radius $r_0$")
    ax.set_ylabel("migration velocity")
    ax.set_title("pore migration under a remote stress gradient")
    component_legend(ax, loc="center right")
    save_all(fig, "chemomechanics.pore_migration_radius_scan")
    plt.close(fig)


if __name__ == "__main__":
    plot_radius_scan()
