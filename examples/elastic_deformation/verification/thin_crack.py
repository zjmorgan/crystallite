"""Stress ahead of a genuinely sharp, extremely thin crack tip, under
remote tension (Mode I), shear (Mode II), and antiplane shear (Mode III)
-- the three-mode, hole-height-matched counterpart of the now-removed
``sharp_crack.py`` (which only covered Mode I, at an unrelated length).

An earlier version of this module built its analytic reference on this
project's Eshelby-eigenstrain FFT construction
(:meth:`crystallite.verification.EllipticalHoleInPlateCase.periodic_analytic_stress`),
reasoning that its exact handling of periodicity would matter once the
sampling line reaches all the way to the domain edge. Checked directly,
that construction turns out to be unusable at this crack's aspect ratio
(width = 1 grid pixel, ~50-100:1): it calibrates a single *equivalent
uniform* eigenstrain by numerically probing the field averaged over the
crack's own interior, which for a slit only one pixel wide is dominated
by grid-Nyquist aliasing -- the resulting curve came back essentially
flat (a peak of ~1.13x remote stress right at the tip, actually dipping
*below* 1x farther out), while the numeric CG solver shows a real ~5.4x
concentration there. The domain-*mean* stress was still correct (matching
the image-sum reference to within floating noise), so the construction
isn't wrong in aggregate -- it just cannot resolve the spatial
concentration profile this test exists to check, which is the opposite
of what an "exact periodic solve" was expected to buy here. (Re-run the
same comparison on the old ``sharp_crack.py`` geometry, a 2-pixel-wide
slit, and the same flattening shows up there too -- this is a property of
the construction at this resolution regime, not specific to the 1-pixel
case.)

This module instead uses the periodic image summation of the isolated
exact closed form, in its finite-contrast Eshelby equivalent-eigenstrain
version (:meth:`~crystallite.verification.EllipticalHoleInPlateCase.periodic_inhomogeneity_stress`
for the two in-plane modes,
:meth:`~crystallite.verification.EllipticalHoleInPlateCase.periodic_antiplane_inhomogeneity_stress`
for Mode III), at the same `CONTRAST` the numeric solver uses, rather
than the void-limit image sums (``periodic_void_stress``/
``periodic_antiplane_void_stress``), which overshoot the solver's
near-tip peak (on the original 1-pixel slit: 5.77x/5.80x vs. the solver's
5.4x, against 5.24x/5.56x for the eigenstrain version).
Neither is affected by the aliasing above, since the closed form is
evaluated pointwise with no shape-FFT step to alias.

The crack is a real, extremely eccentric elliptical hole (not the
circular-hole stand-in the original reference
material used for its own "crack" illustrations):

- ``semi_axis_a`` (crack half-width): ``SLIT_HALF_WIDTH_PIXELS`` (1.5)
  grid spacings, i.e. a 3-pixel-wide slit. The original 1-pixel slit
  gave a larger, noisier mismatch: the near-tip sample is very sensitive
  to where the tip falls on the grid, and a half-width that is not an
  odd multiple of half a pixel (e.g. 1.0) rasterizes to a slit narrower
  than the analytic ellipse assumes. With the near-tip sample excluded
  (``xi < 1.15``), 1.5 lowered the RMS difference from the analytic curve
  (0.141 -> 0.078 Mode I, 0.045 -> 0.040 Mode II, 0.069 -> 0.046 Mode III).
- ``semi_axis_b`` (crack half-length): set equal to
  ``hole_in_plate.py``'s own ``HOLE_RADIUS``, so this crack's full length
  matches the circular hole's diameter exactly, for a fair side-by-side
  comparison of "hole" vs. "genuine sharp crack" of the same overall
  size (``sharp_crack.py`` used an unrelated, larger length instead).

Sharp (non-dealiased) cutoff, not a diffuse boundary: a feature this thin
never reaches the intended `CONTRAST` under dealiasing -- the Lanczos
filter smooths a slit this narrow away almost entirely first (checked
directly in ``sharp_crack.py`` before its removal); the sharp cutoff
reaches the true contrast exactly, matching ``screw_dislocation.py``'s
own precedent for the same kind of extreme-aspect-ratio ellipse.

Because the crack is axis-aligned (long axis along x2, faces normal to
x1) rather than sitting at an arbitrary angle on a circular boundary, the
"ahead of the tip" reading needs no polar rotation at all: directly along the sampling line (x1=center[0], x2 beyond the
tip), the traction on the crack's own faces (normal to x1) is just sigma_11
(opening, Mode I), sigma_12 (in-plane sliding, Mode II), and sigma_13
(antiplane tearing, Mode III) -- exactly the classical local-frame LEFM
components, read with no coordinate transform. Each is also exactly the
remote-background component each mode's own load (tension, shear,
antiplane shear respectively) prescribes far away, so every curve here
approaches 1, not 0, far from the tip.

Sampled over ``xi`` (distance from the crack's own *center*, not the tip,
normalized by ``HOLE_RADIUS``) in ``[1, 4]`` -- the same window and the
same normalization (`xi=1` at the tip) as this project's original (pre-crystallite) reference material's own
``hooke.pdf`` figures, for a directly comparable view across all of this
project's crack/dislocation cases.
"""

from __future__ import annotations

import numpy as np

from _plotting import analytic_numeric_curve, component_legend, plt, save_all
from crystallite.grid import Grid
from crystallite.verification import EllipticalHoleInPlateCase

MATRIX_LAME_LAMBDA = 1.0
MATRIX_LAME_MU = 0.7
HOLE_RADIUS = 0.1  # matches hole_in_plate.py's circular hole: same crack length as hole diameter
CONTRAST = 1.0e-3
GRID_SHAPE = (256, 256, 1)
MAGNITUDE = 0.01
# Half-width in grid pixels. An odd-pixel-count full width (0.5, 1.5, 2.5, ...)
# on a pixel-centered crack makes the discretized slit exactly match the
# analytic ellipse's width; 1.5 (a 3-pixel slit) gave the lowest error against
# the analytic curve of the widths tried (0.5, 1.5, 2.5, at 256^2 and 512^2).
SLIT_HALF_WIDTH_PIXELS = 1.5
N_IMAGES = 1  # periodic image cutoff for the analytic curves (already converged here)

# label -> (load, (i, j) Cartesian stress component read directly ahead
# of the tip, plot label). No polar conversion: the crack's own faces
# are normal to x1, so (i, j) is exactly the traction component each
# mode opens/slides/tears -- see module docstring.
_MODES = {
    "mode1": ("tension", (0, 0), r"$\sigma_{11}$"),
    "mode2": ("shear", (0, 1), r"$\sigma_{12}$"),
    "mode3": ("antiplane", (0, 2), r"$\sigma_{13}$"),
}


def _build_case():
    grid = Grid(shape=GRID_SHAPE, lengths=(1.0, 1.0, 1.0))
    semi_axis_a = SLIT_HALF_WIDTH_PIXELS * grid.spacing[0]
    return EllipticalHoleInPlateCase(
        grid, matrix_lame_lambda=MATRIX_LAME_LAMBDA, matrix_lame_mu=MATRIX_LAME_MU,
        semi_axis_a=semi_axis_a, semi_axis_b=HOLE_RADIUS, contrast=CONTRAST,
        center=(0.5, 0.5),
    )


def _xi_and_mask(case):
    """Grid column at x1=center[0], and the boolean mask/``xi`` (distance
    from the crack's own *center*, normalized by `HOLE_RADIUS`) for
    ``xi`` in ``[1, 4]`` -- `xi=1` exactly at the tip, since
    `semi_axis_b == HOLE_RADIUS`."""
    grid = case.grid
    x1 = np.asarray(grid.x[0])[:, 0, 0]
    x2 = np.asarray(grid.x[1])[0, :, 0]
    col = np.argmin(np.abs(x1 - case.center[0]))
    xi = (x2 - case.center[1]) / HOLE_RADIUS
    mask = (xi >= 1.0) & (xi <= 4.0)
    return col, mask, xi[mask]


def _ahead_of_tip_numeric(case, solver, load, component):
    i, j = component
    if load == "antiplane":
        eps_bar = case.macro_antiplane_strain(MAGNITUDE, solver=solver)
    else:
        eps_bar = case.macro_strain(load, MAGNITUDE, solver=solver)
    sol = solver.solve(eps_bar, tol=1.0e-6, max_iterations=8000)
    print(f"{load} (numeric): converged={sol.converged}, iterations={sol.iterations}, "
          f"residual={sol.residual_norm:.3g}")
    sigma = np.asarray(sol.stress)

    col, mask, xi = _xi_and_mask(case)
    values = sigma[i, j, col, mask, 0] / MAGNITUDE
    return xi, values


def _ahead_of_tip_analytic(case, load, component):
    i, j = component
    if load == "antiplane":
        stress = np.asarray(case.periodic_antiplane_inhomogeneity_stress(MAGNITUDE, n_images=N_IMAGES))
    else:
        stress = np.asarray(case.periodic_inhomogeneity_stress(load, MAGNITUDE, n_images=N_IMAGES))

    col, mask, xi = _xi_and_mask(case)
    values = stress[i, j, col, mask, 0] / MAGNITUDE
    return xi, values


def plot_mode(label):
    load, component, plot_label = _MODES[label]
    print(f"{label} ({load}):")
    case = _build_case()
    solver = case.solver()

    xi_num, s_num = _ahead_of_tip_numeric(case, solver, load, component)
    xi_an, s_an = _ahead_of_tip_analytic(case, load, component)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.set_xlim(1.0, 4.0)
    analytic_numeric_curve(
        ax, xi_an, s_an, s_num, plot_label, "C0", downsample=1, x_numeric=xi_num, markersize=4
    )
    ax.set_xlabel(r"$x_2 / r_0$")
    ax.set_ylabel(r"$\sigma / \sigma_\infty$")
    component_legend(ax)
    save_all(fig, f"elastic_deformation.thin_crack_{label}")
    plt.close(fig)


if __name__ == "__main__":
    plot_mode("mode1")
    plot_mode("mode2")
    plot_mode("mode3")
