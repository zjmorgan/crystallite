"""2D spinodal decomposition example with composition-dependent mobility,
gradient energy, and semi-implicit time stepping.

This example uses the nonlinear ``MassDiffusion`` solver in a minimal
spinodal decomposition setup. The composition field is initialized with a
small random perturbation about the unstable midpoint and then evolves
under a composition-dependent (rather than constant) mobility and
gradient-energy coefficient. Each is a small, reusable model object with a
named method -- passed directly to ``MassDiffusion`` with no wrapper
``lambda`` needed:

- ``mobility`` is :class:`crystallite.mobility.BinaryMobility`, a binary
  A-B interdiffusion model: each species' own diffusivity is first
  interpolated between its value in the two pure end members, then
  Darken-combined and converted to a mobility.
- ``gradient_energy`` is :class:`crystallite.mass_diffusion.GradientEnergy`,
  which mixes a "parent" and "product" phase's (possibly anisotropic)
  gradient-energy tensor by composition -- gradient energy is properly a
  tensor, just like diffusivity, not a bare scalar.
- ``free_energy_derivative`` is :class:`crystallite.mass_diffusion.ChemicalFreeEnergy`,
  the fourth-order Taylor approximation to the bulk free energy,
  parametrized by the free-energy barrier height at the reference
  composition rather than the raw quartic coefficient. Composition is on
  ``[0, 1]`` throughout (``c_alpha=0, c_beta=1``), matching the mole
  fraction convention ``BinaryMobility`` expects -- mixing that convention
  with a ``[-1, 1]`` free energy would let the field push ``BinaryMobility``
  into an unphysical negative-mobility regime once it starts separating.

The end-member diffusivity and gradient-energy tensors themselves come
from :class:`crystallite.material.properties.Solid`, so only the
independent tensor components allowed by the parent/product phases' point
group need to be set -- the rest are filled in (or forced to vanish) by
symmetry, rather than typed out by hand.

``MassDiffusion`` uses ``scheme="semi_implicit"``: real spinodal growth
here is orders of magnitude slower than the time step a plain explicit
scheme can take before its shortest-wavelength (Nyquist) mode goes
unstable, so explicit stepping can only ever show either no visible
decomposition (small, stable time step) or a grid-scale checkerboard
artifact (a time step pushed large enough to see growth in a practical
step count) -- never real microstructure. The semi-implicit scheme treats
a homogeneous reference linearization of the dynamics implicitly, which
handles that stiffness unconditionally and lets the true nonlinear
dynamics evolve explicitly on top of it.

``dealias=True`` additionally applies ``grid.lanczos_filter`` (ported from
the reference Fortran solver's ``sf_lanczos_filter``/``sf_filter``) to the
chemical potential each step, smoothly tapering its Fourier transform to
exactly 0 at the Nyquist frequency -- suppressing the spurious
high-wavenumber content the nonlinear free energy and composition-dependent
mobility/gradient energy generate via aliasing when evaluated
pseudo-spectrally.

Plots use the project's shared style,
:func:`crystallite.visualization.configure_pgf`.
"""

from __future__ import annotations

import numpy as np

from crystallite.mass_diffusion import (
    ChemicalFreeEnergy,
    GradientEnergy,
    MassDiffusion,
    double_well_curvature,
)
from crystallite.material.properties import Solid
from crystallite.mobility import BinaryMobility
from crystallite.grid import Grid

REFERENCE_COMPOSITION = 0.5
BARRIER_HEIGHT = 0.25


def define_phases(point_group="432"):
    """Return symmetry-resolved parent/product diffusivity and
    gradient-energy tensors.

    Parameters
    ----------
    point_group : str, default="432"
        Crystallographic point group shared by both phases (here cubic,
        so a single ``d11``/``kappa11`` constant is enough -- symmetry
        forces ``d22 = d33 = d11`` and every off-diagonal entry to zero).

    Returns
    -------
    parent, product : Solid
    """
    parent = Solid(point_group=point_group)
    parent.set("diffusivity", d11=1.0e-5)
    parent.set("concentration_gradient_energy", kappa11=3.0e-3)

    product = Solid(point_group=point_group)
    product.set("diffusivity", d11=2.0e-5)
    product.set("concentration_gradient_energy", kappa11=6.0e-3)

    return parent, product


def build_solver(grid):
    """Construct the composition-dependent, semi-implicit MassDiffusion
    solver used by this example.

    Returns
    -------
    MassDiffusion
    """
    parent, product = define_phases()

    # Binary A-B interdiffusion mobility: each species' diffusivity is
    # interpolated between its two pure end-member values, then
    # Darken-combined into a mobility. The end members are the
    # parent/product phases' own symmetry-resolved (3, 3) diffusivity
    # tensors; both species are taken to share each end member's value
    # here, for lack of a separate per-species tracer diffusivity.
    mobility_model = BinaryMobility(
        d_a_in_a=parent.get("diffusivity"),
        d_a_in_b=product.get("diffusivity"),
        d_b_in_a=parent.get("diffusivity"),
        d_b_in_b=product.get("diffusivity"),
        temperature=1.0,
    )

    # Gradient-energy tensor mixed between the parent and product phases'
    # own symmetry-resolved values, weighted by composition.
    gradient_energy_model = GradientEnergy(
        parent=parent.get("concentration_gradient_energy"),
        product=product.get("concentration_gradient_energy"),
    )

    # Standard double-well bulk free energy, f(c) = 1/4 c^2 (1 - c)^2 so
    # f'(c) = 2 c (c - 1) (2c - 1), in the barrier-height parametrization
    # (a = 16 * barrier_height / (c_beta - c_alpha)**4 = 4.0 here).
    free_energy_model = ChemicalFreeEnergy(
        barrier_height=BARRIER_HEIGHT, c_alpha=0.0, c_beta=1.0
    )

    # The semi-implicit scheme needs a homogeneous reference point:
    # mobility/gradient-energy/curvature evaluated at the mean
    # composition, exactly matching the linear operator
    # LinearSpectralDiffusion.decay_rate would use to describe small
    # perturbations about that state.
    reference_mobility = mobility_model.mobility(REFERENCE_COMPOSITION)
    reference_gradient_energy = gradient_energy_model.gradient_energy(
        REFERENCE_COMPOSITION
    )
    reference_curvature = double_well_curvature(
        REFERENCE_COMPOSITION,
        a=16.0 * BARRIER_HEIGHT,
        c_alpha=0.0,
        c_beta=1.0,
    )

    return MassDiffusion(
        grid=grid,
        mobility=mobility_model.mobility,
        gradient_energy=gradient_energy_model.gradient_energy,
        free_energy_derivative=free_energy_model.derivative,
        scheme="semi_implicit",
        reference_mobility=reference_mobility,
        reference_gradient_energy=reference_gradient_energy,
        reference_curvature=reference_curvature,
        dealias=True,
    )


def initial_field(grid, amplitude=0.015):
    """Return a small-amplitude noisy field around the unstable state."""
    rng = np.random.default_rng(42)
    noise = amplitude * rng.standard_normal(grid.shape)
    return REFERENCE_COMPOSITION + noise


def run_example(
    shape=(64, 64, 1),
    lengths=(1.0, 1.0, 1.0),
    time_step=0.05,
    steps=100000,
    snapshot_every=2500,
    save_path=None,
    show_plot=False,
):
    """Advance a 2D spinodal decomposition run.

    Parameters
    ----------
    shape : tuple of int, default=(64, 64, 1)
        Real-space grid shape.
    lengths : tuple of float, default=(1.0, 1.0, 1.0)
        Domain length in each coordinate direction.
    time_step : float, default=0.05
        Semi-implicit integration time step.
    steps : int, default=100000
        Number of time steps to take.
    snapshot_every : int, default=2500
        Save a snapshot every so many steps.
    save_path : str or None, default=None
        Optional output file for a final PNG image.
    show_plot : bool, default=False
        If True, display the final field with Matplotlib.

    Returns
    -------
    field : ndarray
        Final composition field.
    snapshots : list of ndarray
        Stored composition snapshots for later plotting.
    """
    grid = Grid(shape=shape, lengths=lengths)
    field = initial_field(grid)
    solver = build_solver(grid)

    snapshots = []
    for step in range(steps):
        field = solver.step(field, time_step)
        # BinaryMobility's X_A * X_B degeneracy factor -- and hence the
        # flux -- vanishes exactly at c=0/1, so once a point drifts past
        # those bounds it stops responding to its own restoring chemical
        # potential while neighboring, still-mobile points keep pushing
        # mass into it: a slow secular drift with no self-correction,
        # not visible until it eventually dwarfs the composition scale.
        # Clipping back to the physical [0, 1] mole-fraction range after
        # each step is the standard practical fix for this in
        # degenerate-mobility Cahn-Hilliard schemes.
        field = np.clip(field, 0.0, 1.0)
        if (step + 1) % snapshot_every == 0 or step == steps - 1:
            snapshots.append(field.copy())

    if save_path is not None or show_plot:
        try:
            from crystallite.visualization import configure_pgf

            plt = configure_pgf()
        except ImportError:  # pragma: no cover
            print(
                "Matplotlib is not installed; skipping plot output for this run."
            )
            return field, snapshots

        fig, ax = plt.subplots(figsize=(4.5, 4.5), constrained_layout=True)
        image = ax.imshow(
            field[:, :, 0],
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title("Spinodal decomposition")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        fig.colorbar(image, ax=ax, label=r"composition $c$")

        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved plot to {save_path}")

        if show_plot:
            plt.show()

        plt.close(fig)

    return field, snapshots


if __name__ == "__main__":
    field, snapshots = run_example(
        shape=(64, 64, 1),
        time_step=0.05,
        steps=100000,
        snapshot_every=2500,
        save_path="spinodal_decomposition.png",
        show_plot=False,
    )
    print(f"final mean composition = {np.mean(field):.6f}")
    print(f"final std composition = {np.std(field):.6f}")
    print(f"stored {len(snapshots)} snapshots")
