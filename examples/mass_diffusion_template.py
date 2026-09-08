"""Template for a nonlinear mass-diffusion simulation.

This is a design template, not a runnable example yet. The names marked
``TODO`` identify the next APIs required for the production solver.
"""

from dataclasses import dataclass

import numpy as np

from crystallite.field import Field, State
from crystallite.phase import Grid
from crystallite.material.properties import Solid


@dataclass(frozen=True)
class RunConfig:
    """Numerical controls for a mass-diffusion run."""

    time_step: float = 1.0e-5
    steps_per_sweep: int = 100
    sweeps: int = 100
    report_every: int = 1
    save_every: int = 0


def define_phases():
    """Define parent and product tensor properties.

    Returns
    -------
    tuple of Solid
        Parent and product material descriptions.

    Notes
    -----
    The phase properties are tensors. Scalar isotropic values are only a
    shorthand for a diagonal tensor and should be expanded by the property
    system when the solver is assembled.
    """
    parent = Solid(point_group="1")
    parent.set(
        "diffusivity",
        d11=1.0e-5,
        d22=1.0e-5,
        d33=1.0e-5,
    )
    parent.set(
        "concentration_gradient_energy",
        kappa11=5.0e-4,
        kappa22=5.0e-4,
        kappa33=5.0e-4,
    )

    product = Solid(point_group="1")
    product.set(
        "diffusivity",
        d11=2.0e-5,
        d22=2.0e-5,
        d33=2.0e-5,
    )
    product.set(
        "concentration_gradient_energy",
        kappa11=7.5e-4,
        kappa22=7.5e-4,
        kappa33=7.5e-4,
    )
    return parent, product


def initial_state(grid):
    """Create a composition field and phase-fraction field."""
    composition = 0.5 + 0.125 * np.sin(4.0 * np.pi * grid.x[0])
    composition = np.broadcast_to(composition, grid.shape).copy()
    phase_fraction = np.zeros(grid.shape)

    return State(
        grid,
        composition=Field(composition, grid, name="composition"),
        phase_fraction=Field(phase_fraction, grid, name="phase_fraction"),
    )


def run():
    """Set up and advance a mass-diffusion simulation."""
    config = RunConfig()
    grid = Grid(shape=(256, 256, 1), lengths=(1.0, 1.0, 1.0))
    parent, product = define_phases()
    state = initial_state(grid)

    # TODO: construct effective M_ij(c, phi) and K_ij(c, phi) from phases.
    # These should use the selected arithmetic or harmonic tensor mixture
    # rule, rather than mixing tensor entries independently for harmonic
    # behavior.
    mobility = effective_mobility(parent, product, state)
    gradient_energy = effective_gradient_energy(parent, product, state)

    # TODO: replace with the nonlinear MassDiffusion solver.
    solver = MassDiffusion(
        grid=grid,
        mobility=mobility,
        gradient_energy=gradient_energy,
        free_energy=free_energy,
        scheme="semi_implicit",
    )

    energy_history = []
    snapshots = []

    for sweep in range(config.sweeps):
        for _ in range(config.steps_per_sweep):
            state = solver.step(state, config.time_step)

        if (sweep + 1) % config.report_every == 0:
            energy = solver.energy(state)
            energy_history.append((state.time, energy))
            print(
                f"sweep={sweep + 1} time={state.time:.6g} "
                f"energy={energy:.6g}"
            )

        if config.save_every and (sweep + 1) % config.save_every == 0:
            snapshots.append(state)

    # TODO: write energy_history and snapshots to ignored output storage.
    # TODO: call a plotting helper for the final composition field.
    return state, energy_history, snapshots


# TODO: These functions become real APIs as the nonlinear solver is built.
def effective_mobility(parent, product, state):
    """Return the mixed mobility tensor field for the current state."""
    raise NotImplementedError


def effective_gradient_energy(parent, product, state):
    """Return the mixed gradient-energy tensor field for the current state."""
    raise NotImplementedError


def free_energy(composition):
    """Return bulk free energy density for the current composition."""
    raise NotImplementedError


class MassDiffusion:
    """Placeholder for the future nonlinear diffusion solver."""

    def __init__(self, **kwargs):
        raise NotImplementedError("MassDiffusion is not implemented yet")
