"""Verification cases based on the Fick presentation examples."""

from dataclasses import dataclass

from crystallite.backend import xp
from crystallite.phase import Grid


@dataclass(frozen=True)
class InterfaceCase:
    """One-dimensional two-phase interface verification case.

    The equilibrium profile for the binary barrier model is a tanh transition
    layer between two stable phases. This case builds a discrete initial step
    across the interface and measures how closely a relaxed field approaches
    the tanh profile.

    Parameters
    ----------
    grid : Grid, optional
        One-dimensional periodic grid. The default uses a 256-point domain.
    center : float, default=0.5
        Interface center in the domain coordinates.
    width : float, default=0.1
        Interface width parameter in the tanh profile.
    left_state : float, default=-1.0
        Stable phase on the left side of the interface.
    right_state : float, default=1.0
        Stable phase on the right side of the interface.
    """

    grid: Grid | None = None
    center: float = 0.5
    width: float = 0.1
    left_state: float = -1.0
    right_state: float = 1.0

    def __post_init__(self):
        if self.grid is None:
            object.__setattr__(
                self,
                "grid",
                Grid(shape=(256, 1, 1), lengths=(1.0, 1.0, 1.0)),
            )
        if self.width <= 0:
            raise ValueError("width must be positive")
        if len(self.grid.fft_axes) != 1:
            raise ValueError("InterfaceCase requires a one-dimensional grid")

    def tanh_profile(self):
        """Return the equilibrium tanh interface profile."""
        x = self.grid.x[0]
        midpoint = 0.5 * (self.left_state + self.right_state)
        half_width = 0.5 * (self.right_state - self.left_state)
        return midpoint + half_width * xp.tanh(
            (x - self.center * self.grid.lengths[0]) / self.width
        )

    def initial_field(self):
        """Return a discrete two-phase initial condition."""
        x = self.grid.x[0]
        profile = xp.where(
            x <= self.center * self.grid.lengths[0],
            self.left_state,
            self.right_state,
        )
        return xp.broadcast_to(profile, self.grid.shape)

    def profile_error(self, field):
        """Return the RMS difference from the tanh equilibrium profile."""
        field = xp.asarray(field)
        if field.shape != self.grid.shape:
            raise ValueError(
                "field shape must match grid shape: "
                f"expected {self.grid.shape}, got {field.shape}"
            )
        target = self.tanh_profile()
        residual = field - target
        return xp.sqrt(xp.mean(residual**2))


@dataclass(frozen=True)
class SinusoidalCase:
    """A periodic sinusoidal composition profile for diffusion tests.

    The default profile matches the historical Fick example,
    ``c = 0.125 * sin(4 * pi * x)`` on a unit square with a 256 by 256
    grid. The case only defines the initial condition and observables;
    time integration is provided by a diffusion solver.

    Parameters
    ----------
    grid : Grid, optional
        Grid on which the profile is defined. The default is the standard
        256 by 256 two-dimensional grid.
    amplitude : float, default=0.125
        Initial sinusoidal amplitude.
    mode : int, default=2
        Integer Fourier mode along the x direction. ``mode=2`` gives
        ``sin(4 * pi * x)`` for a unit-length domain.
    """

    grid: Grid | None = None
    amplitude: float = 0.125
    mode: int = 2

    def __post_init__(self):
        if self.grid is None:
            object.__setattr__(self, "grid", Grid())
        if self.amplitude < 0:
            raise ValueError("amplitude must be nonnegative")
        if self.mode <= 0 or int(self.mode) != self.mode:
            raise ValueError("mode must be a positive integer")
        if len(self.grid.fft_axes) != 2:
            raise ValueError("SinusoidalCase requires a two-dimensional grid")

    @property
    def wave_number(self):
        """Angular wave number of the initial profile."""
        return 2.0 * xp.pi * self.mode / self.grid.lengths[0]

    def initial_field(self):
        """Return the initial sinusoidal composition field."""
        profile = self.amplitude * xp.sin(self.wave_number * self.grid.x[0])
        return xp.broadcast_to(profile, self.grid.shape)

    def amplitude_of(self, field):
        """Extract the selected sinusoidal amplitude from a field.

        Parameters
        ----------
        field : array_like
            Real-space scalar field with the same spatial shape as the
            case grid.

        Returns
        -------
        float
            Least-squares projection onto the case's sine profile.
        """
        field = xp.asarray(field)
        if field.shape != self.grid.shape:
            raise ValueError(
                "field shape must match grid shape: "
                f"expected {self.grid.shape}, got {field.shape}"
            )

        basis = xp.broadcast_to(
            xp.sin(self.wave_number * self.grid.x[0]), self.grid.shape
        )
        numerator = xp.sum(field * basis)
        denominator = xp.sum(basis * basis)
        return numerator / denominator

    def expected_amplitude(self, time, decay_rate):
        """Return the analytical exponential amplitude at ``time``.

        Parameters
        ----------
        time : float or array_like
            Nonnegative time value or values.
        decay_rate : float
            Exponential decay rate. May be negative for a mode
            linearized about an unstable (spinodal) composition, in
            which case the amplitude grows rather than decays.

        Returns
        -------
        array_like
            ``amplitude * exp(-decay_rate * time)``.
        """
        return self.amplitude * xp.exp(-decay_rate * xp.asarray(time))