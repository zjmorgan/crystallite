"""Verification cases based on the Fick presentation examples."""

from dataclasses import dataclass

from crystallite.backend import xp
from crystallite.phase import Grid


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
            Positive exponential decay rate.

        Returns
        -------
        array_like
            ``amplitude * exp(-decay_rate * time)``.
        """
        if decay_rate < 0:
            raise ValueError("decay_rate must be nonnegative")
        return self.amplitude * xp.exp(-decay_rate * xp.asarray(time))