"""Dynamic fields and their shared simulation state."""

from collections.abc import Mapping

from crystallite.backend import xp


class Field:
    """A named real-space field defined on a grid.

    Leading axes hold field components; the final three axes match the
    grid's spatial shape.
    """

    def __init__(self, data, grid, name=None):
        data = xp.asarray(data)

        if data.ndim < 3 or data.shape[-3:] != grid.shape:
            raise ValueError(
                "field data must end with grid shape {}: got {}".format(
                    grid.shape, data.shape
                )
            )

        self.data = data
        self.grid = grid
        self.name = name

    @property
    def shape(self):
        """Full array shape, including any component axes."""
        return self.data.shape


class State(Mapping):
    """Dynamic fields defined on one grid at a simulation time."""

    def __init__(self, grid, fields=None, time=0.0, step=0, **named_fields):
        fields = {} if fields is None else dict(fields)
        fields.update(named_fields)

        for name, field in fields.items():
            if not isinstance(field, Field):
                raise TypeError("field {!r} must be a Field".format(name))
            if field.grid is not grid:
                raise ValueError("field {!r} belongs to a different grid".format(name))

        self.grid = grid
        self._fields = fields
        self.time = time
        self.step = step

    def __getitem__(self, name):
        return self._fields[name]

    def __iter__(self):
        return iter(self._fields)

    def __len__(self):
        return len(self._fields)

    def with_fields(self, **fields):
        """Return a new state with replacements while retaining old fields."""
        updated_fields = dict(self._fields)
        updated_fields.update(fields)
        return type(self)(
            self.grid,
            updated_fields,
            time=self.time,
            step=self.step,
        )