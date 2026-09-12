import numpy as np
import pytest

from crystallite.field import Field, State
from crystallite.grid import Grid


def test_field_accepts_scalar_and_component_data():
    grid = Grid(shape=(4, 3, 2))

    scalar = Field(np.zeros(grid.shape), grid, name="c")
    vector = Field(np.zeros((3,) + grid.shape), grid, name="P")

    assert scalar.shape == grid.shape
    assert vector.shape == (3,) + grid.shape


def test_field_rejects_incompatible_spatial_shape():
    grid = Grid(shape=(4, 3, 2))

    with pytest.raises(ValueError, match="grid shape"):
        Field(np.zeros((4, 3, 3)), grid)


def test_state_exposes_fields_and_preserves_old_state():
    grid = Grid(shape=(4, 3, 2))
    composition = Field(np.zeros(grid.shape), grid, name="c")
    updated_composition = Field(np.ones(grid.shape), grid, name="c")
    state = State(grid, c=composition, time=0.5, step=2)
    updated_state = state.with_fields(c=updated_composition)

    assert state["c"] is composition
    assert updated_state["c"] is updated_composition
    assert updated_state.time == 0.5
    assert updated_state.step == 2


def test_state_rejects_field_from_another_grid():
    grid = Grid(shape=(4, 3, 2))
    other_grid = Grid(shape=(4, 3, 2))

    with pytest.raises(ValueError, match="different grid"):
        State(grid, c=Field(np.zeros(other_grid.shape), other_grid))