import numpy as np
import pytest

from crystallite.material.lattice import Lattice
from crystallite.material.properties import Solid
from crystallite.material.symmetry import point_group_operations
from crystallite.visualization.orientation import InversePoleFigure, ipf_color, miller_label

TRICLINIC = Lattice(3.1, 4.3, 5.2, alpha=81.0, beta=103.0, gamma=97.0)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_cell_geometry():
    lattice = TRICLINIC
    a, b, c = lattice.direct.T
    assert np.linalg.norm(a) == pytest.approx(3.1)
    assert np.linalg.norm(b) == pytest.approx(4.3)
    assert np.linalg.norm(c) == pytest.approx(5.2)
    assert np.degrees(np.arccos(_unit(b) @ _unit(c))) == pytest.approx(81.0)
    assert np.degrees(np.arccos(_unit(c) @ _unit(a))) == pytest.approx(103.0)
    assert np.degrees(np.arccos(_unit(a) @ _unit(b))) == pytest.approx(97.0)
    np.testing.assert_allclose(_unit(c), [0, 0, 1])
    assert a[1] == 0.0 and a[0] > 0.0
    assert np.linalg.det(lattice.direct) > 0.0
    np.testing.assert_allclose(lattice.direct.T @ lattice.reciprocal, np.eye(3), atol=1e-12)


def test_zone_law():
    """The normal of (hkl) is perpendicular to every [uvw] with hu+kv+lw=0."""
    rng = np.random.default_rng(0)
    hkl = rng.integers(-4, 5, size=(200, 3))
    uvw = np.cross(hkl, rng.integers(-4, 5, size=(200, 3)))
    dots = np.einsum("ij,ij->i", TRICLINIC.plane_normals(hkl), TRICLINIC.directions(uvw))
    np.testing.assert_allclose(dots, 0.0, atol=1e-10)


def test_cubic():
    lattice = Lattice.cubic(4.0)
    np.testing.assert_allclose(lattice.directions([1, 1, 1]), [4, 4, 4])
    np.testing.assert_allclose(lattice.plane_normals([1, 1, 1]), [0.25, 0.25, 0.25])
    assert lattice.d_spacing([1, 1, 0]) == pytest.approx(4.0 / np.sqrt(2.0))


def test_d_spacing_formulas():
    a, c = 3.2, 5.1
    hexagonal = Lattice.hexagonal(a, c)
    for h, k, l in [(1, 0, 0), (1, 1, 2), (2, -1, 3)]:
        expected = 1.0 / np.sqrt(4.0 / 3.0 * (h * h + h * k + k * k) / a**2 + l * l / c**2)
        assert hexagonal.d_spacing([h, k, l]) == pytest.approx(expected)
        assert hexagonal.d_spacing([h, k, -h - k, l]) == pytest.approx(expected)
    tetragonal = Lattice.tetragonal(a, c)
    assert tetragonal.d_spacing([1, 1, 1]) == pytest.approx(1.0 / np.sqrt(2.0 / a**2 + 1.0 / c**2))
    b, beta = 4.0, 105.0
    monoclinic = Lattice.monoclinic(a, b, c, beta)
    h, k, l = 1, 2, 1
    s = np.sin(np.radians(beta))
    inverse_d2 = (h * h / a**2 + k * k * s * s / b**2 + l * l / c**2 - 2 * h * l * np.cos(np.radians(beta)) / (a * c)) / s**2
    assert monoclinic.d_spacing([h, k, l]) == pytest.approx(1.0 / np.sqrt(inverse_d2))


def test_hexagonal_four_index():
    lattice = Lattice.hexagonal(3.0, 4.8)
    np.testing.assert_allclose(_unit(lattice.directions([2, -1, -1, 0])), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(_unit(lattice.directions([-1, 2, -1, 0])), [-0.5, np.sqrt(3) / 2, 0], atol=1e-12)
    np.testing.assert_allclose(_unit(lattice.directions([0, 0, 0, 1])), [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(lattice.directions([2, -1, -1, 0]), lattice.directions([1, 0, 0]) * 3.0)
    # prism planes and basal directions share indices: (10-10) is normal to [10-10]
    np.testing.assert_allclose(
        _unit(lattice.plane_normals([1, 0, -1, 0])), _unit(lattice.directions([1, 0, -1, 0])), atol=1e-12
    )
    np.testing.assert_allclose(_unit(lattice.plane_normals([0, 0, 0, 1])), [0, 0, 1], atol=1e-12)
    with pytest.raises(ValueError, match="h \\+ k \\+ i"):
        lattice.plane_normals([1, 0, 0, 0])
    with pytest.raises(ValueError, match="hexagonal axes"):
        Lattice.tetragonal(3.0, 4.0).directions([1, 0, -1, 0])


@pytest.mark.parametrize(
    "lattice, compatible, incompatible",
    [
        (Lattice.cubic(4.0), "m-3m", None),
        (Lattice.tetragonal(3.0, 5.0), "4/mmm", "m-3m"),
        (Lattice.orthorhombic(3.0, 4.0, 5.0), "mmm", "4/mmm"),
        (Lattice.hexagonal(3.0, 5.0), "6/mmm", "m-3m"),
        (Lattice.hexagonal(3.0, 5.0), "-3m", "4/mmm"),
        (Lattice.monoclinic(3.0, 4.0, 5.0, 100.0), "2/m", "mmm"),
        (TRICLINIC, "-1", "2/m"),
        (Lattice(4.0, 4.0, 4.0, 70.0, 70.0, 70.0), "-1", "-3m"),  # rhombohedral axes
    ],
)
def test_compatible_with_standard_setting(lattice, compatible, incompatible):
    assert lattice.is_compatible(compatible)
    if incompatible is not None:
        assert not lattice.is_compatible(incompatible)


def test_invalid_cell():
    with pytest.raises(ValueError):
        Lattice(1.0, 1.0, 1.0, 10.0, 10.0, 150.0)
    with pytest.raises(ValueError):
        Lattice(-1.0, 1.0, 1.0)


def test_equivalent_planes_share_an_ipf_colour():
    """Every plane of a family {hkil} gets one colour, for any c/a, and it
    differs from colouring the raw indices once c/a != 1."""
    lattice = Lattice.hexagonal(3.2, 5.2)
    family = [(1, 0, -1, 1), (0, 1, -1, 1), (-1, 1, 0, 1), (-1, 0, 1, 1), (0, -1, 1, 1), (1, -1, 0, 1)]
    colors = ipf_color(lattice.plane_normals(family), "6/mmm")
    np.testing.assert_allclose(colors, np.broadcast_to(colors[0], colors.shape), atol=1e-12)
    tetragonal = Lattice.tetragonal(3.0, 6.0)
    hkl = np.array([1, 0, 1])
    assert not np.allclose(ipf_color(tetragonal.plane_normals(hkl), "4/mmm"), ipf_color(hkl, "4/mmm"))
    ops = point_group_operations("4/mmm")
    ops = ops.get() if hasattr(ops, "get") else np.asarray(ops)
    normals = tetragonal.plane_normals(hkl) @ ops.transpose(0, 2, 1)
    colors = ipf_color(normals, "4/mmm")
    np.testing.assert_allclose(colors, np.broadcast_to(colors[0], colors.shape), atol=1e-12)


# --- lattice-aware IPF keys -------------------------------------------------


@pytest.mark.parametrize(
    "point_group, lattice",
    [
        ("m-3m", Lattice.cubic(4.0)),
        ("4/mmm", Lattice.tetragonal(3.0, 5.0)),
        ("mmm", Lattice.orthorhombic(3.0, 4.0, 5.0)),
        ("6/mmm", Lattice.hexagonal(3.2, 5.2)),
        ("-3m", Lattice.hexagonal(3.2, 5.2)),
        ("6/m", Lattice.hexagonal(3.2, 5.2)),
        ("m-3", Lattice.cubic(4.0)),
    ],
)
def test_corner_labels_exact_for_any_cell(point_group, lattice):
    """Key corners lie along axes, basal or prism directions or cube
    diagonals, where the orthonormal labels, the lattice directions and the
    plane normals all share indices."""
    default = InversePoleFigure(point_group).labels
    assert InversePoleFigure(point_group, lattice=lattice).labels == default
    assert InversePoleFigure(point_group, lattice=lattice, indices="planes").labels == default


def test_monoclinic_corners_are_plane_normals():
    """With beta != 90 degrees the x corner of the 2/m key is the (100)
    normal, not a rational lattice direction."""
    lattice = Lattice.monoclinic(3.0, 4.0, 5.0, 105.0)
    planes = InversePoleFigure("2/m", lattice=lattice, indices="planes")
    assert planes.labels == ["$100$", r"$\bar{1}00$"]
    directions = InversePoleFigure("2/m", lattice=lattice)
    assert directions.labels[0] == r"$1\,0\,0.16$"
    np.testing.assert_allclose(
        _unit(lattice.plane_normals([1, 0, 0])), planes.vertices[0], atol=1e-12
    )


def test_miller_label_with_lattice():
    lattice = Lattice.hexagonal(3.2, 5.2)
    assert miller_label(lattice.directions([1, 1, -2, 3]), lattice=lattice) == r"$11\bar{2}3$"
    assert miller_label(lattice.plane_normals([1, 0, -1, 2]), lattice=lattice, plane=True) == r"$10\bar{1}2$"
    triclinic = TRICLINIC
    assert miller_label(triclinic.directions([1, -2, 3]), lattice=triclinic) == r"$1\bar{2}3$"
    assert miller_label(triclinic.plane_normals([2, 0, -1]), lattice=triclinic, plane=True) == r"$20\bar{1}$"


def test_key_rejects_incompatible_lattice_and_indices():
    with pytest.raises(ValueError, match="not invariant"):
        InversePoleFigure("m-3m", lattice=Lattice.tetragonal(3.0, 5.0))
    with pytest.raises(ValueError, match="indices"):
        InversePoleFigure("m-3m", indices="hkl")


def test_lattice_does_not_change_colours():
    d = np.random.default_rng(3).normal(size=(200, 3))
    lattice = Lattice.hexagonal(3.2, 5.2)
    np.testing.assert_array_equal(
        InversePoleFigure("6/m", lattice=lattice).color(d), InversePoleFigure("6/m").color(d)
    )


def test_solid_lattice():
    lattice = Lattice.hexagonal(3.21, 5.21)
    solid = Solid("6/mmm", lattice=lattice)
    assert solid.lattice is lattice
    assert Solid("m-3m").lattice is None
    ipf = InversePoleFigure.from_solid(solid, indices="planes")
    assert ipf.lattice is lattice and ipf.group == "6/mmm"
    with pytest.raises(ValueError, match="not invariant"):
        Solid("m-3m", lattice=lattice)
