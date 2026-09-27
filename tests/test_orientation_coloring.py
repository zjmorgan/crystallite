import numpy as np
import pytest

from crystallite.grid import Grid
from crystallite.material.symmetry import (
    crystal_system,
    holohedry,
    laue_class,
    point_group_operations,
)
from crystallite.microstructure import (
    Microstructure,
    random_rotations,
    rotation_from_bunge_euler,
    rotation_from_quaternion,
)
from crystallite.visualization.orientation import (
    InversePoleFigure,
    crystal_directions,
    inverse_stereographic,
    ipf_color,
    miller_label,
    orientation_colors,
    orientation_map,
    reduce_directions,
    spherical_triangle_area,
    spherical_triangle_centroid,
    symmetry_operations,
)

POINT_GROUPS = (
    "1", "-1", "2", "m", "2/m", "222", "mm2", "mmm",
    "4", "-4", "4/m", "422", "4mm", "-42m", "-4m2", "4/mmm",
    "3", "-3", "32", "3m", "-3m",
    "6", "-6", "6/m", "622", "6mm", "-6m2", "-62m", "6/mmm",
    "23", "m-3", "432", "-43m", "m-3m",
)

# Groups whose fundamental sector is a triangle bounded by mirror planes.
SUPPORTED = {"mmm", "4/mmm", "6/mmm", "m-3m", "-43m", "-6m2", "-62m"}

CASES = [
    (point_group, symmetry)
    for symmetry in ("laue", "holohedry", "point_group")
    for point_group in POINT_GROUPS
]


def _host(ops):
    return ops.get() if hasattr(ops, "get") else np.asarray(ops)


def _same_group(a, b):
    return len(a) == len(b) and all(np.isclose(b, op, atol=1e-9).all(axis=(1, 2)).any() for op in a)


def _random_directions(n, seed=0):
    d = np.random.default_rng(seed).normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


# --- symmetry levels --------------------------------------------------------


@pytest.mark.parametrize("point_group", POINT_GROUPS)
def test_laue_class_is_point_group_with_inversion(point_group):
    ops = _host(point_group_operations(point_group))
    with_inversion = np.concatenate([ops, -ops])
    unique = [op for i, op in enumerate(with_inversion) if not any(np.allclose(op, p) for p in with_inversion[:i])]
    assert _same_group(np.array(unique), _host(point_group_operations(laue_class(point_group))))


@pytest.mark.parametrize("point_group", POINT_GROUPS)
def test_point_group_is_subgroup_of_its_holohedry(point_group):
    top = _host(point_group_operations(holohedry(point_group)))
    for op in _host(point_group_operations(point_group)):
        assert np.isclose(top, op, atol=1e-9).all(axis=(1, 2)).any()
    assert crystal_system(holohedry(point_group)) == crystal_system(point_group)


def test_unknown_point_group_and_level_rejected():
    with pytest.raises(ValueError):
        laue_class("7")
    with pytest.raises(ValueError):
        symmetry_operations("m-3m", symmetry="cubic")


# --- symmetry reduction -----------------------------------------------------


@pytest.mark.parametrize("point_group, symmetry", CASES)
def test_reduction_is_a_fundamental_sector(point_group, symmetry):
    """Every equivalent image reduces to the same direction (for any group,
    supported colour key or not)."""
    ops = symmetry_operations(point_group, symmetry)
    d = _random_directions(200)
    reduced = reduce_directions(d, ops)
    for op in ops:
        np.testing.assert_allclose(reduce_directions(d @ op.T, ops), reduced, atol=1e-12)


@pytest.mark.parametrize("point_group, symmetry", CASES)
def test_supported_groups(point_group, symmetry):
    group = {"laue": laue_class, "holohedry": holohedry}.get(symmetry, lambda g: g)(point_group)
    if group in SUPPORTED:
        assert InversePoleFigure(point_group, symmetry).group == group
    else:
        with pytest.raises(NotImplementedError):
            InversePoleFigure(point_group, symmetry)


# --- colours ----------------------------------------------------------------

SUPPORTED_CASES = [(g, "point_group") for g in sorted(SUPPORTED)]


@pytest.mark.parametrize("point_group, symmetry", SUPPORTED_CASES)
def test_colors_invariant_under_symmetry(point_group, symmetry):
    ipf = InversePoleFigure(point_group, symmetry)
    d = _random_directions(500, seed=1)
    color = ipf.color(d)
    assert np.all((color >= 0.0) & (color <= 1.0))
    np.testing.assert_allclose(color.max(axis=-1), 1.0)
    for op in ipf.operations:
        np.testing.assert_allclose(ipf.color(d @ op.T), color, atol=1e-12)


@pytest.mark.parametrize("point_group, symmetry", SUPPORTED_CASES)
def test_colors_continuous(point_group, symmetry):
    """Nearby directions get nearby colours, across the sector edges too."""
    ipf = InversePoleFigure(point_group, symmetry)
    d = _random_directions(20000, seed=2)
    step = 1e-6
    nudged = d + step * _random_directions(20000, seed=3)
    assert np.abs(ipf.color(nudged) - ipf.color(d)).max() < 1e3 * step


@pytest.mark.parametrize(
    "point_group, labels",
    [
        ("m-3m", ["$001$", "$101$", "$111$"]),
        ("6/mmm", ["$0001$", r"$2\bar{1}\bar{1}0$", r"$10\bar{1}0$"]),
        ("4/mmm", ["$001$", "$100$", "$110$"]),
        ("mmm", ["$001$", "$100$", "$010$"]),
    ],
)
def test_corners_red_green_blue_and_centroid_white(point_group, labels):
    ipf = InversePoleFigure(point_group)
    assert ipf.labels == labels
    np.testing.assert_allclose(ipf.color(ipf.vertices), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(ipf.color(ipf.white_point), 1.0)


def test_cubic_key_matches_indices():
    """Cubic colours follow R = l - h, G = h - k, B = k for 0 <= k <= h <= l,
    up to the white-point weights and the max normalization."""
    ipf = InversePoleFigure("m-3m")
    d = ipf.reduce(_random_directions(100, seed=4))
    h, k, l = d.T
    assert np.all((0 <= k + 1e-12) & (k <= h + 1e-12) & (h <= l + 1e-12))
    channels = np.stack([l - h, h - k, k], axis=-1) * ipf._weights * np.array([1.0, np.sqrt(2.0), np.sqrt(3.0)])
    np.testing.assert_allclose(ipf.color(d), channels / channels.max(axis=-1, keepdims=True), atol=1e-12)


def test_laue_and_holohedry_levels():
    """m-3 at the holohedry level merges [111] and [-111]; the Laue level
    (m-3) is not a mirror group; -43m distinguishes them at the point-group
    level but not at the Laue level."""
    np.testing.assert_allclose(
        ipf_color([1, 1, 1], "m-3", "holohedry"), ipf_color([-1, 1, 1], "m-3", "holohedry")
    )
    np.testing.assert_allclose(ipf_color([1, 1, 1], "-43m"), ipf_color([-1, 1, 1], "-43m"))
    point_group = InversePoleFigure("-43m", "point_group")
    assert not np.allclose(point_group.color([1, 1, 1]), point_group.color([-1, -1, -1]))


def test_spherical_triangle_area_and_centroid():
    octant = np.eye(3)
    assert spherical_triangle_area(octant) == pytest.approx(np.pi / 2)
    np.testing.assert_allclose(spherical_triangle_centroid(octant), np.ones(3) / np.sqrt(3))
    vertices = InversePoleFigure("m-3m").vertices
    d = _random_directions(400000, seed=5)
    planes = np.cross(np.roll(vertices, -1, 0), np.roll(vertices, -2, 0))
    planes *= np.sign(np.einsum("ij,ij->i", planes, vertices))[:, None]
    inside = d[np.all(d @ planes.T >= 0, axis=-1)]
    assert len(inside) / len(d) == pytest.approx(spherical_triangle_area(vertices) / (4 * np.pi), rel=0.05)
    mean = inside.mean(axis=0)
    np.testing.assert_allclose(spherical_triangle_centroid(vertices), mean / np.linalg.norm(mean), atol=5e-3)


def test_miller_labels():
    assert miller_label([1, -1, 1]) == r"$1\bar{1}1$"
    assert miller_label([0, 0, 2]) == "$001$"
    assert miller_label([-0.5, 3**0.5 / 2, 0], hexagonal=True) == r"$\bar{1}2\bar{1}0$"


# --- orientations -----------------------------------------------------------


def test_identity_and_known_rotation():
    identity = np.eye(3)
    for axis, expected in zip("xyz", np.eye(3)):
        np.testing.assert_allclose(crystal_directions(identity, axis), expected)
    # R_z(90): crystal a lies along sample y, crystal b along sample -x
    rotation = rotation_from_bunge_euler(90.0, 0.0, 0.0)
    np.testing.assert_allclose(crystal_directions(rotation, "y"), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(crystal_directions(rotation, "x"), [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(crystal_directions(rotation, [0, 1, 1]), np.array([1, 0, 1]) / np.sqrt(2), atol=1e-12)


def test_quaternion_sign_and_cubic_equivalent_orientations():
    q = np.random.default_rng(6).normal(size=(50, 4))
    np.testing.assert_allclose(
        orientation_colors(rotation_from_quaternion(q), "m-3m"),
        orientation_colors(rotation_from_quaternion(-q), "m-3m"),
    )
    rotations = random_rotations(50, seed=7)
    ops = symmetry_operations("m-3m")
    proper = ops[np.linalg.det(ops) > 0]
    for op in proper:  # R and R S describe the same crystal
        for direction in "xyz":
            np.testing.assert_allclose(
                orientation_colors(rotations @ op, "m-3m", direction),
                orientation_colors(rotations, "m-3m", direction),
                atol=1e-12,
            )


def test_key_image_uses_the_same_colors():
    ipf = InversePoleFigure("6/mmm")
    image, (x0, x1, y0, y1) = ipf.key_image(resolution=64)
    ny, nx = image.shape[:2]
    x = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    y = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    directions = inverse_stereographic(*np.meshgrid(x, y))
    inside = image[..., 3] > 0
    assert 0.3 < inside.mean() < 1.0
    np.testing.assert_allclose(image[inside, :3], ipf.color(directions[inside]))
    np.testing.assert_allclose(ipf.project(directions[inside]), np.stack(np.meshgrid(x, y), -1)[inside], atol=1e-12)


def test_orientation_map_from_microstructure():
    grid = Grid(shape=(32, 32, 1))
    ms = Microstructure.voronoi(grid, 12, orientations=8, seed=1)
    rgb = orientation_map(ms.slot_field(), ms.slot_orientations, "m-3m", direction="z")
    assert rgb.shape == grid.shape + (3,)
    np.testing.assert_allclose(rgb, orientation_map(ms.grain_ids, ms.orientations, "m-3m"))
    colors = orientation_colors(ms.orientations, "m-3m")
    for grain in range(ms.n_grains):
        np.testing.assert_allclose(rgb[ms.grain_ids == grain], np.broadcast_to(colors[grain], (np.sum(ms.grain_ids == grain), 3)))
    eta = ms.order_parameters(smooth=False)
    np.testing.assert_allclose(orientation_map(eta.argmax(axis=0), ms.slot_orientations, "m-3m"), rgb)


def test_plotting_smoke():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from crystallite.visualization.orientation import (
        plot_ipf,
        plot_ipf_key,
        plot_orientation_maps,
    )

    grid = Grid(shape=(32, 32, 1))
    ms = Microstructure.voronoi(grid, 6, seed=2)
    figure, axes = plot_orientation_maps(ms.grain_ids, ms.orientations, "6/mmm", extent=(0, 1, 0, 1))
    assert len(axes) == 4
    plot_ipf(ms.orientations, "m-3m", "x")
    plot_ipf_key("mmm")
    plt.close("all")
