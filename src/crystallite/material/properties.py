from crystallite.backend import xp
from crystallite.material.symmetry import (
    magnetic_point_group_operations,
    point_group_operations,
    symmetrize,
)


_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
"""Index pairs (i, j) for Voigt components 1..6 (11, 22, 33, 23, 13, 12)."""


def as_tensor(value, dim=3):
    """Normalize a scalar-or-tensor material property to a square tensor.

    A scalar (isotropic) value becomes ``value * eye(dim)``; a value
    already of shape ``(dim, dim)`` (anisotropic) is returned unchanged.

    Parameters
    ----------
    value : float or array_like
    dim : int, default=3

    Returns
    -------
    ndarray
        Shape ``(dim, dim)``.

    Raises
    ------
    ValueError
        If ``value`` is neither a scalar nor a ``(dim, dim)`` array.
    """
    value = xp.asarray(value)
    if value.ndim == 0:
        return value * xp.eye(dim)
    if value.shape != (dim, dim):
        raise ValueError(f"value must be a scalar or a {dim} by {dim} tensor")
    return value


def voigt_stiffness(medium):
    r"""Rank-4 elastic tensor -> (6, 6) Voigt stiffness matrix.

    Direct reindex, no scaling: with the standard engineering-shear
    strain vector (shear components x2) and an unscaled stress vector,

    .. math::

        \sigma_{\mathrm{V}} = C_{\mathrm{V}} \, \varepsilon_{\mathrm{V}}

    holds using `medium`'s raw :math:`C_{ijkl}` components at the
    Voigt-paired indices.
    """
    i = xp.asarray([p[0] for p in _VOIGT_PAIRS])
    j = xp.asarray([p[1] for p in _VOIGT_PAIRS])
    return medium[i[:, None], j[:, None], i[None, :], j[None, :]]


def voigt_stiffness_to_tensor(c_voigt):
    r"""(6, 6) Voigt stiffness matrix -> (3, 3, 3, 3) elastic tensor.

    Inverse of `voigt_stiffness`: direct reindex, no scaling.
    """
    result = xp.zeros((3, 3, 3, 3), dtype=c_voigt.dtype)
    for row, (i, j) in enumerate(_VOIGT_PAIRS):
        for col, (k, l) in enumerate(_VOIGT_PAIRS):
            value = c_voigt[row, col]
            result[i, j, k, l] = value
            result[j, i, k, l] = value
            result[i, j, l, k] = value
            result[j, i, l, k] = value
    return result


def voigt_compliance_to_tensor(s_voigt):
    r"""(6, 6) Voigt compliance matrix -> (3, 3, 3, 3) compliance tensor.

    Voigt compliance carries factors of 2 (normal-shear entries) and 4
    (shear-shear entries) relative to the raw tensor components
    :math:`S_{ijkl}` in :math:`\varepsilon_{ij} = S_{ijkl}\sigma_{kl}`
    -- dividing them back out is what makes plain matrix inversion of
    `voigt_stiffness`'s output equal the correct tensor compliance.
    """
    scale = xp.where(xp.arange(6) < 3, 1.0, 2.0).astype(s_voigt.dtype)
    s_scaled = s_voigt / (scale[:, None] * scale[None, :])

    result = xp.zeros((3, 3, 3, 3), dtype=s_voigt.dtype)
    for row, (i, j) in enumerate(_VOIGT_PAIRS):
        for col, (k, l) in enumerate(_VOIGT_PAIRS):
            value = s_scaled[row, col]
            result[i, j, k, l] = value
            result[j, i, k, l] = value
            result[i, j, l, k] = value
            result[j, i, l, k] = value

    return result


def voigt_vector(tensor):
    r"""(3, 3, 3) tensor, symmetric in its first two indices -> (6, 3) matrix.

    Row = Voigt-paired :math:`(i, j)`, column = the free third index.
    Rank-3 analogue of `voigt_stiffness`, e.g. for a piezoelectric- or
    piezomagnetic-type tensor :math:`Q_{ijk}` with :math:`Q_{ijk}=Q_{jik}`.
    """
    i = xp.asarray([p[0] for p in _VOIGT_PAIRS])
    j = xp.asarray([p[1] for p in _VOIGT_PAIRS])
    return tensor[i, j, :]


def voigt_vector_to_tensor(matrix):
    r"""(6, 3) matrix -> (3, 3, 3) tensor, symmetric in the first two indices.

    Inverse of `voigt_vector`: direct reindex, no scaling.
    """
    result = xp.zeros((3, 3, 3), dtype=matrix.dtype)
    for row, (i, j) in enumerate(_VOIGT_PAIRS):
        result[i, j, :] = matrix[row, :]
        result[j, i, :] = matrix[row, :]
    return result


_SYMMETRY_TOL = 1e-8


def _outer(a, b):
    return a.reshape(a.shape + (1,) * b.ndim) * b.reshape((1,) * a.ndim + b.shape)


class TensorBlock:
    r"""One leg of a property tensor (Nye's building blocks, plus a
    general pair for gradient-energy terms): a Cartesian vector (rank
    1, 3 components), a Voigt-symmetric pair (rank 2, symmetric, 6
    components, e.g. strain), or a general (non-symmetric) pair (rank
    2, 9 components, e.g. a field gradient :math:`\partial_j P_i`,
    where the two axes are physically distinct roles and not
    interchangeable). `axial` marks a pseudovector/pseudotensor axis
    (e.g. a magnetic field or magnetization): it picks up an extra
    :math:`\det(R)` sign under every point-group operation. Pass a
    single bool to apply to every axis of the block, or (for
    `"matrix"`) a tuple to set each axis independently -- e.g.
    `("matrix", axial=(True, False))` for :math:`\partial_j M_i`,
    where :math:`M_i` is axial but the gradient direction :math:`j` is
    an ordinary polar index.
    """

    _DIMS = {"vector": 3, "pair": 6, "matrix": 9}

    def __init__(self, kind, axial=False):
        if kind not in self._DIMS:
            raise ValueError(f"kind must be one of {sorted(self._DIMS)}, not {kind!r}")
        self.kind = kind
        self.n_axes = 1 if kind == "vector" else 2
        self.dim = self._DIMS[kind]
        self.axial = (axial,) * self.n_axes if isinstance(axial, bool) else tuple(axial)
        if len(self.axial) != self.n_axes:
            raise ValueError(f"axial must have {self.n_axes} entries for kind {kind!r}")

    def basis(self, index):
        """Local (3,)*n_axes basis tensor for 0-indexed `index`."""
        local = xp.zeros((3,) * self.n_axes)
        if self.kind == "vector":
            local[index] = 1.0
        elif self.kind == "pair":
            i, j = _VOIGT_PAIRS[index]
            local[i, j] = local[j, i] = 1.0
        else:
            local[divmod(index, 3)] = 1.0
        return local

    def index_tuple(self, index):
        """0-indexed Cartesian axis values selecting `index`'s representative."""
        if self.kind == "vector":
            return (index,)
        if self.kind == "pair":
            return _VOIGT_PAIRS[index]
        return divmod(index, 3)


class PropertyTensor:
    r"""General crystal property tensor: 1 or 2 `TensorBlock` legs, an
    overall time-reversal parity, and (for 2 same-dimension blocks) an
    optional major symmetry -- the two legs are physically
    interchangeable, e.g. elastic stiffness's two strain-index pairs.

    This is the single mechanism behind `Solid.set`/`Solid.get`: apply
    Neumann's principle (`symmetrize`) to each named basis tensor, then
    solve/validate a linear system against the resulting projector --
    one engine for every rank and shape in Nye's tables, rather than
    bespoke code per property.
    """

    _ALPHABETS = {3: "123", 6: "123456", 9: "123456789"}
    _projector_cache = {}

    def __init__(self, *blocks, time_reversal_odd=False, major_symmetric=False):
        if len(blocks) not in (1, 2):
            raise ValueError("PropertyTensor supports 1 or 2 blocks")
        if major_symmetric and (len(blocks) != 2 or blocks[0].dim != blocks[1].dim):
            raise ValueError("major_symmetric needs 2 same-dimension blocks")

        self.blocks = blocks
        self.time_reversal_odd = time_reversal_odd
        self.major_symmetric = major_symmetric
        self.rank = sum(b.n_axes for b in blocks)

        axes, axis = [], 0
        for block in blocks:
            axes.append(tuple(range(axis, axis + block.n_axes)))
            axis += block.n_axes
        self.axial_axes = tuple(
            a
            for block, axs in zip(blocks, axes)
            for is_axial, a in zip(block.axial, axs)
            if is_axial
        )

        if len(blocks) == 1:
            self.keys = list(self._ALPHABETS[blocks[0].dim])
        elif major_symmetric:
            alphabet = self._ALPHABETS[blocks[0].dim]
            self.keys = [r + c for r in alphabet for c in alphabet if r <= c]
        else:
            self.keys = [
                r + c
                for r in self._ALPHABETS[blocks[0].dim]
                for c in self._ALPHABETS[blocks[1].dim]
            ]

    def _local_indices(self, key):
        return tuple(int(ch) - 1 for ch in key)

    def basis_tensor(self, key):
        """`keys` entry -> full (3,)*rank basis tensor."""
        indices = self._local_indices(key)
        if len(self.blocks) == 1:
            return self.blocks[0].basis(indices[0])
        if not self.major_symmetric:
            return _outer(self.blocks[0].basis(indices[0]), self.blocks[1].basis(indices[1]))
        i, j = indices
        tensor = _outer(self.blocks[0].basis(i), self.blocks[1].basis(j))
        if i != j:
            tensor = tensor + _outer(self.blocks[0].basis(j), self.blocks[1].basis(i))
        return tensor

    def extract(self, tensor):
        """Full (3,)*rank tensor -> dict of `keys` -> value."""
        result = {}
        for key in self.keys:
            indices = self._local_indices(key)
            full_index = self.blocks[0].index_tuple(indices[0])
            if len(self.blocks) == 2:
                full_index += self.blocks[1].index_tuple(indices[1])
            result[key] = tensor[full_index]
        return result

    def to_tensor(self, values):
        """dict of `keys` -> value -> full (3,)*rank tensor."""
        tensor = xp.zeros((3,) * self.rank)
        for key, value in values.items():
            tensor = tensor + value * self.basis_tensor(key)
        return tensor

    def to_matrix(self, tensor):
        """Full (3,)*rank tensor -> conventional Voigt-like matrix.

        Shape `(dim,)` for 1 block, `(dim0, dim1)` for 2 -- always the
        full (possibly redundant) matrix, independent of
        `major_symmetric`, matching `voigt_stiffness`'s convention.
        """
        if len(self.blocks) == 1:
            return xp.asarray(
                [tensor[self.blocks[0].index_tuple(i)] for i in range(self.blocks[0].dim)]
            )
        return xp.asarray(
            [
                [
                    tensor[self.blocks[0].index_tuple(i) + self.blocks[1].index_tuple(j)]
                    for j in range(self.blocks[1].dim)
                ]
                for i in range(self.blocks[0].dim)
            ]
        )

    def from_matrix(self, matrix):
        """Conventional Voigt-like matrix -> full (3,)*rank tensor."""
        tensor = xp.zeros((3,) * self.rank)
        if len(self.blocks) == 1:
            for i in range(self.blocks[0].dim):
                tensor = tensor + matrix[i] * self.blocks[0].basis(i)
            return tensor
        for i in range(self.blocks[0].dim):
            for j in range(self.blocks[1].dim):
                tensor = tensor + matrix[i, j] * _outer(
                    self.blocks[0].basis(i), self.blocks[1].basis(j)
                )
        return tensor

    def projector(self, group_key, operations, time_reversals=None):
        cache_key = (id(self), group_key)
        if cache_key not in self._projector_cache:
            columns = []
            for key in self.keys:
                projected = symmetrize(
                    self.basis_tensor(key),
                    operations,
                    time_reversals=time_reversals,
                    axial_axes=self.axial_axes,
                    time_reversal_odd=self.time_reversal_odd,
                )
                extracted = self.extract(projected)
                columns.append(xp.asarray([extracted[k] for k in self.keys]))
            self._projector_cache[cache_key] = xp.stack(columns, axis=1)
        return self._projector_cache[cache_key]

    def resolve(self, group_key, operations, given, time_reversals=None):
        """Fill `None` entries in `given` (`keys` -> value|`None`) via
        the projector's fixed-point equation; validate the rest.
        """
        keys = self.keys
        n = len(keys)
        known = [i for i, k in enumerate(keys) if given[k] is not None]
        unknown = [i for i, k in enumerate(keys) if given[k] is None]
        v = xp.asarray([0.0 if given[k] is None else given[k] for k in keys])

        a = self.projector(group_key, operations, time_reversals) - xp.eye(n)
        if unknown:
            rhs = -a[:, known] @ v[known] if known else xp.zeros(n)
            solution, *_ = xp.linalg.lstsq(a[:, unknown], rhs, rcond=None)
            v[unknown] = solution

        if xp.max(xp.abs(a @ v)) > _SYMMETRY_TOL:
            raise ValueError(
                f"given constants are inconsistent with the required "
                f"symmetry: {dict(zip(keys, v.tolist()))}"
            )
        return dict(zip(keys, v.tolist()))

    def rotate(self, tensor, rotation, time_reversed=False):
        r"""Transform `tensor` to a rotated frame -- the general,
        arbitrary-rotation analogue of `resolve`'s group-averaged
        `projector` (a single `rotation`, not a group, and no
        symmetry/fixed-point solving: just apply Nye's transformation
        law once). `rotation` is any 3x3 proper or improper orthogonal
        matrix (e.g. built from Euler angles or Rodrigues' formula);
        `time_reversed` applies the extra sign for a T-odd property
        under an operation combined with time reversal.
        """
        return symmetrize(
            tensor,
            xp.asarray([rotation]),
            time_reversals=xp.asarray([1.0 if time_reversed else 0.0]),
            axial_axes=self.axial_axes,
            time_reversal_odd=self.time_reversal_odd,
        )


_PAIR = TensorBlock("pair")
_AXIAL_PAIR = TensorBlock("pair", axial=True)
_VECTOR = TensorBlock("vector")
_AXIAL_VECTOR = TensorBlock("vector", axial=True)

# Nye's property-tensor catalogue (*Physical Properties of Crystals*), as
# (spec, constant-name prefix, needs a magnetic point group) -- see
# `PropertyTensor`'s docstring for the general mechanism. Specs are
# deliberately shared across properties with identical tensor character
# (e.g. every "direct transport" property below, or every rank-4
# strain/field-squared coupling).

_ELASTIC = PropertyTensor(_PAIR, TensorBlock("pair"), major_symmetric=True)
"""Stiffness/compliance: strain-strain, polar, T-even, major-symmetric
(derivable from a quadratic energy)."""

_STRAIN_FIELD_SQUARED = PropertyTensor(_PAIR, TensorBlock("pair"))
"""Magnetostriction/electrostriction/elasto-optic/piezo-optic: a
symmetric pair (strain, stress, or the optical indicatrix change)
coupled to another symmetric pair (a field-squared/direction-cosine
product, or strain/stress) -- minor symmetry only, no major symmetry
(the two pairs are physically distinct quantities). `axial_axes` would
be a documented no-op here regardless (a field-squared pair's two legs
share the same character, so `det(R)^2=1`), so every use of this spec
takes the same (polar) block."""

_PIEZOELECTRIC = PropertyTensor(_PAIR, _VECTOR)
"""Piezoelectricity: strain coupled to E/D, an ordinary polar,
time-reversal-even vector."""

_PIEZOMAGNETIC = PropertyTensor(_PAIR, _AXIAL_VECTOR, time_reversal_odd=True)
"""Piezomagnetism: strain coupled to H, an axial vector; since strain is
T-even, the coupling itself must be time-reversal-odd -- needs the
exact magnetic point group, not just the ordinary (Laue) one."""

_PYROELECTRIC = PropertyTensor(_VECTOR)
r"""Pyroelectricity: :math:`P_i = p_i\,\Delta T` -- an ordinary polar,
time-reversal-even vector."""

_PYROMAGNETIC = PropertyTensor(_AXIAL_VECTOR, time_reversal_odd=True)
r"""Pyromagnetism: :math:`M_i = p_i\,\Delta T` -- M is axial and
time-reversal-odd while :math:`\Delta T` is T-even, so `p` itself must
be T-odd; needs the exact magnetic point group, like piezomagnetism."""

_MAGNETOELECTRIC = PropertyTensor(_VECTOR, _AXIAL_VECTOR, time_reversal_odd=True)
r"""Magnetoelectricity: couples a polar and an axial vector, e.g.
:math:`P_i = \alpha_{ij}H_j`; P is T-even, H is T-odd, so
:math:`\alpha` itself must be T-odd -- needs the exact magnetic point
group. Not major-symmetric: the two legs are physically distinct
(polar vs. axial), not interchangeable."""

_TRANSPORT = PropertyTensor(_VECTOR, TensorBlock("vector"), major_symmetric=True)
r"""Electric/thermal conductivity, diffusivity, electric susceptibility,
thermal expansion, misfit strain: polar, T-even, and symmetric -- by
Onsager reciprocity for the transport properties (absent an applied
magnetic field), and intrinsically (strain is symmetric by definition)
for thermal expansion and misfit strain. Misfit strain is the
composition-driven eigenstrain :math:`\varepsilon^0_{ij} =
\alpha_{ij}(c-c_0)` (Cahn-Hilliard elasticity) -- identical shape to
thermal expansion, just driven by composition instead of temperature."""

_MAGNETIC_TRANSPORT = PropertyTensor(_AXIAL_VECTOR, TensorBlock("vector", axial=True), major_symmetric=True)
"""Magnetic susceptibility: M and H are both axial, but `det(R)^2=1`
makes this transform identically to `_TRANSPORT` -- kept as a separate,
explicitly-axial spec for documentation, not because it computes
anything different."""

_CROSS = PropertyTensor(_VECTOR, TensorBlock("vector"))
"""Seebeck coefficient: polar, T-even, but *not* symmetrized -- Onsager's
theorem here relates the Seebeck and Peltier tensors to each other
(transposes, via the Kelvin relation), it does not make either one
self-symmetric."""

_MATRIX = TensorBlock("matrix")

_POLAR_GRADIENT = PropertyTensor(_MATRIX, _MATRIX, major_symmetric=True)
r"""Polarization gradient energy: :math:`\frac{1}{2}g_{ijkl}\,
(\partial_j P_i)(\partial_l P_k)`. Each :math:`(\partial_j P_i)` leg is
a *general* (non-symmetric) pair, unlike strain -- :math:`i` (which P
component) and :math:`j` (which spatial direction) are physically
distinct roles, not interchangeable, so there's no Voigt reduction
within a leg. Major-symmetric across the two legs (mixed partials
commute), polar, T-even."""

_MAGNETIC_GRADIENT = PropertyTensor(
    TensorBlock("matrix", axial=(True, False)),
    TensorBlock("matrix", axial=(True, False)),
    major_symmetric=True,
)
r"""Magnetization gradient (exchange-stiffness) energy: same shape as
`_POLAR_GRADIENT` but :math:`M_i` is axial and T-odd while
:math:`\partial_j` stays polar and T-even; :math:`M_iM_k` (product of
two T-odd factors) is T-even overall and `axial_axes` is a documented
no-op (`det(R)^2=1`), exactly the `_STRAIN_FIELD_SQUARED` pattern --
ordinary point group, no magnetic group needed."""

_PROPERTIES = {
    "stiffness": (_ELASTIC, "c", False),
    "magnetostriction": (_STRAIN_FIELD_SQUARED, "q", False),
    "electrostriction": (_STRAIN_FIELD_SQUARED, "m", False),
    "elasto_optic": (_STRAIN_FIELD_SQUARED, "p", False),
    "piezo_optic": (_STRAIN_FIELD_SQUARED, "pi", False),
    "piezoelectricity": (_PIEZOELECTRIC, "d", False),
    "piezomagnetism": (_PIEZOMAGNETIC, "q", True),
    "pyroelectricity": (_PYROELECTRIC, "p", False),
    "pyromagnetism": (_PYROMAGNETIC, "p", True),
    "magnetoelectricity": (_MAGNETOELECTRIC, "alpha", True),
    "electrical_conductivity": (_TRANSPORT, "sigma", False),
    "thermal_conductivity": (_TRANSPORT, "kappa", False),
    "diffusivity": (_TRANSPORT, "d", False),
    "electric_susceptibility": (_TRANSPORT, "chie", False),
    "magnetic_susceptibility": (_MAGNETIC_TRANSPORT, "chim", False),
    "thermal_expansion": (_TRANSPORT, "alpha", False),
    "misfit_strain": (_TRANSPORT, "alpha", False),
    "seebeck_coefficient": (_CROSS, "s", False),
    "concentration_gradient_energy": (_TRANSPORT, "kappa", False),
    "polarization_gradient_energy": (_POLAR_GRADIENT, "g", False),
    "magnetization_gradient_energy": (_MAGNETIC_GRADIENT, "a", False),
}

_INVERSE_PROPERTIES = {
    "compliance": ("stiffness", "s"),
    "electrical_resistivity": ("electrical_conductivity", "rho"),
    "thermal_resistivity": ("thermal_conductivity", "r"),
}
"""Properties defined as the Voigt-matrix inverse of a primary one
(name -> (primary name, constant-name prefix)), e.g.
`compliance = inv(stiffness)`, `electrical_resistivity =
inv(electrical_conductivity)`. Reuses the primary property's own
`PropertyTensor` -- same symmetry, just a matrix inversion at the end.
Diffusivity has no such standard named inverse, so it is excluded."""

_AFFINE_PROPERTIES = {
    "electric_permittivity": ("electric_susceptibility", "epsilon", 1.0),
    "magnetic_permeability": ("magnetic_susceptibility", "mu", 1.0),
}
"""Properties defined as a primary one plus an isotropic offset
(name -> (primary name, constant-name prefix, offset)), e.g. relative
permittivity :math:`\\varepsilon_r = 1 + \\chi_e`. The offset is
isotropic (a multiple of the identity), so it never affects which
entries symmetry forces to vanish or repeat -- it only shifts the
diagonal, which is why this is "just a rescaling" and the primary
property's own symmetry pattern applies unchanged."""


class Solid:
    r"""Anisotropic medium: a general storage/transformation framework
    for crystal property tensors (Nye, *Physical Properties of
    Crystals*). Every property is a `PropertyTensor` registered in
    `_PROPERTIES` -- see `TensorBlock`/`PropertyTensor` for the
    mechanism. `set(name, **values)`/`get(name)`/`rotate(name, ...)`
    are generic; there is no bespoke code per property. A few
    properties are instead an alternate view of a primary one:
    `set_inverse(name, ...)`/`get_inverse(name)` handle the Voigt-matrix
    inverse (see `_INVERSE_PROPERTIES`) -- `compliance` (inverse of
    `stiffness`), `electrical_resistivity` (inverse of
    `electrical_conductivity`), `thermal_resistivity` (inverse of
    `thermal_conductivity`); `set_affine(name, ...)`/`get_affine(name)`
    handle a primary property plus an isotropic offset (see
    `_AFFINE_PROPERTIES`) -- `electric_permittivity` (= 1 +
    `electric_susceptibility`), `magnetic_permeability` (= 1 +
    `magnetic_susceptibility`). `diffusivity` has no standard named
    inverse or affine counterpart.

    `point_group` (one of the 32 crystallographic point groups) governs
    most properties. A few -- anything time-reversal-odd, i.e. coupling
    to a magnetic quantity linearly rather than quadratically:
    `piezomagnetism`, `pyromagnetism`, `magnetoelectricity` -- instead
    need `uni_number` (spglib's magnetic-space-group database number,
    1..1651 -- look one up with `spglib.get_magnetic_spacegroup_type`
    or `crystallite.material.symmetry.list_magnetic_point_groups`),
    since the exact magnetic point group, not just the ordinary one,
    is what constrains them.

    Registered properties: `stiffness`, `magnetostriction`,
    `electrostriction`, `elasto_optic`, `piezo_optic`,
    `piezoelectricity`, `piezomagnetism`, `pyroelectricity`,
    `pyromagnetism`, `magnetoelectricity`, `electrical_conductivity`,
    `thermal_conductivity`, `diffusivity`, `electric_susceptibility`,
    `magnetic_susceptibility`, `thermal_expansion`, `misfit_strain`
    (the composition-driven eigenstrain :math:`\varepsilon^0_{ij} =
    \alpha_{ij}(c-c_0)` of Cahn-Hilliard elasticity -- same shape as
    `thermal_expansion`, driven by composition instead of temperature),
    `seebeck_coefficient`, `concentration_gradient_energy`,
    `polarization_gradient_energy`, `magnetization_gradient_energy`
    (the last three are Cahn-Hilliard/Landau-Ginzburg-type gradient
    energy coefficients, e.g. :math:`\frac{1}{2}\kappa_{ij}(\partial_i
    c)(\partial_j c)` for a scalar concentration field `c`).
    """

    def __init__(self, point_group="1", uni_number=None):
        point_group_operations(point_group)  # validates point_group
        if uni_number is not None:
            magnetic_point_group_operations(uni_number)  # validates uni_number
        self.point_group = point_group
        self.uni_number = uni_number
        for name in _PROPERTIES:
            setattr(self, name, None)

    def _group(self, magnetic):
        if magnetic:
            if self.uni_number is None:
                raise ValueError(
                    "this property needs a magnetic point group -- pass "
                    "uni_number to Solid()"
                )
            ops, flags = magnetic_point_group_operations(self.uni_number)
            return ops, flags, ("magnetic", self.uni_number)
        return point_group_operations(self.point_group), None, ("ordinary", self.point_group)

    def set(self, name, **values):
        """Set property `name` from its independent named constants.

        Constants required to vanish or repeat by symmetry may be left
        unset (`None`); given explicitly, they must be consistent with
        that symmetry. Keys are `f"{prefix}{key}"` for the property's
        own constant prefix (e.g. `"c11"` for stiffness, `"q11"` for
        magnetostriction) and each of its `PropertyTensor.keys`.
        """
        spec, prefix, magnetic = _PROPERTIES[name]
        ops, flags, group_key = self._group(magnetic)
        given = {key: values.pop(f"{prefix}{key}", None) for key in spec.keys}
        if values:
            raise ValueError(f"unrecognized constants for {name!r}: {sorted(values)}")
        resolved = spec.resolve(group_key, ops, given, time_reversals=flags)
        setattr(self, name, spec.to_tensor(resolved))
        return self

    def get(self, name):
        """Return property `name`'s conventional Voigt-like matrix.

        Shape `(dim,)` for a lone vector, `(dim0, dim1)` for two blocks
        (e.g. `(6, 6)` Voigt for stiffness, `(6, 3)` for piezomagnetism,
        `(3, 3)` for conductivity) -- always the full matrix, per
        `PropertyTensor.to_matrix`.
        """
        spec = _PROPERTIES[name][0]
        tensor = getattr(self, name)
        if tensor is None:
            raise ValueError(f"{name!r} has not been set")
        return spec.to_matrix(tensor)

    def rotate(self, name, rotation, time_reversed=False):
        r"""Return property `name` transformed to a rotated frame.

        `rotation` is any 3x3 proper or improper orthogonal matrix
        (crystal-frame axes expressed in the new frame), e.g. built
        from Euler angles or Rodrigues' formula -- this does not touch
        or require `point_group`/`uni_number`, it just applies Nye's
        transformation law once (see `PropertyTensor.rotate`). Returns
        the conventional Voigt-like matrix, like `get`.
        """
        spec = _PROPERTIES[name][0]
        tensor = getattr(self, name)
        if tensor is None:
            raise ValueError(f"{name!r} has not been set")
        return spec.to_matrix(spec.rotate(tensor, rotation, time_reversed=time_reversed))

    def set_inverse(self, name, **values):
        """Set the primary property behind inverse property `name`
        (e.g. `"compliance"` sets `stiffness`, `"electrical_resistivity"`
        sets `electrical_conductivity`) from `name`'s own independent
        named constants -- see `_INVERSE_PROPERTIES`.

        Same symmetry as `set(primary, ...)` (it is the *same*
        `PropertyTensor`); the resulting Voigt-like matrix is inverted
        directly to the primary property, e.g. `stiffness =
        inv(compliance)`.
        """
        primary, prefix = _INVERSE_PROPERTIES[name]
        spec, _, magnetic = _PROPERTIES[primary]
        ops, flags, group_key = self._group(magnetic)
        given = {key: values.pop(f"{prefix}{key}", None) for key in spec.keys}
        if values:
            raise ValueError(f"unrecognized constants for {name!r}: {sorted(values)}")
        resolved = spec.resolve(group_key, ops, given, time_reversals=flags)
        inverse_matrix = spec.to_matrix(spec.to_tensor(resolved))
        setattr(self, primary, spec.from_matrix(xp.linalg.inv(inverse_matrix)))
        return self

    def get_inverse(self, name):
        """Return inverse property `name`'s Voigt-like matrix, i.e.
        `inv(get(primary))` -- see `_INVERSE_PROPERTIES`.
        """
        primary = _INVERSE_PROPERTIES[name][0]
        return xp.linalg.inv(self.get(primary))

    def set_affine(self, name, **values):
        """Set the primary property behind affine property `name`
        (e.g. `"electric_permittivity"` sets `electric_susceptibility`)
        from `name`'s own independent named constants -- see
        `_AFFINE_PROPERTIES`. The isotropic `offset` is subtracted from
        diagonal entries before resolving against the primary
        property's own symmetry (an isotropic shift never affects that
        symmetry), and re-added by `get_affine`.
        """
        primary, prefix, offset = _AFFINE_PROPERTIES[name]
        spec, _, magnetic = _PROPERTIES[primary]
        ops, flags, group_key = self._group(magnetic)
        given = {}
        for key in spec.keys:
            raw = values.pop(f"{prefix}{key}", None)
            given[key] = None if raw is None else raw - offset * (key[0] == key[1])
        if values:
            raise ValueError(f"unrecognized constants for {name!r}: {sorted(values)}")
        resolved = spec.resolve(group_key, ops, given, time_reversals=flags)
        setattr(self, primary, spec.to_tensor(resolved))
        return self

    def get_affine(self, name):
        """Return affine property `name`'s Voigt-like matrix, i.e.
        `get(primary) + offset * I` -- see `_AFFINE_PROPERTIES`.
        """
        primary, _, offset = _AFFINE_PROPERTIES[name]
        matrix = self.get(primary)
        return matrix + offset * xp.eye(matrix.shape[0])
