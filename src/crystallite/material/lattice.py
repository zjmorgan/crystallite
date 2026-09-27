r"""Crystal lattices: Miller indices of directions and planes as Cartesian
vectors in the crystal frame.

**Setting.** The crystal axes are placed with :math:`\mathbf{c}` along z and
:math:`\mathbf{a}` in the xz plane (positive x), :math:`\mathbf{b}` completing
a right-handed cell. For conventional cells this matches the standard setting
of :func:`crystallite.material.symmetry.point_group_operations`: principal
axis along z, hexagonal :math:`\mathbf{a}_1` along x, monoclinic unique axis
:math:`\mathbf{b}` along y. Trigonal crystals take hexagonal axes;
:meth:`Lattice.is_compatible` checks a lattice against a point group.

A direction :math:`[uvw]` is :math:`A\,(u,v,w)` and the normal of a plane
:math:`(hkl)` is the reciprocal-lattice vector :math:`B\,(h,k,l)`, where the
columns of :math:`A` are the axes and those of :math:`B = A^{-T}` the
reciprocal axes (without the factor :math:`2\pi`, so that
:math:`|B\,(h,k,l)| = 1/d_{hkl}`). For hexagonal axes, four-index
Miller--Bravais indices :math:`[uvtw]` and :math:`(hkil)` are accepted too.

These vectors are what the orientation colouring takes, e.g.
``ipf_color(lattice.plane_normals(hkl), "6/mmm")``
(:mod:`crystallite.visualization.orientation`).
"""

from __future__ import annotations

import numpy as np

from crystallite.material.symmetry import point_group_operations


class Lattice:
    """Crystal lattice from its cell parameters.

    Parameters
    ----------
    a, b, c : float
        Axis lengths.
    alpha, beta, gamma : float
        Interaxial angles in degrees (``alpha`` between b and c, ``beta``
        between c and a, ``gamma`` between a and b).

    Attributes
    ----------
    direct : ndarray of shape (3, 3)
        Axes a, b, c as columns, in the crystal frame.
    reciprocal : ndarray of shape (3, 3)
        Reciprocal axes a*, b*, c* as columns (``direct.T @ reciprocal`` is
        the identity).
    """

    def __init__(self, a, b, c, alpha=90.0, beta=90.0, gamma=90.0):
        self.a, self.b, self.c = float(a), float(b), float(c)
        self.alpha, self.beta, self.gamma = float(alpha), float(beta), float(gamma)
        ca, cb, cg = np.cos(np.radians([alpha, beta, gamma]))
        sb = np.sin(np.radians(beta))
        bx = (cg - ca * cb) / sb
        by2 = 1.0 - bx * bx - ca * ca
        if min(a, b, c) <= 0.0 or by2 <= 0.0:
            raise ValueError(
                f"no cell with a={a}, b={b}, c={c}, alpha={alpha}, beta={beta}, gamma={gamma}"
            )
        self.direct = np.array(
            [
                [a * sb, b * bx, 0.0],
                [0.0, b * np.sqrt(by2), 0.0],
                [a * cb, b * ca, c],
            ]
        )
        self.direct[np.abs(self.direct) < 1e-15] = 0.0
        self.reciprocal = np.linalg.inv(self.direct).T

    @classmethod
    def cubic(cls, a):
        return cls(a, a, a)

    @classmethod
    def tetragonal(cls, a, c):
        return cls(a, a, c)

    @classmethod
    def orthorhombic(cls, a, b, c):
        return cls(a, b, c)

    @classmethod
    def hexagonal(cls, a, c):
        """Hexagonal axes, for hexagonal and trigonal crystals."""
        return cls(a, a, c, gamma=120.0)

    @classmethod
    def monoclinic(cls, a, b, c, beta):
        """Unique axis b."""
        return cls(a, b, c, beta=beta)

    def __repr__(self):
        return (
            f"Lattice({self.a:g}, {self.b:g}, {self.c:g}, alpha={self.alpha:g}, "
            f"beta={self.beta:g}, gamma={self.gamma:g})"
        )

    @property
    def is_hexagonal(self):
        """Whether the axes are hexagonal (a = b, gamma = 120 degrees, the
        other angles 90), which admits four-index indices."""
        return (
            np.isclose(self.a, self.b)
            and np.isclose(self.gamma, 120.0)
            and np.isclose(self.alpha, 90.0)
            and np.isclose(self.beta, 90.0)
        )

    def _three_index(self, indices, plane):
        indices = np.asarray(indices, dtype=float)
        if indices.shape[-1] == 3:
            return indices
        if indices.shape[-1] != 4:
            raise ValueError(f"Miller indices need 3 or 4 components, got shape {indices.shape}")
        if not self.is_hexagonal:
            raise ValueError("four-index Miller-Bravais indices need hexagonal axes")
        first, second, third, last = np.moveaxis(indices, -1, 0)
        if not np.allclose(first + second + third, 0.0):
            name = "h + k + i" if plane else "u + v + t"
            raise ValueError(f"four-index indices need {name} = 0, got {indices.tolist()}")
        if plane:  # (hkil) -> (hkl)
            return np.stack([first, second, last], axis=-1)
        return np.stack([first - third, second - third, last], axis=-1)  # [uvtw] -> [uvw]

    def directions(self, indices):
        """Cartesian vectors of lattice directions ``[uvw]`` (or ``[uvtw]``),
        shape ``(..., 3)``."""
        return self._three_index(indices, plane=False) @ self.direct.T

    def plane_normals(self, indices):
        """Reciprocal-lattice vectors (plane normals, of length
        ``1 / d``) of planes ``(hkl)`` (or ``(hkil)``), shape ``(..., 3)``."""
        return self._three_index(indices, plane=True) @ self.reciprocal.T

    def d_spacing(self, indices):
        """Interplanar spacing of planes ``(hkl)`` (or ``(hkil)``)."""
        return 1.0 / np.linalg.norm(self.plane_normals(indices), axis=-1)

    def is_compatible(self, point_group, tol=1e-6):
        """Whether the point group's operations (standard setting) map the
        lattice onto itself, i.e. are integer matrices on the axes."""
        ops = point_group_operations(point_group)
        ops = ops.get() if hasattr(ops, "get") else np.asarray(ops)
        on_axes = np.linalg.inv(self.direct) @ ops @ self.direct
        return bool(np.allclose(on_axes, np.round(on_axes), atol=tol))
