from crystallite.backend import xp


class DifferentialOperators:
    """Spectral differential operators on scalar, vector, and tensor fields.

    Parameters
    ----------
    grid : Grid
    """

    def __init__(self, grid):
        self.grid = grid

    def grad(self, f):
        r"""Gradient: inserts a new length-3 axis before the spatial axes.

        .. math::

            \widehat{\nabla f}_{i} = i k_i \hat{f}

        ========  ==================
        Input     Output
        ========  ==================
        scalar    vector
        vector    rank-2 tensor
        rank-n    rank-(n + 1) tensor
        ========  ==================

        Parameters
        ----------
        f : ndarray

        Returns
        -------
        ndarray
        """
        return xp.stack(
            tuple(1j * k * f for k in self.grid.k),
            axis=-4,
        )

    def div(self, f):
        r"""Divergence over the last tensor index.

        .. math::

            \widehat{\nabla \cdot f} = i k_i \hat{f}_{i}

        ========  ==================
        Input     Output
        ========  ==================
        vector    scalar
        tensor    vector
        rank-n    rank-(n - 1) tensor
        ========  ==================

        Parameters
        ----------
        f : ndarray
            Length-3 axis immediately before the spatial axes.

        Returns
        -------
        ndarray

        Raises
        ------
        ValueError
            If `f` lacks that axis.
        """
        if f.ndim < 4 or f.shape[-4] != 3:
            raise ValueError(
                "Divergence requires a length-3 tensor axis "
                "immediately before the spatial axes."
            )

        return sum(
            1j * k * xp.take(f, i, axis=-4)
            for i, k in enumerate(self.grid.k)
        )

    def curl(self, f):
        r"""Curl over the last tensor index.

        .. math::

            \widehat{\nabla \times f}_{i} = i \epsilon_{ijk} k_j \hat{f}_{k}

        ========  ========
        Input     Output
        ========  ========
        vector    vector
        tensor    tensor
        ========  ========

        Parameters
        ----------
        f : ndarray
            Length-3 axis immediately before the spatial axes.

        Returns
        -------
        ndarray

        Raises
        ------
        ValueError
            If `f` lacks that axis.
        """
        if f.ndim < 4 or f.shape[-4] != 3:
            raise ValueError(
                "Curl requires a length-3 tensor axis "
                "immediately before the spatial axes."
            )

        k0, k1, k2 = self.grid.k

        f0 = xp.take(f, 0, axis=-4)
        f1 = xp.take(f, 1, axis=-4)
        f2 = xp.take(f, 2, axis=-4)

        return xp.stack(
            (
                1j * (k1 * f2 - k2 * f1),
                1j * (k2 * f0 - k0 * f2),
                1j * (k0 * f1 - k1 * f0),
            ),
            axis=-4,
        )

    def laplacian(self, f):
        r"""Laplacian of an arbitrary-rank field: :math:`\widehat{\nabla^2 f}
        = -k^2 \hat{f}`.

        Parameters
        ----------
        f : ndarray

        Returns
        -------
        ndarray
        """
        return -self.grid.k2 * f
