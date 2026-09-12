"""Mixture rules for scalar and tensor-valued material properties."""

from crystallite.backend import xp


def arithmetic(a, b, fraction):
    """Return the arithmetic mixture of two properties.

    Parameters
    ----------
    a, b : array_like
        Properties of the two constituents. They may be scalars or arrays,
        including anisotropic property tensors (e.g. ``(3, 3)``).
    fraction : array_like
        Fraction of constituent ``a``. The fraction of ``b`` is
        ``1 - fraction``. May itself be a scalar or a spatial field (e.g.
        a composition field, unrelated in shape to a tensor-valued ``a``
        or ``b``) -- broadcast against the tensor's leading axes rather
        than numpy's default trailing-axis alignment, matching the
        leading-tensor-axis convention used for property fields elsewhere
        (e.g. :class:`crystallite.mass_diffusion.MassDiffusion`).

    Returns
    -------
    array_like
        The elementwise arithmetic mixture.
    """
    a = xp.asarray(a)
    b = xp.asarray(b)
    fraction = xp.asarray(fraction)
    tensor_ndim = max(a.ndim, b.ndim)
    if tensor_ndim and fraction.ndim:
        pad = (1,) * fraction.ndim
        if a.ndim:
            a = a.reshape(a.shape + pad)
        if b.ndim:
            b = b.reshape(b.shape + pad)
        fraction = fraction.reshape((1,) * tensor_ndim + fraction.shape)
    return fraction * a + (1.0 - fraction) * b


def harmonic(a, b, fraction):
    """Return the harmonic mixture of two properties.

    For scalar properties this computes the weighted reciprocal. For
    matrix-valued properties, the reciprocal is a matrix inverse, so the
    result preserves the tensor character of anisotropic properties.

    Parameters
    ----------
    a, b : array_like
        Properties of the two constituents. Scalars are mixed with scalar
        reciprocals; square matrices are mixed with matrix inverses.
    fraction : array_like
        Fraction of constituent ``a``. The fraction of ``b`` is
        ``1 - fraction``.

    Returns
    -------
    array_like
        The weighted harmonic mixture.

    Raises
    ------
    ValueError
        If either property is a non-square matrix.
    """
    a = xp.asarray(a)
    b = xp.asarray(b)
    fraction = xp.asarray(fraction)

    if a.ndim == 0 and b.ndim == 0:
        return 1.0 / (fraction / a + (1.0 - fraction) / b)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("tensor properties must be square matrices")
    if a.shape[0] != a.shape[1] or b.shape[0] != b.shape[1]:
        raise ValueError("tensor properties must be square matrices")

    inverse = fraction * xp.linalg.inv(a) + (1.0 - fraction) * xp.linalg.inv(b)
    return xp.linalg.inv(inverse)