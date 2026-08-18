from crystallite.backend import xp


_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
"""Index pairs (i, j) for Voigt components 1..6 (11, 22, 33, 23, 13, 12)."""


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


class Solid:
    def __init__(self):
        pass
