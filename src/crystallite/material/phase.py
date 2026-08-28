from crystallite.backend import xp
from crystallite.material.symmetry import point_group_operations


class Transformation:
    r"""A phase transformation between a parent and product `Solid`.

    Generates the crystallographically distinct variants of a
    transformation (Bain) strain by applying Neumann's principle to the
    *strain itself*: since the parent phase has more symmetry than the
    product, a given transformation strain can sit in several
    symmetry-equivalent orientations relative to the parent lattice --
    these are the martensite variants. The count works out to
    :math:`|G|/|\mathrm{Stab}_G(U)|` by orbit-stabilizer, where `G` is
    the parent's point group and the stabilizer is generically (a
    conjugate of) the product's point group, though this class only
    needs `G` -- it doesn't require the product's point group to be
    known or embeddable in any particular way.

    `parent`/`product` are kept as full `Solid` instances (not just
    their point groups) so later additions -- elastic-energy-weighted
    variant selection, self-accommodation, misfit-driven nucleation --
    have direct access to their other properties (e.g. `stiffness`,
    `misfit_strain`).
    """

    def __init__(self, parent, product):
        self.parent = parent
        self.product = product

    def variants(self, strain, tol=1e-6):
        r"""Distinct crystallographically-equivalent orientations of
        `strain` under the parent phase's point group.

        `strain` is a symmetric (3, 3) transformation strain (e.g. a
        Bain strain, or `product.get("misfit_strain")`) at one point
        along a transformation path (an order parameter running from
        the parent to the product): the same rotation set applies at
        every point along such a path, so call this once per sample
        rather than re-deriving the rotation set each time.
        """
        strain = xp.asarray(strain)
        ops = point_group_operations(self.parent.point_group)
        variants = []
        for op in ops:
            candidate = op @ strain @ op.T
            if not any(xp.allclose(candidate, v, atol=tol) for v in variants):
                variants.append(candidate)
        return variants
