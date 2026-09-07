import numpy as np

from crystallite.material.phase import Transformation, Variant
from crystallite.material.properties import Solid


def test_variants_include_transformation_rotation_and_eigenstrain():
    parent = Solid("m-3m")
    product = Solid("m-3m")
    transformation = Transformation(parent, product)
    strain = np.diag((0.01, 0.02, 0.03))

    variants = transformation.variants(strain)

    assert len(variants) == 6
    assert all(isinstance(variant, Variant) for variant in variants)
    assert all(variant.transformation is transformation for variant in variants)
    assert all(variant.rotation.shape == (3, 3) for variant in variants)
    assert all(variant.eigenstrain.shape == (3, 3) for variant in variants)

    eigenstrains = [np.asarray(variant.eigenstrain) for variant in variants]
    assert any(np.allclose(eigenstrain, strain) for eigenstrain in eigenstrains)


def test_equivalent_strains_produce_one_variant():
    parent = Solid("m-3m")
    product = Solid("m-3m")
    transformation = Transformation(parent, product)

    variants = transformation.variants(np.eye(3))

    assert len(variants) == 1