# import spglib

# cell = (lattice, positions, numbers)

# sym = spglib.get_symmetry(cell, symprec=1e-5)

# rotations = sym["rotations"]  # (N, 3, 3)
# translations = sym["translations"]

# # Unique point-group operators
# point_ops = np.unique(rotations, axis=0)

# print(len(point_ops))

# for R in point_ops:
#     print(R)
