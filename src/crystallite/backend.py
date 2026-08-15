try:
    import cupy as cp
except ImportError:
    cp = None

import numpy as np

xp = cp if cp is not None else np
