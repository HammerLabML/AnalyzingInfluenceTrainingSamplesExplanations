import numpy as np


def get_f_dist(desc, epsilon=1e-3):
    if desc == "l0":
        return lambda x_orig, x_cf: np.sum(np.abs(x_orig - x_cf) < epsilon)
    elif desc == "l1":
        return lambda x_orig, x_cf: np.sum(np.abs(x_orig - x_cf))
    elif desc == "l2":
        return lambda x_orig, x_cf: np.sum(np.square(x_orig - x_cf))
    else:
        raise ValueError(f"Unknown distance function '{desc}'")
