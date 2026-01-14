from math import sqrt
import numpy as np

def gaussian_sketch(A: np.ndarray, B: np.ndarray, d: int, seed: int | None = None) -> np.ndarray:
    """
    Gaussian Sketch approximate of A^T @ B

    A: (n, m)
    B: (n, p)
    d: Sketch dimension (d << n)
    seed: Random seed for reproducibility
    """
    n = A.shape[0]
    rng = np.random.default_rng(seed)

    # S ~ N(0, 1/d), Î = S^T @ S
    # E[Î_ij] = 1 if i=j, 0 otherwise
    # var(Î_ij) = 2/d if i=j, 1/d otherwise
    S = rng.standard_normal((d, n)) / sqrt(d)

    SA = S @ A  # (d, m)
    SB = S @ B  # (d, p)

    return SA.T @ SB

def count_sketch():
    return None
