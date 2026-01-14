import numpy as np
import numpy.random
import pytest
from sketchlinalg.sketchlinalg.sketches.gaussian_count import *


# Expected Frobenius norm error and error variance should decrease as d increases
def test_variance_accuracy():
    n, m, p = 300, 30, 20
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((n, p))
    actual = A.T @ B

    seeds = range(100)
    def errs(d):
        return np.array([
            np.linalg.norm(actual - gaussian_sketch(A, B, d, seed), ord='fro')
            for seed in seeds
        ])

    err_small, err_large = errs(20), errs(120)

    assert np.var(err_large) < np.var(err_small)
    assert np.median(err_large) < np.median(err_small)


# Ensures (SA).T @ S(B1 + B2) = (SA).T @ SB1 + (SA).T @ SB2
def test_fixed_S_linearity_in_B():
    n, m, p, d = 80, 5, 6, 30
    rng = np.random.default_rng(0)
    S = gaussian_sketch_matrix(n, d)
    A = rng.standard_normal((n, m))
    B1 = rng.standard_normal((n, p))
    B2 = rng.standard_normal((n, p))

    left = (S @ A).T @ S @ (B1 + B2)
    right = (S @ A).T @ (S @ B1) + (S @ A).T @ (S @ B2)
    assert np.allclose(left, right)


def test_does_not_modify_inputs():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((50, 4))
    B = rng.standard_normal((50, 3))
    A0, B0 = A.copy(), B.copy()
    _ = gaussian_sketch(A, B, d=10, seed=0)
    np.testing.assert_array_equal(A, A0)
    np.testing.assert_array_equal(B, B0)


def test_no_nan_inf_large_values():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((200, 5)) * 1e8
    B = rng.standard_normal((200, 6)) * 1e8
    C = gaussian_sketch(A, B, d=50, seed=0)
    assert np.isfinite(C).all()

