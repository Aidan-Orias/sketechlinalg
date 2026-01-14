import numpy as np
import pytest
from sketchlinalg.sketchlinalg.sketches.gaussian_count import gaussian_sketch

class TestGaussianSketch:
    # Expected Frobenius norm error should decrease as d increases
    def test_accuracy(self):
        n, m, p = 300, 30, 20
        A = np.random.random((n, m))
        B = np.random.random((n, p))
        actual = A.T @ B

        ds = [(20, 120)]
        seeds = range(20)

        for d_small, d_large in ds:
            err_smalls = []
            err_larges = []
            for seed in seeds:
                err_smalls.append(np.linalg.norm(actual - gaussian_sketch(A, B, d_small, seed), ord='fro'))
                err_larges.append(np.linalg.norm(actual - gaussian_sketch(A, B, d_large, seed), ord='fro'))

        assert np.median(err_larges) < np.median(err_smalls)
