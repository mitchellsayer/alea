import pytest

from alea import RandVec
from alea.discrete import DiscreteRandVec, RootDiscreteRandVar

import numpy as np


def almost_equal(x, y, epsilon=1e-5):
    return abs(x - y) <= epsilon


class TestIndependentVector:

    def test_mean(self):
        pmf = lambda _ : 1.0/3
        X1 = RootDiscreteRandVar({1, 4, 7}, pmf)
        X2 = RootDiscreteRandVar({2, 5, 8}, pmf)
        X3 = RootDiscreteRandVar({3, 6, 9}, pmf)
        X = RandVec([X1, X2, X3])

        assert(len(X) == 3)
        assert(almost_equal(X.mean()[0], 4))
        assert(almost_equal(X.mean()[1], 5))
        assert(almost_equal(X.mean()[2], 6))


    def test_variance(self):
        pmf = lambda _ : 1.0/3
        X1 = RootDiscreteRandVar({1, 4, 7}, pmf)
        X2 = RootDiscreteRandVar({2, 5, 8}, pmf)
        X3 = RootDiscreteRandVar({3, 6, 9}, pmf)
        X = RandVec([X1, X2, X3])

        np.testing.assert_allclose(X.variance(), np.identity(3) * 6)


class TestDependentVector:

    def test_mean(self):
        support = {(1, 2, 3), (4, 5, 6), (7, 8, 9)}
        pmf = lambda _ : 1.0/3
        X = DiscreteRandVec(support, pmf)

        assert(len(X) == 3)
        assert(almost_equal(X.mean()[0], 4))
        assert(almost_equal(X.mean()[1], 5))
        assert(almost_equal(X.mean()[2], 6))


    def test_variance(self):
        support = {(1, 2, 3), (4, 5, 6), (7, 8, 9)}
        pmf = lambda _ : 1.0/3
        X = DiscreteRandVec(support, pmf)

        np.testing.assert_allclose(X.variance(), np.ones((3, 3)) * 6)
