import pytest

from alea.discrete import RootDiscreteRandVar


def almost_equal(x, y, epsilon=1e-5):
    return abs(x - y) <= epsilon


class TestDiscrete:

    def test_sample_mean(self):
        support = {-1, 1}
        pmf = lambda x : 0.5 if x in support else 0

        X = RootDiscreteRandVar(support, pmf)

        assert(almost_equal(X.sample_mean(), 0, 1e-1))


    def test_mean(self):
        support = {-1, 1}
        pmf = lambda x : 0.5 if x in support else 0

        X = RootDiscreteRandVar(support, pmf)

        assert(almost_equal(X.mean(), 0))


    def test_variance(self):
        support = {-1, 1}
        pmf = lambda x : 0.5 if x in support else 0

        X = RootDiscreteRandVar(support, pmf)

        assert(almost_equal(X.variance(), 1))


    def test_covariance(self):
        support = {-1, 1}
        pmf = lambda x : 0.5 if x in support else 0

        X = RootDiscreteRandVar(support, pmf)
        Y = RootDiscreteRandVar(support, pmf)

        # X, Y are independent, so their covariance must be zero
        assert(almost_equal(X.covariance(Y), 0))
