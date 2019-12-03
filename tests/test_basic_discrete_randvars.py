import pytest

from alea.discrete import RootDiscreteRandVar


class TestDiscrete:

    def test_mean(self):
        support = {-1, 1}
        pmf = lambda x : 0.5 if x in support else 0

        X = RootDiscreteRandVar(support, pmf)

        assert (X.mean() == 0)


    def test_variance(self):
        support = {-1, 1}
        pmf = lambda x : 0.5 if x in support else 0

        X = RootDiscreteRandVar(support, pmf)

        assert (X.variance() == 1)


    def test_covariance(self):
        support = {-1, 1}
        pmf = lambda x : 0.5 if x in support else 0

        X = RootDiscreteRandVar(support, pmf)
        Y = RootDiscreteRandVar(support, pmf)

        # X, Y are independent, so their covariance must be zero
        assert (X.covariance(Y) == 0)
