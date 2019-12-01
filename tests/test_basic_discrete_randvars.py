import pytest

from alea.discrete import DiscreteRandVar


class TestDiscrete:

    def test_mean(self):

        def pmf(x):
            if x == -1:
                return 0.5
            else:
                return 0.5

        support = {-1, 1}

        X = DiscreteRandVar(support, pmf)

        assert (X.mean() == 0)


    def test_variance(self):

        def pmf(x):
            if x == -1:
                return 0.5
            else:
                return 0.5

        support = {-1, 1}

        X = DiscreteRandVar(support, pmf)

        assert (X.variance() == 1)


    def test_covariance(self):

        def pmf(x):
            if x == -1:
                return 0.5
            else:
                return 0.5

        support = {-1, 1}

        X = DiscreteRandVar(support, pmf)
        Y = DiscreteRandVar(support, pmf)

        # X, Y are independent, so their
        # covariance must be zero
        assert (X.covariance(Y) == 0)
