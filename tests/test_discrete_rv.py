import pytest

from alea import DiscreteRandomVariable, binomial


def almost_equal(x, y):
    return abs(x - y) <= 1e-5


class TestDiscrete:
    def test_mean(self):
        def coin_mf(x):
            if x == -1:
                return 0.5
            elif x == 1:
                return 0.5
            else:
                return 0

        coin_ss = {-1, 1}

        X = DiscreteRandomVariable(coin_ss, coin_mf)

        assert (X.mean == 0)

    def test_variance(self):
        def coin_mf(x):
            if x == -1:
                return 0.5
            elif x == 1:
                return 0.5
            else:
                return 0

        coin_ss = {-1, 1}

        X = DiscreteRandomVariable(coin_ss, coin_mf)

        assert (X.variance == 1)

    def test_sample(self):
        pass


class TestBinomial:
    def test_mean(self):
        X = DiscreteRandomVariable(*binomial(11, 0.6))
        assert (almost_equal(X.mean, 11 * 0.6))

    def test_variable(self):
        X = DiscreteRandomVariable(*binomial(6, 0.6))
        assert (almost_equal(X.variance, 6 * 0.6 * 0.4))
