import pytest

from alea.discrete import DiscreteRandVar

class TestDiscrete:

    def test_mean(self):
        def coin_mf(x):
            if x == -1:
                return 0.5
            else:
                return 0.5

        coin_ss = {-1, 1}

        X = DiscreteRandVar(coin_ss, coin_mf)

        assert (X.mean() == 0)

    def test_variance(self):
        def coin_mf(x):
            if x == -1:
                return 0.5
            else:
                return 0.5

        coin_ss = {-1, 1}

        X = DiscreteRandVar(coin_ss, coin_mf)

        assert (X.variance() == 1)
