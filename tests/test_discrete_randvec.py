import pytest

from alea.discrete import DiscreteRandVec


def almost_equal(x, y, epsilon=1e-5):
    return abs(x - y) <= epsilon


class TestIndependentVector:

    def test_mean(self):
        support = {(1, 2, 3), (4, 5, 6), (7, 8, 9)}
        pmf = lambda x : 1.0/3 if x in support else 0

        X = DiscreteRandVec(support, pmf)

        assert(len(X) == 3)
        print(X.mean())
        assert(almost_equal(X.mean()[0], 4))
        assert(almost_equal(X.mean()[1], 5))
        assert(almost_equal(X.mean()[2], 6))
