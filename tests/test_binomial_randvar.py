import pytest

from alea.discrete import BinomialRandVar


def almost_equal(x, y):
    return abs(x - y) <= 1e-5

class TestBinomial:
    def test_mean(self):
        X = BinomialRandVar(100000, 0.6)
        assert(almost_equal(X.mean(), 60000))

    def test_variable(self):
        X = BinomialRandVar(100000, 0.6)
        assert(almost_equal(X.variance(), 24000))
