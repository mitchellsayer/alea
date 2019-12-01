import pytest

from alea.discrete import BinomialRandVar


def almost_equal(x, y):
    return abs(x - y) <= 1e-5


class TestConstantSum:

    def test_mean(self):
        X = BinomialRandVar(100000, 0.6)
        Y = X + 100
        assert(almost_equal(Y.mean(), 60100))


    def test_variance(self):
        X = BinomialRandVar(100000, 0.6)
        Y = X + 100
        assert(almost_equal(Y.variance(), 24000))


class TestConstantSum:

    def test_mean(self):
        X = BinomialRandVar(2, 0.3)
        Y = BinomialRandVar(3, 0.3)
        Z = X + Y
        A = BinomialRandVar(5, 0.3)
        assert(almost_equal(Z.mean(), A.mean()))


    def test_variance(self):
        X = BinomialRandVar(2, 0.3)
        Y = BinomialRandVar(3, 0.3)
        Z = X + Y
        A = BinomialRandVar(5, 0.3)
        assert(almost_equal(Z.variance(), A.variance()))


class TestConstantMultiply:

    def test_mean(self):
        X = BinomialRandVar(100, 0.6)
        Y = X * 100
        assert(almost_equal(Y.mean(), 6000))


    def test_variance(self):
        X = BinomialRandVar(100, 0.6)
        Y = X * 100
        assert(almost_equal(Y.variance(), 240000))
