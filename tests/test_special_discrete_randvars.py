import pytest

from alea.discrete import BernoulliRandVar, BinomialRandVar, UniformRandVar


def almost_equal(x, y):
    return abs(x - y) <= 1e-5


class TestBernoulli:

    def test_mean(self):
        X = BernoulliRandVar(0.6)
        assert(almost_equal(X.mean(), 0.6))


    def test_variance(self):
        X = BernoulliRandVar(0.6)
        assert(almost_equal(X.variance(), 0.24))


class TestBinomial:

    def test_mean(self):
        X = BinomialRandVar(100000, 0.6)
        assert(almost_equal(X.mean(), 60000))


    def test_variance(self):
        X = BinomialRandVar(100000, 0.6)
        assert(almost_equal(X.variance(), 24000))


class TestUniform:

    def test_mean(self):
        X = UniformRandVar({-2, 553, 43})
        assert(almost_equal(X.mean(), 198))
        Y = UniformRandVar({1, 2, 3, 4, 5, 6})
        assert(almost_equal(Y.mean(), 3.5))


    def test_variance(self):
        X = UniformRandVar({-2, 553, 43})
        assert(almost_equal(X.variance(), 63350))
        Y = UniformRandVar({1, 2, 3, 4, 5, 6})
        assert(almost_equal(Y.variance(), 2.916667))
