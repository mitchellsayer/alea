import pytest

from alea.discrete import BernoulliRandVar, UniformDiscreteRandVar, UnaryDiscreteRandVar


def almost_equal(x, y, epsilon=1e-5):
    return abs(x - y) <= epsilon


class TestUnaryMultiplication:

    def test_sample_mean(self):
        X = BernoulliRandVar(0.6)
        g = lambda x : x * 50
        gX = UnaryDiscreteRandVar(X, g)
        gX2 = X * 50
        assert(almost_equal(gX.sample_mean(), gX2.sample_mean(), 0.5))


    def test_mean(self):
        X = BernoulliRandVar(0.6)
        g = lambda x : x * 50
        gX = UnaryDiscreteRandVar(X, g)
        gX2 = X * 50
        assert(almost_equal(gX.mean(), gX2.mean()))


    def test_variance(self):
        X = BernoulliRandVar(0.6)
        g = lambda x : x * 50
        gX = UnaryDiscreteRandVar(X, g)
        gX2 = X * 50
        assert(almost_equal(gX.variance(), gX2.variance()))


class TestUnaryAddition:

    def test_sample_mean(self):
        X = BernoulliRandVar(0.6)
        g = lambda x : x + 50
        gX = UnaryDiscreteRandVar(X, g)
        gX2 = X + 50
        assert(almost_equal(gX.sample_mean(), gX2.sample_mean(), 0.5))


    def test_mean(self):
        X = BernoulliRandVar(0.6)
        g = lambda x : x + 50
        gX = UnaryDiscreteRandVar(X, g)
        gX2 = X + 50
        assert(almost_equal(gX.mean(), gX2.mean()))


    def test_variance(self):
        X = BernoulliRandVar(0.6)
        g = lambda x : x + 50
        gX = UnaryDiscreteRandVar(X, g)
        gX2 = X + 50
        assert(almost_equal(gX.variance(), gX2.variance()))


class TestUnaryTrivial:

    def test_sample_mean(self):
        X = UniformDiscreteRandVar(set(range(1000)))
        g = lambda x : 1
        gX = UnaryDiscreteRandVar(X, g)
        assert(almost_equal(gX.sample_mean(), 1))


    def test_mean(self):
        X = UniformDiscreteRandVar(set(range(1000)))
        g = lambda x : 1
        gX = UnaryDiscreteRandVar(X, g)
        assert(almost_equal(gX.mean(), 1))


    def test_variance(self):
        X = UniformDiscreteRandVar(set(range(1000)))
        g = lambda x : 1
        gX = UnaryDiscreteRandVar(X, g)
        assert(almost_equal(gX.variance(), 0))
