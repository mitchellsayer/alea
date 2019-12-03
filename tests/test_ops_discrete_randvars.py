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


class TestDiscreteSum:

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


class TestExponentiation:

    def test_bernoulli_moments(self):
        # The moments of a Bernoulli random variable are identical 
        X = BinomialRandVar(1, 0.6)
        for i in range(2, 20):
            Y = X ** i
            assert(almost_equal(Y.mean(), 0.6))


    def test_binomial_moments(self):
        # Use exponentiation to calculate the second & third moments of X
        N = 5
        p = 0.6
        X = BinomialRandVar(N, p)
        Y = X ** 2
        Z = X ** 3
        Np = N * p
        assert(almost_equal(Y.mean(), Np * (1 - p + Np)))
        assert(almost_equal(Z.mean(), Np * (1 - 3 * p + 3 * Np + 2 * p * p - 3 * Np * p + Np * Np)))


    def test_binomial_central_moments(self):
        # Use exponential and subtraction to calculate the 
        # second, third, fourth central moments of X
        N = 5
        p = 0.6
        X = BinomialRandVar(N, p)
        Xc = X - X.mean()
        Yc = Xc ** 2
        Zc = Xc ** 3
        Ac = Xc ** 4
        q = 1 - p
        Np = N * p
        assert(almost_equal(Yc.mean(), Np * q))
        assert(almost_equal(Zc.mean(), Np * q * (1 - 2 * p)))
        assert(almost_equal(Ac.mean(), Np * q * (3 * p * p * (2 - N) + 3 * p * (N - 2) + 1)))


class TestDiscreteMultiply:

    def test_distributive_property(self):
        X = BinomialRandVar(1, 0.5)
        Y = BinomialRandVar(1, 0.5)
        Z = X * (X * (X + Y))
        Z2 = (X ** 3) + (X ** 2) * Y
        assert(almost_equal(Z.mean(), Z2.mean()))
        # assert(almost_equal(Z.variance(), Z2.variance()))


    def test_foil(self):
        X = BinomialRandVar(1, 0.5)
        Y = BinomialRandVar(1, 0.5)
        Z = (X + Y) * (X + Y)
        Z2 = (X * X) + (X * Y) + (X * Y) + (Y * Y)
        assert(almost_equal(Z.mean(), Z2.mean()))
        # assert(almost_equal(Z.variance(), Z2.variance()))


    def test_slow_exponentiation(self):
        X = BinomialRandVar(1, 0.5)
        Y = BinomialRandVar(1, 0.5)
        Z = X * X * X * X * X * X
        Z2 = X ** 6
        assert(almost_equal(Z.mean(), Z2.mean()))
        # assert(almost_equal(Z.variance(), Z2.variance()))
