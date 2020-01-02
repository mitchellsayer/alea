import pytest

from alea.discrete import RootDiscreteRandVar, BinomialRandVar, BernoulliRandVar, UniformDiscreteRandVar, UnaryDiscreteRandVar


def almost_equal(x, y, epsilon=1e-5):
    return abs(x - y) <= epsilon


class TestBasic:

    def test_sample_mean(self):
        support = {-1, 1}
        pmf = lambda x : 0.5 if x in support else 0

        X = RootDiscreteRandVar(support, pmf)

        assert(almost_equal(X.sample_mean(), 0, 1e-1))
        assert(almost_equal(X.sample_variance(), 1, 1e-1))


    def test_mean(self):
        support = {-1, 1}
        pmf = lambda x : 0.5 if x in support else 0

        X = RootDiscreteRandVar(support, pmf)

        assert(almost_equal(X.mean(), 0))


    def test_variance(self):
        support = {-1, 1}
        pmf = lambda x : 0.5 if x in support else 0

        X = RootDiscreteRandVar(support, pmf)

        assert(almost_equal(X.variance(), 1))


    def test_covariance(self):
        support = {-1, 1}
        pmf = lambda x : 0.5 if x in support else 0

        X = RootDiscreteRandVar(support, pmf)
        Y = RootDiscreteRandVar(support, pmf)

        # X, Y are independent, so their covariance must be zero
        assert(almost_equal(X.covariance(Y), 0))


class TestBernoulli:

    def test_sample_mean(self):
        X = BernoulliRandVar(0.6)
        assert(almost_equal(X.sample_mean(), 0.6, 0.5))


    def test_mean(self):
        X = BernoulliRandVar(0.6)
        assert(almost_equal(X.mean(), 0.6))


    def test_variance(self):
        X = BernoulliRandVar(0.6)
        assert(almost_equal(X.variance(), 0.24))


class TestBinomial:

    def test_sample_mean(self):
        X = BinomialRandVar(10, 0.6)
        assert(almost_equal(X.sample_mean(), 6, 0.5))


    def test_mean(self):
        X = BinomialRandVar(100000, 0.6)
        assert(almost_equal(X.mean(), 60000))


    def test_variance(self):
        X = BinomialRandVar(100000, 0.6)
        assert(almost_equal(X.variance(), 24000))


class TestUniform:

    def test_sample_mean(self):
        X = UniformDiscreteRandVar({-2, 553, 43})
        assert(almost_equal(X.sample_mean(), 198, 10))
        Y = UniformDiscreteRandVar({1, 2, 3, 4, 5, 6})
        assert(almost_equal(Y.sample_mean(), 3.5, 0.5))


    def test_mean(self):
        X = UniformDiscreteRandVar({-2, 553, 43})
        assert(almost_equal(X.mean(), 198))
        Y = UniformDiscreteRandVar({1, 2, 3, 4, 5, 6})
        assert(almost_equal(Y.mean(), 3.5))


    def test_variance(self):
        X = UniformDiscreteRandVar({-2, 553, 43})
        assert(almost_equal(X.variance(), 63350))
        Y = UniformDiscreteRandVar({1, 2, 3, 4, 5, 6})
        assert(almost_equal(Y.variance(), 2.916667))


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
        assert(almost_equal(Z.variance(), Z2.variance()))


    def test_foil(self):
        X = BinomialRandVar(1, 0.5)
        Y = BinomialRandVar(1, 0.5)
        Z = (X + Y) * (X + Y)
        Z2 = (X * X) + (X * Y) + (X * Y) + (Y * Y)
        assert(almost_equal(Z.mean(), Z2.mean()))
        assert(almost_equal(Z.variance(), Z2.variance()))


    def test_slow_exponentiation(self):
        X = BinomialRandVar(1, 0.5)
        Y = BinomialRandVar(1, 0.5)
        Z = X * X * X * X * X * X
        Z2 = X ** 6
        assert(almost_equal(Z.mean(), Z2.mean()))
        assert(almost_equal(Z.variance(), Z2.variance()))


    def test_large_root_distributions(self):
        X = BinomialRandVar(30, 0.1)
        Y = BinomialRandVar(30, 0.2)
        Z = BinomialRandVar(30, 0.3)
        A = X * Y * Z * X * Y * Z
        A2 = (X ** 2) * (Y ** 2) * (Z ** 2)
        assert(almost_equal(A.mean(), A2.mean()))
        assert(almost_equal(A.variance(), A2.variance()))


class TestUnaryMultiplication:

    def test_sample_mean(self):
        X = BernoulliRandVar(0.6)
        g = lambda x : x * 50
        gX = UnaryDiscreteRandVar(X, g)
        gX2 = X * 50
        assert(almost_equal(gX.sample_mean(), gX2.sample_mean(), 1))


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
        assert(almost_equal(gX.sample_mean(), gX2.sample_mean(), 1))


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
