import math


def nCr(n, r):
    f = math.factorial
    return f(n) / f(r) / f(n - r)


def uniform(samples):
    def distribution(x):
        if x in samples:
            return 1 / len(samples)
        return 0

    return (samples, distribution)


def bernoulli(q):
    def distribution(x):
        if x == 0:
            return 1 - q
        elif x == 1:
            return q
        return 0

    return ({0, 1}, distribution)


def binomial(n, q):
    def distribution(x):
        if x > n or x < 0:
            return 0

        comb = nCr(n, x)
        successes = q**x
        failures = (1 - q)**(n - x)

        return comb * successes * failures

    return (set(range(n + 1)), distribution)
