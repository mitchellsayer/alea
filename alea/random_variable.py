import copy
from functools import partial

# DiscreteRandomVariable class

EPSILON = 1e-4


class DiscreteRandomVariable:
    def __init__(self, sample_space, mass_function, *args):
        self.sample_space = copy.copy(sample_space)

        self.mass_function = partial(mass_function, *args)

        self._validate()

        self.pcache = self._get_probability_cache()
        self.mean = self._get_mean()
        self.variance = self._get_variance()

    def _validate(self):
        total = 0

        for x in self.sample_space:
            total += self.mass_function(x)

        print(total)
        assert (abs(total - 1.0) <= EPSILON)

    def _get_probability_cache(self):
        cache = {}
        for x in self.sample_space:
            cache[x] = self.mass_function(x)
        return cache

    def _get_mean(self):
        mean = 0

        for x in self.sample_space:
            p = self.pcache[x]
            mean = mean + (p * x)

        return mean

    def _get_variance(self):
        variance = 0

        for x in self.sample_space:
            p = x**2 * self.pcache[x]
            variance = variance + p

        return variance - self.mean**2