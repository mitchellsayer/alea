import copy

import numpy as np

from ..randvar import RandVar

class DiscreteRandVar(RandVar):

    def __init__(self, sample_space, mass_function):
        RandVar.__init__(self)
        self.sample_space = copy.copy(sample_space)
        self.mass_function = mass_function
        self.pcache = {}


    def _get_probability(self, x):
        if x in self.pcache:
            return self.pcache[x]
        else:
            p = self.mass_function(x)
            self.pcache[x] = p
            return p


    def _get_sample(self):
        elements = list(self.sample_space)
        probabilities = [self._get_probability(x) for x in elements]
        return np.random.choice(elements, 1, p=probabilities)[0]


    def _get_mean(self):
        mean = 0
        for x in self.sample_space:
            p = self._get_probability(x)
            mean += p * x
        return mean


    def _get_variance(self):
        variance = 0
        for x in self.sample_space:
            p = self._get_probability(x)
            variance += x**2 * p
        return variance - self.mean()**2


    def __add__(self, obj):
        # TODO: Generic addition for constants, other discrete random variables
        pass


    def __sub__(self, obj):
        # TODO: Generic subtraction for constants, other discrete random variables
        pass


    def __mul__(self, obj):
        # TODO: Generic multiplication for constants, other discrete random variables
        pass
