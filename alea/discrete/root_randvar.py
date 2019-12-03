from .discrete_randvar import DiscreteRandVar

import random
import math
import numpy as np


class RootDiscreteRandVar(DiscreteRandVar):

    def __init__(self, sample_space, mass_function):
        DiscreteRandVar.__init__(self, sample_space, mass_function)
        self.elements = None


    def _new_sample(self):
        # By default, we assume that we are choosing from the random
        # variable's probability distribution. This, in turn, assumes 
        # that this random variable does not have any parents and thus
        # represents an independent, real-world event
        if self.elements is None:
            self.elements = list(self.sample_space)
        probabilities = [self._get_probability(x) for x in self.elements]
        return np.random.choice(self.elements, 1, p=probabilities)[0]


    def _new_mean(self, fixed_means):
        mean = 0
        for x in self.sample_space:
            p = self._get_probability(x)
            mean += p * x
        return mean


    def _new_variance(self):
        variance = 0
        for x in self.sample_space:
            p = self._get_probability(x)
            variance += x ** 2 * p
        return variance - self.mean() ** 2
