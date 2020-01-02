from .randvar import DiscreteRandVar

from functools import lru_cache

import random
import math
import numpy as np
import copy


class RootDiscreteRandVar(DiscreteRandVar):

    def __init__(self, sample_space, mass_function):
        '''
        Initializes a root discrete random variable using
        a support and probability mass function. The
        support is a set of numbers representing
        the values that the random variable can take
        with non-zero probability. The mass function
        maps a number in the support to a probability.

        Together, the support and mass function must
        form a valid probability distribution. Notably,
        the sum of all probabilities must be equal to 1.

        Args:
            sample_space: The set of values of the
            random variable (the support)
            mass_function: A function mapping a value in
            the support to a non-zero probability
        '''

        DiscreteRandVar.__init__(self)
        self.sample_space = copy.copy(sample_space)
        self.mass_function = mass_function
        self.sample_list = None


    def _new_roots(self):
        return {self}


    def _new_sample(self):
        # By default, we assume that we are choosing from the random
        # variable's probability distribution. This, in turn, assumes 
        # that this random variable does not have any parents and thus
        # represents an independent, real-world event
        if self.sample_list is None:
            self.sample_list = list(self.sample_space)
        probabilities = [self.mass_function(x) for x in self.sample_list]
        return np.random.choice(self.sample_list, 1, p=probabilities)[0]


    def _new_mean(self, fixed_means):
        mean = 0
        for x in self.sample_space:
            p = self.mass_function(x)
            mean += p * x
        return mean


    def _new_variance(self):
        variance = 0
        for x in self.sample_space:
            p = self.mass_function(x)
            variance += x ** 2 * p
        return variance - self.mean() ** 2
