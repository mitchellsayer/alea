from .randvar import DiscreteRandVar

import random
import math
import numpy as np
import copy


class RootDiscreteRandVar(DiscreteRandVar):

    def __init__(self, sample_space, mass_function):
        DiscreteRandVar.__init__(self)
        self.sample_space = copy.copy(sample_space)
        self.mass_function = mass_function
        self.pcache = {}
        self.sample_list = None


    def probability_of(self, x):
        if x in self.pcache:
            return self.pcache[x]
        else:
            p = self.mass_function(x)
            self.pcache[x] = p
            return p


    def _new_roots(self):
        return {self}


    def _new_sample(self):
        # By default, we assume that we are choosing from the random
        # variable's probability distribution. This, in turn, assumes 
        # that this random variable does not have any parents and thus
        # represents an independent, real-world event
        if self.sample_list is None:
            self.sample_list = list(self.sample_space)
        probabilities = [self.probability_of(x) for x in self.sample_list]
        return np.random.choice(self.sample_list, 1, p=probabilities)[0]


    def _new_mean(self, fixed_means):
        mean = 0
        for x in self.sample_space:
            p = self.probability_of(x)
            mean += p * x
        return mean


    def _new_variance(self):
        variance = 0
        for x in self.sample_space:
            p = self.probability_of(x)
            variance += x ** 2 * p
        return variance - self.mean() ** 2
