import copy

import numpy as np

from ..randvar import RandVar

class BaseDiscreteRandVar(RandVar):

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


    def _new_sample(self):
        # By default, we assume that we are choosing from the random
        # variable's probability distribution. This, in turn, assumes 
        # that this random variable does not have any parents and thus
        # represents an independent, real-world event
        assert(len(self.parents) == 0)
        elements = list(self.sample_space)
        probabilities = [self._get_probability(x) for x in elements]
        return np.random.choice(elements, 1, p=probabilities)[0]


    def _new_mean(self):
        mean = 0
        for x in self.sample_space:
            p = self._get_probability(x)
            mean += p * x
        return mean


    def _new_variance(self):
        # Variance is calculated by doing E[X^2] - E[X]^2
        # E[X^2] can be calculated using the transformation theorem.
        variance = 0
        for x in self.sample_space:
            p = self._get_probability(x)
            variance += x**2 * p
        return variance - self.mean()**2


    def _new_covariance(self, rv):
        # Covariance is equal to E[XY] - E[X]E[Y] 
        if isinstance(self, BaseDiscreteRandVar):
            # For discrete random variables, 
            # E[XY] can be calculated using the transformation theorem
            xy_mean = 0
            for x in self.sample_space:
                for y in rv.sample_space:
                    p = self._get_probability(x) * rv._get_probability(y)
                    xy_mean += x * y * p
        else:
            # For hybrid or continuous covariances, we should refer
            # potentially to the joint probability distribution of the
            # new random variable, XY
            xy_mean = (self * rv).mean()
        return xy_mean - self.mean() * rv.mean()

