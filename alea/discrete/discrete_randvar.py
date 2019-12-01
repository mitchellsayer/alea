from ..randvar import RandVar

import copy
import numpy as np
import collections


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
        if isinstance(self, DiscreteRandVar):
            # E[XY] can be calculated using the transformation theorem
            # provided that X and Y are both discrete random variables
            product_mean = 0
            for x in self.sample_space:
                for y in rv.sample_space:
                    p = self._get_probability(x) * rv._get_probability(y)
                    product_mean += x * y * p
        else:
            # Handle the general case by simply multiplying the two
            # random variables together and taking the mean of their product
            product_mean = (self * rv).mean()
        return product_mean - self.mean() * rv.mean()


    def __add__(self, obj):
        if isinstance(obj, int) or isinstance(obj, float):
            return ConstantPlusDiscreteRandVar(self, obj)
        elif isinstance(obj, DiscreteRandVar):
            return DiscretePlusDiscreteRandVar(self, obj)
        else:
            raise ValueError("Right operand must be a constant or random variable")


    def __mul__(self, obj):
        if isinstance(obj, int) or isinstance(obj, float):
            return ConstantTimesDiscreteRandVar(self, obj)
        # TODO: Add support for discrete random variables 
        else:
            raise ValueError("Right operand must be a constant or random variable")



class ConstantPlusDiscreteRandVar(DiscreteRandVar):

    def __init__(self, rv, c):

        def pmf(x):
            return rv.mass_function(x - c)

        DiscreteRandVar.__init__(self, {x + c for x in rv.sample_space}, pmf)
        self.rv = rv
        self.c = c

        rv.children.add(self)
        self.parents.add(rv)


    def _new_sample(self):
        assert(len(self.parents) == 1)
        return self.rv.sample() + self.c


    def _new_mean(self):
        return self.rv.mean() + self.c


    def _new_variance(self):
        return self.rv.variance()


class DiscretePlusDiscreteRandVar(DiscreteRandVar):

    def __init__(self, rv1, rv2):
        mapping = collections.defaultdict(set)
        for x in rv1.sample_space:
            for y in rv2.sample_space:
                mapping[x + y].add((x, y))

        def pmf(x):
            combs = mapping[x]
            probs = [rv1._get_probability(y) * rv2._get_probability(z) for (y, z) in combs]
            return sum(probs)

        DiscreteRandVar.__init__(self, set(mapping.keys()), pmf)
        self.rv1 = rv1
        self.rv2 = rv2

        rv1.children.add(self)
        rv2.children.add(self)
        self.parents.add(rv1)
        self.parents.add(rv2)


    def _new_sample(self):
        assert(len(self.parents) == 2)
        return self.rv1.sample() + self.rv2.sample()


    def _new_mean(self):
        return self.rv1.mean() + self.rv2.mean()


    def _new_variance(self):
        return self.rv1.variance() + self.rv2.variance() + 2 * self.rv1.covariance(self.rv2)



class ConstantTimesDiscreteRandVar(DiscreteRandVar):

    def __init__(self, rv, c):

        def pmf(x):
            return rv.mass_function(x / c)

        DiscreteRandVar.__init__(self, {x * c for x in rv.sample_space}, pmf)
        self.rv = rv
        self.c = c

        rv.children.add(self)
        self.parents.add(rv)


    def _new_sample(self):
        assert(len(self.parents) == 1)
        return self.rv.sample() * self.c


    def _new_mean(self):
        return self.rv.mean() * self.c


    def _new_variance(self):
        return self.rv.variance() * self.c * self.c
