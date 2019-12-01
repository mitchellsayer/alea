from .randvar_base import BaseDiscreteRandVar

import collections

class DiscreteRandVar(BaseDiscreteRandVar):

    def __init__(self, sample_space, mass_function):
        BaseDiscreteRandVar.__init__(self, sample_space, mass_function)


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
