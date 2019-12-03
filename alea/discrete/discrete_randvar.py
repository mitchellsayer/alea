from ..randvar import RandVar
from collections import deque

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


    def _new_variance(self):
        # Variance is equal to E[X^2] - E[X]E[X]
        return (self ** 2).mean() - self.mean() ** 2


    def _new_covariance(self, rv):
        # Covariance is equal to E[XY] - E[X]E[Y] 
        return (self * rv).mean() - self.mean() * rv.mean()


    def __add__(self, obj):
        if isinstance(obj, int) or isinstance(obj, float):
            return ConstantPlusDiscreteRandVar(self, obj)
        elif isinstance(obj, DiscreteRandVar):
            return DiscretePlusDiscreteRandVar(self, obj)
        else:
            raise ValueError("Right operand must be constant or randvar")


    def __mul__(self, obj):
        if isinstance(obj, int) or isinstance(obj, float):
            return ConstantTimesDiscreteRandVar(self, obj)
        elif isinstance(obj, DiscreteRandVar):
            return DiscreteTimesDiscreteRandVar(self, obj)
        else:
            raise ValueError("Right operand must be constant or randvar")


    def __pow__(self, num):
        if not isinstance(num, int):
            raise ValueError("Right operand must be an integer")
        if num < 1:
            raise ValueError("Exponent must be greater than zero")
        return ExponentDiscreteRandVar(self, num)


class ConstantPlusDiscreteRandVar(DiscreteRandVar):

    def __init__(self, rv, c):

        def pmf(x):
            return rv._get_probability(x - c)

        DiscreteRandVar.__init__(self, {x + c for x in rv.sample_space}, pmf)
        self.rv = rv
        self.c = c

        rv.children.add(self)
        self.parents.add(rv)


    def _new_sample(self):
        assert(len(self.parents) == 1)
        return self.rv.sample() + self.c


    def _new_mean(self, fixed_means):
        return self.rv.mean(fixed_means) + self.c


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


    def _new_mean(self, fixed_means):
        return self.rv1.mean(fixed_means) + self.rv2.mean(fixed_means)


    def _new_variance(self):
        return self.rv1.variance() + self.rv2.variance() + 2 * self.rv1.covariance(self.rv2)


class ConstantTimesDiscreteRandVar(DiscreteRandVar):

    def __init__(self, rv, c):

        def pmf(x):
            return rv._get_probability(x / c)

        DiscreteRandVar.__init__(self, {x * c for x in rv.sample_space}, pmf)
        self.rv = rv
        self.c = c

        rv.children.add(self)
        self.parents.add(rv)


    def _new_sample(self):
        assert(len(self.parents) == 1)
        return self.rv.sample() * self.c


    def _new_mean(self, fixed_means):
        return self.rv.mean(fixed_means) * self.c


    def _new_variance(self):
        return self.rv.variance() * self.c * self.c


class DiscreteTimesDiscreteRandVar(DiscreteRandVar):

    def __init__(self, rv1, rv2):
        mapping = collections.defaultdict(set)
        for x in rv1.sample_space:
            for y in rv2.sample_space:
                mapping[x * y].add((x, y))

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
        return self.rv1.sample() * self.rv2.sample()


    def _new_mean(self, fixed_means):

        def find_roots(rv, accum):
            if len(rv.parents) == 0:
                accum.add(rv)
            else:
                for parent in rv.parents:
                    find_roots(parent, accum)

        roots1 = set()
        roots2 = set()
        find_roots(self.rv1, roots1)
        find_roots(self.rv2, roots2)
        shared_roots = list(roots1.intersection(roots2))

        # If X and Y do not share any roots, then they are independent
        # This implies that E[XY] = E[X]E[Y], which is a quick calculation
        if len(shared_roots) == 0:
            return self.rv1.mean(fixed_means) * self.rv2.mean(fixed_means)

        # X and Y are dependent random variables! This would be impossible to calculate.
        # However, we know that because X and Y are dependent, they must share at least
        # one root discrete variable acting as a probabilistic generation.

        # If we generate every possible combination that the shared roots can take, we 
        # can make X and Y 'independent' again
        combinations = deque([[]])
        for srv in shared_roots:
            if srv in fixed_means:
                srv_space = [(fixed_means[srv], 1)]
            else:
                srv_space = [(x, srv._get_probability(x)) for x in srv.sample_space]
            level = len(combinations)
            for _ in range(level):
                xs = combinations.popleft()
                for x in srv_space:
                    combinations.append(xs + [x])
        # Then, we use the law of total probability to calculate mean
        mean = 0
        for fix_combination in combinations:
            weight = 1
            fixes = copy.copy(fixed_means)
            for (srv, (fix, prob)) in zip(shared_roots, fix_combination):
                weight *= prob
                fixes[srv] = fix
            mean += weight * self.rv1.mean(fixes) * self.rv2.mean(fixes)
        return mean


class ExponentDiscreteRandVar(DiscreteRandVar):

    def __init__(self, rv, power):

        def pmf(x):
            root = x ** (1 / float(power))
            if power % 2 == 1:
                return rv._get_probability(root)
            else:
                prob1 = rv._get_probability(root) if root in rv.sample_space else 0
                prob2 = rv._get_probability(-root) if -root in rv.sample_space else 0
                return prob1 + prob2

        DiscreteRandVar.__init__(self, {(x ** power) for x in rv.sample_space}, pmf)
        self.rv = rv
        self.power = power

        rv.children.add(self)
        self.parents.add(rv)


    def _new_sample(self):
        assert(len(self.parents) == 1)
        return self.rv.sample() ** self.power


    def _new_mean(self, fixed_means):
        if self.rv in fixed_means:
            return fixed_means[self.rv] ** self.power
        # Applying transformation theorem to calculate E[X^n].
        # Should be slightly faster than naive approach because
        # integer-valued exponentiations can be done in log time.
        mean = 0
        for x in self.rv.sample_space:
            p = self.rv._get_probability(x)
            mean += p * (x ** self.power)
        return mean
