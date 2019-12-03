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


    def _new_mean(self, fixed_means):
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
            variance += x ** 2 * p
        return variance - self.mean() ** 2


    def _new_covariance(self, rv):
        # Covariance is equal to E[XY] - E[X]E[Y] 
        return (self * rv).mean() - self.mean() * rv.mean()


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
        elif isinstance(obj, DiscreteRandVar):
            return DiscreteTimesDiscreteRandVar(self, obj)
        else:
            raise ValueError("Right operand must be a constant or random variable")


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
        cartesian_product = []
        for srv in shared_roots:
            if srv in fixed_means:
                sample_space = [(fixed_means[srv], 1)]
            else:
                sample_space = [(x, srv._get_probability(x)) for x in srv.sample_space]
            if len(cartesian_product) == 0:
                cartesian_product = [[x] for x in sample_space]
            else:
                ncp = []
                for xs in ncp:
                    for x in sample_space:
                        ncp.append(xs + [x])
                cartesian_product = ncp
        # Then, we use the law of total probability to calculate mean
        mean = 0
        for fixes in cartesian_product:
            weight = 1
            updated_fixed_means = copy.copy(fixed_means)
            for i in range(len(shared_roots)):
                srv = shared_roots[i]
                (fix, prob) = fixes[i]
                weight *= prob
                updated_fixed_means[srv] = fix
            mean += weight * self.rv1.mean(updated_fixed_means) * self.rv2.mean(updated_fixed_means)
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
