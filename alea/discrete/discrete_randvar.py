from ..randvar import RandVar
from collections import deque, defaultdict

import itertools
import copy


class DiscreteRandVar(RandVar):
    '''
    A discrete random variable is a strict classification of random
    variables. It fits either one of these two criteria:
        1. Has a well-defined probability mass function and corresponding
        sample space. We call this a 'root' discrete random variable. Think
        of this variable as the result of an isolated, real-world experiment.
        2. Is constructed using other discrete random variables, which may or
        may not be roots. These constructions can include additions, multiplications,
        subtractions, and arbitrary transformations.

    Attributes:
        Inherited attributes from RandVar
    '''

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
        # For now, perform exponentiation by squaring, reducing
        # exponentiation to logarithmic time
        if not isinstance(num, int):
            raise ValueError("Right operand must be an integer")
        if num < 1:
            raise ValueError("Exponent must be greater than zero")
        elif num == 1:
            return self
        elif num % 2 == 0:
            return (self * self) ** (num // 2)
        else:
            return self * (self ** (num - 1))


class ConstantPlusDiscreteRandVar(DiscreteRandVar):

    def __init__(self, rv, c):
        DiscreteRandVar.__init__(self)
        self.rv = rv
        self.c = c

        rv.children.add(self)
        self.parents.add(rv)


    def _new_roots(self):
        return self.rv.roots()


    def _new_sample(self):
        return self.rv.sample() + self.c


    def _new_mean(self, fixed_means):
        return self.rv.mean(fixed_means) + self.c


    def _new_variance(self):
        return self.rv.variance()


class DiscretePlusDiscreteRandVar(DiscreteRandVar):

    def __init__(self, rv1, rv2):
        DiscreteRandVar.__init__(self)
        self.rv1 = rv1
        self.rv2 = rv2

        rv1.children.add(self)
        rv2.children.add(self)
        self.parents.add(rv1)
        self.parents.add(rv2)


    def _new_roots(self):
        return self.rv1.roots().union(self.rv2.roots())


    def _new_sample(self):
        return self.rv1.sample() + self.rv2.sample()


    def _new_mean(self, fixed_means):
        return self.rv1.mean(fixed_means) + self.rv2.mean(fixed_means)


    def _new_variance(self):
        return self.rv1.variance() + self.rv2.variance() + 2 * self.rv1.covariance(self.rv2)


class ConstantTimesDiscreteRandVar(DiscreteRandVar):

    def __init__(self, rv, c):
        DiscreteRandVar.__init__(self)
        self.rv = rv
        self.c = c

        rv.children.add(self)
        self.parents.add(rv)


    def _new_roots(self):
        return self.rv.roots()


    def _new_sample(self):
        return self.rv.sample() * self.c


    def _new_mean(self, fixed_means):
        return self.rv.mean(fixed_means) * self.c


    def _new_variance(self):
        return self.rv.variance() * self.c * self.c


class DiscreteTimesDiscreteRandVar(DiscreteRandVar):

    def __init__(self, rv1, rv2):
        DiscreteRandVar.__init__(self)
        self.rv1 = rv1
        self.rv2 = rv2

        rv1.children.add(self)
        rv2.children.add(self)
        self.parents.add(rv1)
        self.parents.add(rv2)


    def _new_roots(self):
        return self.rv1.roots().union(self.rv2.roots())


    def _new_sample(self):
        return self.rv1.sample() * self.rv2.sample()


    def _new_mean(self, fixed_means):
        shared_roots = list(self.rv1.roots().intersection(self.rv2.roots()))

        # If X and Y do not share any roots, then they are independent
        # This implies that E[XY] = E[X]E[Y], which is a quick calculation
        if len(shared_roots) == 0:
            return self.rv1.mean(fixed_means) * self.rv2.mean(fixed_means)

        # X and Y are dependent random variables! This would be impossible to calculate.
        # However, we know that because X and Y are dependent, they must share at least
        # one root discrete variable acting as a probabilistic generation.

        # If we generate every possible combination that the shared roots can take, we 
        # can make X and Y 'independent' again
        srv_supports = []
        for srv in shared_roots:
            if srv in fixed_means:
                srv_space = [(fixed_means[srv], 1)]
            else:
                srv_space = [(x, srv.probability_of(x)) for x in srv.sample_space]
            srv_supports.append(srv_space)
        # Then, we use the law of total probability to calculate mean
        mean = 0
        for combination in itertools.product(*srv_supports):
            weight = 1
            fixes = copy.copy(fixed_means)
            for (srv, (fix, prob)) in zip(shared_roots, combination):
                weight *= prob
                fixes[srv] = fix
            mean += weight * self.rv1.mean(fixes) * self.rv2.mean(fixes)
        return mean
