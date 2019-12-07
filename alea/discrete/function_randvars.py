from .discrete_randvar import DiscreteRandVar

import copy
import itertools


class UnaryDiscreteRandVar(DiscreteRandVar):
    '''
    Given a random variable and a function, this returns the
    new random variable that is the result of applying the
    function to the output of the original random variable.
    That is, given a discrete random variable X and
    a function g, this is the random variable g(X).

    The function g must be well-defined for every value
    in the support of X and g(X) must satisfy the properties
    of a discrete random variable.

    Attributes:
        Inherited attributes from DiscreteRandVar
    '''

    def __init__(self, rv, func):
        DiscreteRandVar.__init__(self)
        self.rv = rv
        self.func = func

        rv.children.add(self)
        self.parents.add(rv)


    def _new_roots(self):
        return self.rv.roots()


    def _new_sample(self):
        return self.func(self.rv.sample())


    def _new_mean(self, fixed_means):
        # Uses the same computation process as that of multiplication
        # Start at the roots and work our way up
        roots = self.roots()
        rrv_supports = []
        for rrv in roots:
            if rrv in fixed_means:
                rrv_space = [(fixed_means[rrv], 1)]
            else:
                rrv_space = [(x, rrv.probability_of(x)) for x in rrv.sample_space]
            rrv_supports.append(rrv_space)
        mean = 0
        for combination in itertools.product(*rrv_supports):
            weight = 1
            fixes = copy.copy(fixed_means)
            for (rrv, (fix, prob)) in zip(roots, combination):
                weight *= prob
                fixes[rrv] = fix
            mean += weight * self.func(self.rv.mean(fixes))
        return mean
