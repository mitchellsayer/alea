from .randvar_base import BaseDiscreteRandVar

class DiscreteRandVar(BaseDiscreteRandVar):

    def __init__(self, sample_space, mass_function):
        BaseDiscreteRandVar.__init__(self, sample_space, mass_function)


    def __add__(self, obj):
        if isinstance(obj, int) or isinstance(obj, float):
            return ConstantPlusDiscreteRandVar(self, obj)
        # TODO: Add support for discrete random variables 
        else:
            raise ValueError("Right operand must be a constant or random variable")


    def __mul__(self, obj):
        # TODO: Generic multiplication for constants, other discrete random variables
        pass


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
