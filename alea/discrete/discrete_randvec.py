from ..randvec import RandVec
from .discrete_randvar import DiscreteRandVar
from .root_randvar import RootDiscreteRandVar
from .function_randvar import UnaryDiscreteRandVar

import copy

class DiscreteRandVec(RandVec):
    '''
    A discrete random variable is a strict classification of random
    variables. It fits either one of these two criteria:
        1. Has a well-defined probability mass function and corresponding
        sample space. We call this a 'root' discrete random variable. Think
        of this variable as the result of an isolated, real-world experiment.
        2. Is constructed using other discrete random variables, which may or
        may not be roots. These constructions can include additions, multiplications,
        subtractions, and arbitrary transformations.
    '''

    def __init__(self, sample_space, mass_function):
        sample_list = list(sample_space)
        root_pmf = lambda x : mass_function(sample_list[x])
        self.root = RootDiscreteRandVar(list(range(len(sample_space))), root_pmf)
        randvars = []
        for idx in range(len(sample_list[0])):
            def transformation(x, i=idx):
                return sample_list[x][i]
            randvars.append(UnaryDiscreteRandVar(root, transformation))
        RandVec.__init__(self, randvars)


    def resample(self):
        self.root.resample()
