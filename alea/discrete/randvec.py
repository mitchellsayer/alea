from ..randvec import RandVec
from .root_randvar import RootDiscreteRandVar
from .function_randvar import UnaryDiscreteRandVar

class DiscreteRandVec(RandVec):
    '''
    A discrete random vector has a set of k-length vectors
    describing its support along with a joint probability
    distribution that maps each vector in its support to a
    non-zero probability. This class will produce k dependent
    discrete random variables that satisfy the given parameters.
    '''

    def __init__(self, sample_space, mass_function):
        sample_list = list(sample_space)
        root_pmf = lambda x : mass_function(sample_list[x])
        self.root = RootDiscreteRandVar(list(range(len(sample_space))), root_pmf)
        randvars = []
        for idx in range(len(sample_list[0])):
            def transformation(x, i=idx):
                return sample_list[x][i]
            randvars.append(UnaryDiscreteRandVar(self.root, transformation))
        RandVec.__init__(self, randvars)


    def resample(self):
        self.root.resample()
