from ..randvec import RandVec
from .root_randvar import RootDiscreteRandVar
from .unary_randvar import UnaryDiscreteRandVar

class DiscreteRandVec(RandVec):
    '''
    A discrete random vector has a set of k-length vectors
    describing its support along with a joint probability
    distribution that maps each vector in its support to a
    non-zero probability. This class will produce k dependent
    discrete random variables that satisfy the given parameters.
    '''

    def __init__(self, sample_space, mass_function):
        '''
        The support is a set of k-length numerical vectors
        representing the values that the random vector
        can take with non-zero probability. The mass
        function maps a vector in the support to a
        probability.

        Together, the support and mass function must
        form a valid probability distribution. Notably,
        the sum of all probabilities must be equal to 1.

        Vectors are preferrably represented by tuples. It is
        also allowable to represent them via numpy 2D arrays.

        Args:
            sample_space: A set of k-length vectors that the
            random vector can take
            mass_function: A function taking in a vector in
            the support and outputing a non-zero probability
        '''

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
