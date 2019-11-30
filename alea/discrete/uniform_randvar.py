from .discrete_randvar import DiscreteRandVar

import random

class UniformRandVar(DiscreteRandVar):

    def __init__(self, sample_space):
        p = 1 / len(sample_space)
        DiscreteRandVar.__init__(self, sample_space, lambda x : p)


    def _get_sample(self):
        return random.choice(self.sample_space)
