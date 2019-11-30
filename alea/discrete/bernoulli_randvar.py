from .discrete_randvar import DiscreteRandVar

import random

class BernoulliRandVar(DiscreteRandVar):

    def __init__(self, success_rate):
        def pmf(x):
            if x == 0:
                return 1 - success_rate
            else:
                return success_rate
        DiscreteRandVar.__init__(self, {0, 1}, pmf)
        self.success_rate = success_rate


    def _get_sample(self):
        return 1 if random.uniform(0, 1) < self.success_rate else 0


    def _get_mean(self):
        return self.success_rate


    def _get_variance(self):
        return self.success_rate * (1 - self.success_rate)
