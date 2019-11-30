from .discrete_randvar import DiscreteRandVar

import random

class BinomialRandVar(DiscreteRandVar):

    def __init__(self, trials, success_rate):
        def pmf(x):
            # https://stackoverflow.com/questions/3025162/statistics-combinations-in-python 
            def choose(n, k):
                if 0 <= k <= n:
                    ntok = 1
                    ktok = 1
                    for t in xrange(1, min(k, n - k) + 1):
                        ntok *= n
                        ktok *= t
                        n -= 1
                    return ntok // ktok
                else:
                    return 0
            return choose(trials, x) * (success_rate ** x) * ((1 - success_rate) ** (trials - x))
        DiscreteRandVar.__init__(self, set(range(trials + 1)), pmf)
        self.trials = trials
        self.success_rate = success_rate


    def _get_sample(self):
        successes = 0
        for _ in self.trials:
            if random.uniform(0, 1) < self.success_rate:
                successes += 1
        return successes


    def _get_mean(self):
        return self.trials * self.success_rate


    def _get_variance(self):
        return self.trials * self.success_rate * (1 - self.success_rate)
