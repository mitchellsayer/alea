from .root_randvar import RootDiscreteRandVar

import random
import math


class BernoulliRandVar(RootDiscreteRandVar):
    '''
    A Bernoulli random variable models an experiment
    where the outcome can only be success or failure and
    success occurs with probability p. The variable
    outputs 1 with probability p and 0 with probability
    1 - p.
    '''

    def __init__(self, success_rate):

        def pmf(x):
            if x == 0:
                return 1 - success_rate
            else:
                return success_rate

        RootDiscreteRandVar.__init__(self, {0, 1}, pmf)
        self.success_rate = success_rate


    def _new_sample(self):
        return 1 if random.uniform(0, 1) < self.success_rate else 0


    def _new_mean(self, fixed_means):
        return self.success_rate


    def _new_variance(self):
        return self.success_rate * (1 - self.success_rate)


class BinomialRandVar(RootDiscreteRandVar):
    '''
    A Binomial random variable models a sequence of
    n Bernoulli random variables where success occurs
    with probability p. The variable outputs how
    many successes occurred.
    '''

    def __init__(self, trials, success_rate):

        def pmf(x):
            # https://stackoverflow.com/questions/3025162/statistics-combinations-in-python 
            def choose(n, k):
                if math.isclose(n, math.floor(n)):
                    n = int(n)
                if math.isclose(k, math.floor(k)):
                    k = int(k)
                if 0 <= k <= n: 
                    ntok = 1
                    ktok = 1
                    for t in range(1, min(k, n - k) + 1):
                        ntok *= n
                        ktok *= t
                        n -= 1
                    return ntok // ktok
                else:
                    return 0
            return choose(trials, x) * (success_rate ** x) * ((1 - success_rate) ** (trials - x))

        RootDiscreteRandVar.__init__(self, set(range(trials + 1)), pmf)
        self.trials = trials
        self.success_rate = success_rate


    def _new_sample(self):
        X = BernoulliRandVar(self.success_rate)
        successes = 0
        for _ in range(self.trials):
            X.resample()
            successes += X.sample()
        return successes


    def _new_mean(self, fixed_means):
        return self.trials * self.success_rate


    def _new_variance(self):
        return self.trials * self.success_rate * (1 - self.success_rate)


class UniformDiscreteRandVar(RootDiscreteRandVar):
    '''
    A uniform discrete random variable is a simplistic model of
    an experiment where there are n distinct outcomes, each outcome
    occurs with the same probability, and every outcome is mapped to
    a distinct number.
    '''

    def __init__(self, sample_space):
        p = 1 / len(sample_space)
        RootDiscreteRandVar.__init__(self, sample_space, lambda x : p)


    def _new_sample(self):
        if self.sample_list is None:
            self.sample_list = list(self.sample_space)
        return random.choice(self.sample_list)
