import numpy as np


class RandVec:
    '''
    Represents a basic abstraction around a random variable.
    A random variable can be sampled to produce some number,
    has information about its mean, variance, and covariance with
    other random variables, and can be added/multiplied/transformed
    to produce new random variables.
    '''

    def __init__(self, randvars):
        self.randvars = randvars


    def sample(self):
        '''
        Returns the most recently generated numerical sample for this random variable.
        Will generate the sample if none exist.

        Returns:
            The sample as a number
        '''
        return [x.sample() for x in self.randvars]


    def resample(self):
        '''
        Generates a new sample for this random variable.
        This may cause other random variables to re-sample as well in the interest of
        consistency. Specifically, the resampling procedure works as follows:
            1. Find all ancestors of this random variable that are roots. A root random
            variable has no parents and represents a completely independent event
            occurring in nature.
            2. Resample these roots.
            3. Propogate the result of the roots up through the graph. The propogation
            will occur in topological order, since this graph is a DAG. This ensures that,
            before a random variable is resampled, all of its parents will have been
            resampled. Thus, the given random variable can access its parent's samples
            to perform its own calculations.
        '''

        for x in self.randvars:
            x.resample()


    def sample_mean(self, trials=10000):
        '''
        Performs a point estimate of the mean by simply
        sampling for {trial} amount of times and then averaging the
        samples. This will always work because of the law of
        large numbers.

        Note that a large number of samples will result in better
        approximation of the mean but will take more time to generate.
        A random variable will a high variance will converge to the
        sampled mean more slowly.

        Args:
            trials: The number of samples to take

        Returns:
            An approximation of the mean
        '''

        return [x.sample_mean(trials) for x in self.randvars]


    def sample_variance(self, trials=10000):
        '''
        Performs a point estimate of the variance after calculating
        an approximate mean. As with the sample mean, a large
        number of samples will result in better approximation of
        the variance but at a significant time cost.

        Args:
            trials: The number of samples to take

        Returns:
            An approximation of the mean
        '''

        return [x.sample_variance(trials) for x in self.randvars]


    def mean(self, fixed_means={}):
        '''
        Returns the theoretical mean. This is calculated using complex
        numerical calculations. However, not only is it numerically
        precise, but it represents what the sample average should
        converge to. The theoretical mean is cached upon calculation because
        random variable distributions are immutable.

        The {fixed_means} dictionary is a mapping from random variables
        to numbers. It is used for internal calcuations, specifically in
        the multiplication of dependent random variables. It allows us
        to 'fix' a random variable's mean to a certain value, essentially
        bypassing any normal recursive calculations that would be performed.
        As a user, you will likely not need to use this dictionary.

        Args:
            fixed_means: A dictionary mapping random variables to preset means

        Returns:
            The theoretical mean of the random variable
        '''

        return [x.mean(fixed_means) for x in self.randvars]


    def variance(self):
        '''
        Returns the theoretical variance. Like the mean, it is acquired using
        complex numerical calculations. It is also cached after being calculated.

        Returns:
            The theoretical variance of the random variable
        '''

        return self.cross_covariance(self)


    def cross_covariance(self, othervec):
        '''
        Given this random variable {self} and another random variable {rv},
        calculates the covariance between the two random variables. Covariances
        are also cached and the cache space is compressed by taking advantage
        of the fact that covariance is symmetric between two random variables.

        Returns:
            The theoretical covariance between this random variable and another
        '''

        vrnce = np.empty((len(self.randvars), len(othervec.randvars)))
        for i in range(len(self.randvars)):
            for j in range(len(othervec.randvars)):
                vrnce[i][j] = self.randvars[i].covariance(othervec.randvars[j])
        return vrnce


