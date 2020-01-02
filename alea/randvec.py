import numpy as np


class RandVec:
    '''
    Represents a basic abstraction around a random vector.
    A random vector is a k-length vector of random variables.
    Like with random variables, random vectors can be sampled and
    their mean and variance can be calculated.
    '''

    def __init__(self, randvars):
        self.randvars = randvars


    def __len__(self):
        '''
        The length of the random vector. Referred to as 'k'.

        Returns:
            The number of random variables in the vector
        '''

        return len(self.randvars)


    def sample(self):
        '''
        Returns the most recently generated numerical sample for
        this random vector. Will generate the sample if none exist.

        Returns:
            The sample as a k-length numpy array
        '''

        return np.asarray([x.sample() for x in self.randvars])


    def resample(self):
        '''
        Generates a new sample for this random variable, causing
        whatever random variables are in the vector to resample
        as well.
        '''

        for x in self.randvars:
            x.resample()


    def sample_mean(self, trials=10000):
        '''
        For each random variable, a point estimate of the variable's
        mean is calculated. This produces a vector of sample means,
        where the ith entry is the sample mean of the ith random
        variable.

        Args:
            trials: The number of samples to use in calculating
            each average

        Returns:
            An approximation of the mean as a k-length numpy array
        '''

        return np.asarray([x.sample_mean(trials) for x in self.randvars])


    def mean(self, fixed_means={}):
        '''
        The theoretical mean of a random vector is the vector such
        that the ith entry is the theoretical mean of the ith random
        variable. This takes into account fixed means as well.

        Args:
            fixed_means: A dictionary mapping random variables to
            preset means

        Returns:
            The theoretical mean of the random vector as a k-length
            numpy array
        '''

        return [x.mean(fixed_means) for x in self.randvars]


    def variance(self):
        '''
        Calculates the variance matrix of this random vector.
        Also known as the covariance or variance/covariance matrix.

        The variance matrix of a vector has interesting properties,
        namely that it is symmetric and postive-definite due to
        the properties of covariance. A vector of independent
        random variables, each with variance of v will produce
        vI where I is the k-by-k identity matrix.

        Returns:
            The variance matrix of the random vector as a k-by-k
            numpy matrix
        '''

        return self.cross_covariance(self)


    def cross_covariance(self, othervec):
        '''
        Calculates the cross-covariance matrix between this random
        vector and the given random vector. The (i, j)th entry of
        this matrix will be the covariance between the ith random
        variable of this vector and the jth random variable of the
        other vector.

        The resulting matrix will be m rows and n columns where m
        is the length of this vector and n is the length of the
        other vector.

        Returns:
            The cross covariance matrix between this and the
            given random vector as a m-by-n numpy matrix
        '''

        vrnce = np.empty((len(self.randvars), len(othervec.randvars)))
        for i in range(len(self.randvars)):
            for j in range(len(othervec.randvars)):
                vrnce[i][j] = self.randvars[i].covariance(othervec.randvars[j])
        return vrnce


