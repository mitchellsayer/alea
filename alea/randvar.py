from abc import ABC, abstractmethod
from weakref import WeakSet
from collections import deque


class RandVar(ABC):
    '''
    Represents a basic abstraction around a random variable.
    A random variable can be sampled to produce some number,
    has information about its mean, variance, and covariance with
    other random variables, and can be added/multiplied/transformed
    to produce new random variables.

    Note that a random variable holds strongly on to its parents but
    weakly on to its children. This is because the properties of a
    random variable depend strongly on what its parent random
    variables. However, its distribution is not affected by any
    child random variables. Thus, if any child random variables are
    garbage collected, it will not affect this random variable.
    '''

    def __init__(self):
        self.saved_sample = None
        self.saved_mean = None
        self.saved_variance = None
        self.saved_covariances = {}
        self.parents = set()
        self.children = WeakSet()
        self.saved_roots = None


    def roots(self):
        '''
        Finds the roots associated with the random variable. Root random variables
        are direct mappings from a probability space. The roots of a random variable
        will never change, so after they are found, they are simply cached.

        Returns:
            An immutable set of root random variables
        '''
        if self.saved_roots is None:
            self.saved_roots = self._new_roots()
        return self.saved_roots


    def sample(self):
        '''
        Returns the most recently generated numerical sample for this random variable.
        Will generate the sample if none exist.

        Returns:
            The sample as a number
        '''

        if self.saved_sample is None:
            self.resample()
        return self.saved_sample


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

        # Topologically sort the subgraph that is
        # connected to the root nodes 
        visited = set()
        topo = deque()
        def visit(node):
            if node in visited:
                return
            for child in node.children:
                visit(child)
            visited.add(node)
            topo.appendleft(node)
        for node in self.roots():
            visit(node)

        # In topological order, generate new samples
        for node in topo:
            node.saved_sample = node._new_sample()


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

        mean = 0
        for _ in range(trials):
            self.resample()
            mean += self.sample()
        return mean / trials


    def sample_variance(self, trials=10000):
        '''
        Performs a point estimate of the variance after calculating
        an approximate mean. As with the sample mean, a large
        number of samples will result in better approximation of
        the variance but at a significant time cost.

        Args:
            trials: The number of samples to take

        Returns:
            An approximation of the variance
        '''

        mean = self.sample_mean(trials)
        variance = 0
        for _ in range(trials):
            self.resample()
            variance += (self.sample() - mean) ** 2
        return variance / (trials - 1)


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

        if self in fixed_means:
            return fixed_means[self]
        if len(fixed_means) > 0:
            return self._new_mean(fixed_means)
        if self.saved_mean is None:
            self.saved_mean = self._new_mean(fixed_means)
        return self.saved_mean


    def variance(self):
        '''
        Returns the theoretical variance. Like the mean, it is acquired using
        complex numerical calculations. It is also cached after being calculated.

        Returns:
            The theoretical variance of the random variable
        '''

        if self.saved_variance is None:
            self.saved_variance = self._new_variance()
        return self.saved_variance


    def covariance(self, rv):
        '''
        Given this random variable {self} and another random variable {rv},
        calculates the covariance between the two random variables. Covariances
        are also cached and the cache space is compressed by taking advantage
        of the fact that covariance is symmetric between two random variables.

        Returns:
            The theoretical covariance between this random variable and another
        '''

        if rv in self.saved_covariances:
            result = self.saved_covariances[rv]
        # Covariance is symmetric: Cov[X, Y] = Cov[Y, X]
        elif self in rv.saved_covariances:
            result = rv.saved_covariances[self]
        else:
            result = self._new_covariance(rv)
            self.saved_covariances[rv] = result
        return result


    def _new_roots(self):
        '''Default approach to use BFS to find sources of a graph'''

        queue = deque([self])
        roots= set()
        while len(queue) > 0:
            curr = queue.popleft()
            if len(curr.parents) == 0:
                roots.add(curr)
            else:
                queue.extend(curr.parents)
        return roots


    @abstractmethod
    def _new_sample(self):
        '''Implemented by subclasses, represents the calculation of a
        potentially recursive sample'''

        pass


    @abstractmethod
    def _new_mean(self, fixed_means):
        '''Implemented by subclasses, represents the calculation of the
        theoretical mean without caching'''

        pass


    @abstractmethod
    def _new_variance(self):
        '''Implemented by subclasses, represents the calculation of the
        theoretical variance without caching'''

        pass


    @abstractmethod
    def _new_covariance(self):
        '''Implemented by subclasses, represents the calculation of
        theoretical covariance without caching'''

        pass


    @abstractmethod
    def __add__(self, obj):
        pass


    @abstractmethod
    def __mul__(self, obj):
        pass


    @abstractmethod
    def __pow__(self, num):
        pass


    def __sub__(self, obj):
        return self + (obj * -1)

