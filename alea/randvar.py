from abc import ABC, abstractmethod
from weakref import WeakSet
from collections import deque


class RandVar(ABC):

    def __init__(self):
        self.saved_sample = None
        self.saved_mean = None
        self.saved_variance = None
        self.saved_covariances = {}
        self.parents = set()
        self.children = WeakSet()


    def sample(self):
        if self.saved_sample is None:
            self.resample()
        return self.saved_sample


    def resample(self):
        # Identify root random variables
        queue = deque([self])
        roots = []
        while len(queue) > 0:
            curr = queue.popleft()
            if len(curr.parents) == 0:
                roots.append(curr)
            else:
                queue.extend(curr.parents)

        # Topologically sort the subgraph that is
        # connected to the root nodes 
        visited = set()
        topo = deque()
        # queue.extend(roots)
        # while len(queue) > 0:
        #     curr = queue.pop()
        #     for child in curr.children:
        #         if child not in visited:
        #             queue.append(child)
        #             visited.add(child)
        #     topo.appendleft(curr)
        def visit(node):
            if node in visited:
                return
            for child in node.children:
                visit(child)
            visited.add(node)
            topo.appendleft(node)
        for node in roots:
            visit(node)

        # In topological order, generate new samples
        for node in topo:
            node.saved_sample = node._new_sample()


    def sample_average(self, n=10000):
        mean = 0
        for _ in range(n):
            self.resample()
            mean += self.sample() / n
        return mean


    def mean(self, fixed_means={}):
        if self in fixed_means:
            return fixed_means[self]
        if len(fixed_means) > 0:
            return self._new_mean(fixed_means)
        if self.saved_mean is None:
            self.saved_mean = self._new_mean(fixed_means)
        return self.saved_mean


    def variance(self):
        if self.saved_variance is None:
            self.saved_variance = self._new_variance()
        return self.saved_variance


    def covariance(self, rv):
        if rv in self.saved_covariances:
            result = self.saved_covariances[rv]
        else:
            result = self._new_covariance(rv)
            self.saved_covariances[rv] = result
        return result


    @abstractmethod
    def _new_mean(self, fixed_means):
        pass


    @abstractmethod
    def _new_variance(self):
        pass


    @abstractmethod
    def _new_sample(self):
        pass


    @abstractmethod
    def _new_covariance(self):
        pass


    @abstractmethod
    def __add__(self, obj):
        pass


    @abstractmethod
    def __mul__(self, obj):
        pass


    def __sub__(self, obj):
        return self + (obj * -1)


    @abstractmethod
    def __pow__(self, num):
        pass

