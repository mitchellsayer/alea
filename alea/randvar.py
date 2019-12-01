from abc import ABC, abstractmethod
from weakref import WeakSet
from collections import deque

class RandVar(ABC):

    def __init__(self):
        self.saved_mean = None
        self.saved_variance = None
        self.saved_sample = None
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

        # Perform one-pass BFS, resampling as we go
        queue.extend(roots)
        while len(queue) > 0:
            curr = queue.popleft()
            curr.saved_sample = curr._new_sample()


    def mean(self):
        if self.saved_mean is None:
            self.saved_mean = self._new_mean()
        return self.saved_mean


    def variance(self):
        if self.saved_variance is None:
            self.saved_variance = self._new_variance()
        return self.saved_variance


    @abstractmethod
    def _new_mean(self):
        pass


    @abstractmethod
    def _new_variance(self):
        pass


    @abstractmethod
    def _new_sample(self):
        pass


    @abstractmethod
    def __add__(self, obj):
        pass


    @abstractmethod
    def __mul__(self, obj):
        pass


    def __sub__(self, obj):
        return self + (obj * -1)
