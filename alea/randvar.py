from abc import ABC, abstractmethod

class RandVar(ABC):

    def __init__(self):
        self._mean = None
        self._variance = None 


    def mean(self):
        if self._mean is None:
            self._mean = self._get_mean()
        return self._mean


    def variance(self):
        if self._variance is None:
            self._variance = self._get_variance()
        return self._variance


    def sample(self):
        return self._get_sample()


    @abstractmethod
    def _get_sample(self):
        pass


    @abstractmethod
    def _get_mean(self):
        pass


    @abstractmethod
    def _get_variance(self):
        pass


    @abstractmethod
    def __add__(self, obj):
        pass


    @abstractmethod
    def __sub__(self, obj):
        pass


    @abstractmethod
    def __mul__(self, obj):
        pass
