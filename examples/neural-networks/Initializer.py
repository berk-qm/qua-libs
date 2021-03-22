import numpy as np
from abc import ABC, abstractmethod


class Initializer(ABC):
    def __init__(self, shape):
        self.shape = shape

    @abstractmethod
    def get_weights(self, shape=None):
        pass


class Uniform(Initializer):
    def __init__(self, shape=None, low=-0.5, high=0.5):
        super().__init__(shape)
        self.low = low
        self.high = high

    def get_weights(self, shape=None):
        shape = shape if shape else self.shape
        return np.random.uniform(self.low, self.high, size=shape)


class Normal(Initializer):
    def __init__(self, shape=None, mean=0, scale=0.5):
        super().__init__(shape)
        self.mean = mean
        self.scale = scale

    def get_weights(self, shape=None):
        shape = shape if shape else self.shape
        return np.random.normal(self.mean, self.scale, size=shape)
