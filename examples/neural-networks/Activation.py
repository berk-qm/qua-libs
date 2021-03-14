from qm.qua import *
from abc import ABC, abstractmethod


class Activation(ABC):

    def __init__(self, func=None, gradient=None):
        self.func = func
        self.gradient = gradient

    @abstractmethod
    def forward(self, var):
        self.func(var)

    @abstractmethod
    def backward(self, var):
        self.gradient(var)


class ReLu(Activation):
    def __init__(self):
        super().__init__()
        self._zero_ = declare(fixed, value=0)
        self._one_ = declare(fixed, value=1)

    def forward(self, var):
        with if_(var < 0):
            assign(var, self._zero_)

    def backward(self, var):
        with if_(var < 0):
            assign(var, self._zero_)
        with else_():
            assign(var, self._one_)
