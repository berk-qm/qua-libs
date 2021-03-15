from qm.qua import *
from abc import ABC, abstractmethod


class Activation(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def forward(self, var):
        pass

    @abstractmethod
    def backward(self, var):
        pass


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
