from qm.qua import *
from abc import ABC, abstractmethod


class Activation(ABC):
    def __init__(self):
        self._res_ = declare(fixed)

    @abstractmethod
    def forward(self, var):
        pass

    @abstractmethod
    def backward(self, var):
        pass


class Id(Activation):
    def __init__(self):
        super(Id, self).__init__()
        self._one_ = declare(fixed, value=1)

    def forward(self, var):
        assign(self._res_, var)

    def backward(self, var):
        assign(self._res_, self._one_)


class ReLu(Activation):
    def __init__(self):
        super().__init__()
        self._zero_ = declare(fixed, value=0)
        self._one_ = declare(fixed, value=1)

    def forward(self, var):
        with if_(var < 0):
            assign(self._res_, self._zero_)
        with else_():
            assign(self._res_, var)

    def backward(self, var):
        with if_(var < 0):
            assign(self._res_, self._zero_)
        with else_():
            assign(self._res_, self._one_)
