from qm.qua import *
from abc import ABC, abstractmethod


class Loss(ABC):

    def __init__(self, func=None, gradient=None):
        self.func = func
        self.gradient = gradient

    @abstractmethod
    def forward(self, pred, true):
        pass

    @abstractmethod
    def backward(self, pred, true):
        pass


class MeanSquared(Loss):
    def __init__(self, output_size=None):
        super().__init__()
        if output_size:
            self._gradient_ = declare(fixed, size=output_size)
        self._k_ = declare(int)
        self._loss_ = declare(fixed)

    def forward(self, pred, true):
        assign(self._loss_, 0)
        with for_(self._k_, 0, self._k_ < pred.length(), self._k_ + 1):
            assign(self._loss_,
                   self._loss_ + 0.5 * (pred[self._k_] - true[self._k_]) * (pred[self._k_] - true[self._k_]))

    def backward(self, pred, true):
        with for_(self._k_, 0, self._k_ < pred.length(), self._k_ + 1):
            assign(self._gradient_[self._k_], (pred[self._k_] - true[self._k_]))
