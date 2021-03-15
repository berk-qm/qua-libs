from qm.qua import *
from abc import ABC, abstractmethod


class Loss(ABC):

    def __init__(self, output_size=None):
        self._gradient_ = None
        if output_size:
            self._gradient_ = declare(fixed, size=output_size)
        self._loss_ = declare(fixed)

    @abstractmethod
    def forward(self, pred, label):
        pass

    @abstractmethod
    def backward(self, pred, label):
        pass


class MeanSquared(Loss):
    def __init__(self, output_size=None):
        super().__init__(output_size)
        self._k_ = declare(int)
        self._diff_ = declare(fixed)

    def forward(self, pred, label):
        assign(self._loss_, 0)
        with for_(self._k_, 0, self._k_ < pred.length(), self._k_ + 1):
            assign(self._diff_, pred[self._k_] - label[self._k_])
            assign(self._loss_, self._loss_ + 0.5 * self._diff_ * self._diff_)
            assign(self._gradient_[self._k_], self._diff_)

    def backward(self, pred, label):
        pass
