from qm.qua import *
from abc import ABC, abstractmethod


class Loss(ABC):
    def __init__(self, output_size=None):
        self._error_ = (
            None  # contains a vector of the difference between the output and the label
        )
        if output_size:
            self._error_ = declare(fixed, size=output_size)
        self._loss_ = declare(fixed)  # constains the total loss

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

    def forward(self, pred, label):
        assign(self._loss_, 0.0)
        with for_(self._k_, 0, self._k_ < pred.length(), self._k_ + 1):
            assign(self._error_[self._k_], pred[self._k_] - label[self._k_])
            assign(
                self._loss_,
                self._loss_ + 0.5 * self._error_[self._k_] * self._error_[self._k_],
            )

    def backward(self, pred, label):
        pass

