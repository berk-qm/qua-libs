from qm.qua import *
import numpy as np
from Activation import *
from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self, input_size=None, output_size=None, activation=None, weights=None, bias=None):
        self._input_size = input_size
        self._output_size = output_size
        self.activation: Activation = activation
        self.weights = weights
        self.bias = bias
        self._res_ = declare(fixed, size=self.output_size)
        self._z_ = declare(fixed, size=self.output_size)
        self._gradient_ = declare(fixed, size=self.output_size)

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, array):
        if type(array) != np.ndarray:
            raise TypeError("Weights must be given as a 2D Numpy array")
        self._weights = array
        self._input_size = self._weights.shape[1]
        self._output_size = self._weights.shape[0]
        # qua
        self._weights_ = declare(fixed, value=self.weights.flatten().tolist())

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, b):
        if b is not None:
            if len(b) != self._output_size:
                raise ValueError("The bias size must match the output size")

            # qua
            self._bias_ = declare(fixed, value=b.tolist())

        self._bias = b

    @abstractmethod
    def forward(self, input_var, output_var=None, save_to=None):
        pass

    @abstractmethod
    def backward(self, input_var, output_var=None):
        pass


class Dense(Layer):
    """
    Implementation of fully connected layer in Qua
    """

    def __init__(self, input_size=None, output_size=None, activation=None, weights=None, bias=None):
        super().__init__(input_size, output_size, activation, weights, bias)
        self._j_ = declare(int)
        self._k_ = declare(int)

    def forward(self, input_var, output_var=None, stream_or_tag=None):
        """
        Propagate the input through the layer. Implements a matrix multiplication

        :param input_var: a Qua array containing the input to the layer
        :param output_var: a Qua array to contain the output of the layer
        :param stream_or_tag: a tag or stream to save the output to
        """

        with for_(self._k_, 0, self._k_ < self.output_size, self._k_ + 1):
            assign(self._z_[self._k_], 0)

            with for_(self._j_, 0, self._j_ < self.input_size, self._j_ + 1):
                assign(self._z_[self._k_],
                       self._z_[self._k_] + input_var[self._j_] * self._weights_[self._k_ * self.input_size + self._j_])

            if self.bias is not None:
                assign(self._z_[self._k_], self._z_[self._k_] + self._bias_[self._k_])

            assign(self._res_[self._k_], self._z_[self._k_])

            if self.activation:
                self.activation.forward(self._res_[self._k_])

            if output_var:
                assign(output_var[self._k_], self._res_[self._k_])

            if stream_or_tag:
                save(self._res_[self._k_], stream_or_tag)

    def backward(self):
        pass
