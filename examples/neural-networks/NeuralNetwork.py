from qm.qua import *
import numpy as np


class NeuralNetwork:
    """
    Implements a neural network in qua
    """

    def __init__(self, layers):
        """
        :param layers: a list of layers in the desired order
        """
        self.layers = layers
        self.depth = len(layers)

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, l):
        for i in range(len(l) - 1):
            if l[i].output_size != l[i + 1].input_size:
                raise ValueError(f"The input/output sizes of layers {i} and {i + 1} must match")
        self._layers = l
        self._input_size = self._layers[0].input_size
        self._output_size = self._layers[-1].output_size

    def feed_forward(self, input_var, output_var=None, save_to=None):
        """
        Propagate the input through the network. Implements matrix multiplication

        :param input_var: a Qua array containing the input to the net
        :param output_var: a Qua array to contain the output of the net
        :param save_to: a tag or stream to save the output to
        """
        for i in range(self.depth - 1):
            layer = self.layers[i]
            temp_output = declare(fixed, size=layer.output_size)
            layer.feed_forward(input_var, temp_output)
            input_var = temp_output
        layer = self.layers[self.depth - 1]
        layer.feed_forward(input_var, output_var, save_to)


class DenseLayer:
    """
    Implementation of fully connected layer in Qua
    """

    def __init__(self, weights, activation=None):
        self.weights = weights
        self.activation = activation

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

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    def feed_forward(self, input_var, output_var=None, save_to=None):
        """
        Propagate the input through the layer. Implements a matrix multiplication

        :param input_var: a Qua array containing the input to the layer
        :param output_var: a Qua array to contain the output of the layer
        :param save_to: a tag or stream to save the output to
        """
        w = declare(fixed, value=self.weights.flatten().tolist())
        j = declare(int)
        k = declare(int)
        res = declare(fixed)
        with for_(k, 0, k < self.output_size, k + 1):
            assign(res, 0)
            with for_(j, 0, j < self.input_size, j + 1):
                assign(res, res + input_var[j] * w[k * self.input_size + j])
            if self.activation:
                self.activation(res)
            if output_var:
                assign(output_var[k], res)
            if save_to:
                save(res, save_to)
