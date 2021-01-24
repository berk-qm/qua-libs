from qm.qua import *


class NeuralNetwork:
    """
    Implements a neural network in qua
    """

    def __init__(self, layers):
        self.layers = layers
        self.depth = len(layers)
        self.input_size = layers[0].input_size
        self.output_size = layers[-1].output_size

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, l):
        for i in range(len(l) - 1):
            if l[i].output_size != l[i + 1].input_size:
                raise ValueError(f"The input/output sizes of layers {i} and {i + 1} must match")
        self._layers = l

    def feed_forward(self, input_var, output_var=None, save_to=None):
        """
        Propagate the input through the layer. Implements matrix multiplication

        :param input_var: a Qua array containing the input to the layer
        :param output_var: a Qua array to contain the output of the layer
        :param save_to: a tag or stream to save the output to
        """



class DenseLayer:
    """
    Implementation of fully connected layer in Qua
    """

    def __init__(self, input_size, output_size, weights, activation=None):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = weights
        self.activation = activation

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, array):
        if array.shape[1] == self.input_size and array.shape[0] == self.output_size:
            self._weights = array
        else:
            raise ValueError("Weights array size must match the input and output sizes of the layer")

    def feed_forward(self, input_var, output_var=None, save_to=None):
        """
        Propagate the input through the layer. Implements matrix multiplication

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
