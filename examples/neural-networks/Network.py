from qm.qua import *
from Layer import *
from Activation import *
from Loss import *


class Network:
    """
    Implements a neural network in qua
    """

    def __init__(self, *layers, loss=None, learning_rate=0.01):
        """
        :param layers: a list of layers in the desired order
        """
        self.layers = layers
        self.loss: Loss = loss
        self.learning_rate = learning_rate
        self.depth = len(layers)
        self._res_ = declare(fixed, size=self._output_size)
        self._index_ = declare(int)

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

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, loss):
        if loss:
            self._loss = loss.__class__(output_size=self._output_size)

    def forward(self, input_var, output_var=None, stream_or_tag=None):
        """
        Propagate the input through the network. Implements matrix multiplication

        :param input_var: a Qua array containing the input to the net
        :param output_var: a Qua array to contain the output of the net
        :param stream_or_tag: a tag or stream to save the output to
        """
        for i in range(self.depth - 1):
            layer = self.layers[i]
            layer.forward(input_var)
            input_var = layer._res_
        layer = self.layers[self.depth - 1]
        layer.forward(input_var, self._res_, stream_or_tag)
        if output_var:
            with for_(self._index_, 0, self._index_ < self._output_size, self._index_ + 1):
                assign(output_var[self._index_], self._res_[self._index_])

    def backprop(self, label):
        self.loss.forward(self._res_, label)
        for i in range(self.depth - 1, -1, -1):
            pass
