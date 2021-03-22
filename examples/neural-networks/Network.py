from qm.qua import *
from Layer import *
from Activation import *
from Loss import *
from Initializer import *


class Network:
    """
    Implements a neural network in qua
    """

    def __init__(self, *layers, loss=None, learning_rate=0.01, name="nn"):
        """
        :param layers: a list of layers in the desired order
        """
        self.layers = layers
        self.loss: Loss = loss
        self.learning_rate = learning_rate
        self.name = name
        self.depth = len(layers)
        self._res_ = declare(fixed, size=self._output_size)
        self._index_ = declare(int)
        self._input = None
        self._results_stream_ = declare_stream()

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, l):
        for i in range(len(l) - 1):
            if l[i].output_size != l[i + 1].input_size:
                raise ValueError(
                    f"The input/output sizes of layers {i} and {i + 1} must match"
                )
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

    def forward(self, input_var, output_var=None):
        """
        Propagate the input through the network. Implements matrix multiplication

        :param input_var: a Qua array containing the input to the net
        :param output_var: a Qua array to contain the output of the net
        """
        self._input = input_var
        for i in range(self.depth - 1):
            layer = self.layers[i]
            layer.forward(input_var)
            input_var = layer._res_
        layer = self.layers[self.depth - 1]
        layer.forward(input_var, self._res_)

        if output_var:
            with for_(
                self._index_, 0, self._index_ < self._output_size, self._index_ + 1
            ):
                assign(output_var[self._index_], self._res_[self._index_])
                save(self._res_[self._index_], self._results_stream_)
        else:
            with for_(
                self._index_, 0, self._index_ < self._output_size, self._index_ + 1
            ):
                save(self._res_[self._index_], self._results_stream_)

    def backprop(self, label):
        self.loss.forward(self._res_, label)
        save(self.loss._loss_, f"{self.name}_loss_stream")

        error = self.loss._error_
        for i in range(self.depth - 1, -1, -1):
            layer = self.layers[i]
            if i > 0:
                input_ = self.layers[i - 1]._res_
            else:
                input_ = self._input
            layer.backward(error, input_, self.learning_rate)
            error = layer._error_

    def training_step(self, input_, label):
        self.forward(input_)
        self.backprop(label)

    def save_weights(self):
        for i in range(self.depth):
            layer = self.layers[i]
            layer.save_weights_(f"{self.name}_layer{i}_")

    def save_results(self, tag="_results_stream"):
        with stream_processing():
            self._results_stream_.buffer(self._output_size).save_all(self.name + tag)
