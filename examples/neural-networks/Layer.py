from qm.qua import *
import numpy as np
from Activation import *
from Initializer import *
from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(
        self,
        input_size=None,
        output_size=None,
        activation=None,
        initializer=Uniform(),
        weights=None,
        bias=None,
    ):
        self._input_size = input_size
        self._output_size = output_size
        self.activation: Activation = activation
        self.initializer: Initializer = initializer
        self.weights = weights
        self.bias = bias
        self._index_ = declare(int)
        self._weights_stream_ = declare_stream()
        self._bias_stream_ = declare_stream()
        self._res_ = declare(fixed, size=self.output_size)
        self._z_ = declare(fixed, size=self.output_size)
        self._error_ = declare(fixed, size=self.input_size)

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, act):
        self._activation = act if act else Id()

    @property
    @abstractmethod
    def initializer(self):
        pass

    @initializer.setter
    @abstractmethod
    def initializer(self, init):
        pass

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, array):
        if array is None:
            array = self.initializer.get_weights()
        else:
            if type(array) != np.ndarray:
                raise TypeError("Weights must be given as a 2D Numpy array")
        self._weights = array
        # qua
        self._weights_ = declare(fixed, value=self.weights.flatten().tolist())
        self._gradient_ = declare(fixed, value=self.weights.flatten().tolist())

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, b):
        if b is not None:
            if np.any(b.shape != self._output_size):
                raise ValueError("The bias size must match the output size")
            # qua
            self._bias_ = declare(fixed, value=b.flatten().tolist())
        else:
            # b = self.initializer.get_weights(self._output_size)
            b = np.zeros(self._output_size)
            self._bias_ = declare(fixed, value=b.flatten().tolist())

        self._bias = b

    @abstractmethod
    def forward(self, input_var, output_var=None, save_to=None):
        pass

    @abstractmethod
    def backward(self, error, input_, learning_rate):
        pass

    @abstractmethod
    def save_weights_(self, tag):
        pass


class Dense(Layer):
    """
    Implementation of fully connected layer in Qua
    """

    def __init__(
        self,
        input_size=None,
        output_size=None,
        activation=None,
        initializer=Uniform(),
        weights=None,
        bias=None,
    ):
        """

        @param input_size:
        @type input_size:
        @param output_size:
        @type output_size:
        @param activation:
        @type activation:
        @param initializer:
        @type initializer:
        @param weights:
        @type weights:
        @param bias:
        @type bias:
        """
        super().__init__(
            input_size, output_size, activation, initializer, weights, bias
        )
        self._j_ = declare(int)
        self._k_ = declare(int)
        self._index_ = declare(int)

    @property
    def initializer(self):
        return self._initializer

    @initializer.setter
    def initializer(self, init):
        self._initializer = init.__class__(shape=(self.output_size, self.input_size))

    def forward(self, input_var, output_var=None, save_to=None):
        """
        Propagate the input through the layer. Implements a matrix multiplication

        :param input_var: a Qua array containing the input to the layer
        :param output_var: a Qua array to contain the output of the layer
        :param save_to: a tag or stream to save the output to
        """

        with for_(self._k_, 0, self._k_ < self.output_size, self._k_ + 1):
            assign(self._z_[self._k_], 0)

            with for_(self._j_, 0, self._j_ < self.input_size, self._j_ + 1):
                assign(
                    self._z_[self._k_],
                    self._z_[self._k_]
                    + input_var[self._j_]
                    * self._weights_[self._k_ * self.input_size + self._j_],
                )

            assign(self._z_[self._k_], self._z_[self._k_] + self._bias_[self._k_])

            # apply activation
            self.activation.forward(self._z_[self._k_])
            assign(self._res_[self._k_], self.activation._res_)

            if output_var:
                assign(output_var[self._k_], self._res_[self._k_])

            if save_to:
                save(self._res_[self._k_], save_to)

    def backward(self, error, input_, learning_rate):
        with for_(self._j_, 0, self._j_ < self.input_size, self._j_ + 1):
            assign(self._error_[self._j_], 0)
            with for_(self._k_, 0, self._k_ < self.output_size, self._k_ + 1):
                assign(self._index_, self._k_ * self.input_size + self._j_)

                # calculate activation derivative
                self.activation.backward(self._z_[self._k_])

                # calculate the gradient using the error from next layer, activation and input
                assign(
                    self._gradient_[self._index_],
                    error[self._k_] * self.activation._res_ * input_[self._j_],
                )

                # update weights using the gradient
                assign(
                    self._weights_[self._index_],
                    self._weights_[self._index_]
                    - learning_rate * self._gradient_[self._index_],
                )

                # update bias
                assign(
                    self._bias_[self._k_],
                    self._bias_[self._k_]
                    - learning_rate * error[self._k_] * self.activation._res_,
                )

                # update error to pass backwards
                assign(
                    self._error_[self._j_],
                    self._error_[self._j_] + self._gradient_[self._index_],
                )

    def save_weights_(self, tag):
        with for_(
            self._index_, 0, self._index_ < self._weights_.length(), self._index_ + 1
        ):
            save(self._weights_[self._index_], self._weights_stream_)
        with for_(
            self._index_, 0, self._index_ < self._bias_.length(), self._index_ + 1
        ):
            save(self._bias_[self._index_], self._bias_stream_)

        with stream_processing():
            self._weights_stream_.buffer(self.output_size, self.input_size).save_all(
                tag + "weights"
            )
            self._bias_stream_.buffer(self.output_size).save_all(tag + "bias")


class Conv(Layer):
    def __init__(
        self,
        input_shape=None,
        kernel_shape=None,
        stride=(1, 1),
        padding="valid",
        activation=None,
        kernel_initializer=Uniform(),
        kernel_weights=None,
        kernel_bias=None,
    ):
        """

        @param input_shape:
        @type input_shape:
        @param kernel_shape:
        @type kernel_shape:
        @param stride:
        @type stride:
        @param padding:
        @type padding:
        @param activation:
        @type activation:
        @param kernel_initializer:
        @type kernel_initializer:
        @param kernel_weights:
        @type kernel_weights:
        @param kernel_bias:
        @type kernel_bias:
        """
        self.kernel_shape = kernel_shape
        self._input_size = int(input_shape[0] * input_shape[1])
        self.input_shape = input_shape
        self.padding = padding
        self.stride = stride
        self.output_shape = self._conv_output_shape()
        self._output_size = int(self.output_shape[0] * self.output_shape[1])
        super().__init__(
            input_size=self.input_size,
            output_size=self.output_size,
            activation=activation,
            initializer=kernel_initializer,
            weights=kernel_weights,
            bias=kernel_bias,
        )
        self._j_ = declare(int)
        self._k_ = declare(int)
        self._m_ = declare(int)
        self._n_ = declare(int)
        self._index_ = declare(int)

    def _conv_output_shape(self):
        if self.padding is None or self.padding == "valid":
            return np.ceil(
                (np.array(self.input_shape) - np.array(self.kernel_shape) + 1)
                / np.array(self.stride)
            ).flatten()
        if self.padding == "same":
            pass

        return 0

    @property
    def initializer(self):
        return self._initializer

    @initializer.setter
    def initializer(self, init):
        self._initializer = init.__class__(shape=self.kernel_shape)

    def _apply_padding(self, input_var):
        padded_input = None
        return padded_input

    def forward(self, input_var, output_var=None, save_to=None):
        padded_input = self._apply_padding(input_var)

        if self.padding == "valid":
            with for_(self._k_, 0, self._k_ < self.output_shape[0], self._k_ + 1):
                with for_(self._j_, 0, self._j_ < self.output_shape[1], self._j_ + 1):

                    assign(self._index_, self._k_ * self.output_shape[1] + self._j_)

                    assign(self._z_[self._index_], 0)

                    # calculate the convolution
                    with for_(
                        self._m_, 0, self._m_ < self.kernel_shape[0], self._m_ + 1
                    ):
                        with for_(
                            self._n_, 0, self._n_ < self.kernel_shape[1], self._n_ + 1
                        ):
                            assign(
                                self._z_[self._index_],
                                self._z_[self._index_]
                                + input_var[
                                    self._k_ * self.stride[0] * self.input_shape[1]
                                    + self._j_ * self.stride[1]
                                    + self._m_ * self.input_shape[1]
                                    + self._n_
                                ]
                                * self._weights_[
                                    self._m_ * self.kernel_shape[1] + self._n_
                                ]
                            )

                    # apply bias
                    assign(
                        self._z_[self._index_],
                        self._z_[self._index_] + self._bias_[self._index_],
                    )

                    # apply activation
                    self.activation.forward(self._z_[self._index_])
                    assign(self._res_[self._index_], self.activation._res_)

                    if output_var:
                        assign(output_var[self._index_], self._res_[self._index_])

                    if save_to:
                        save(self._res_[self._index_], save_to)

        if self.padding == "same":
            pass

    def backward(self, error, input_, learning_rate):
        pass

    def save_weights_(self, tag):
        with for_(
            self._index_, 0, self._index_ < self._weights_.length(), self._index_ + 1
        ):
            save(self._weights_[self._index_], self._weights_stream_)
        with for_(
            self._index_, 0, self._index_ < self._bias_.length(), self._index_ + 1
        ):
            save(self._bias_[self._index_], self._bias_stream_)

        with stream_processing():
            self._weights_stream_.buffer(*self.kernel_shape).save_all(tag + "weights")
            self._bias_stream_.buffer(*self.kernel_shape).save_all(tag + "bias")
