import numpy as np
from multilayer_perceptron import initializers
from multilayer_perceptron import activations
from multilayer_perceptron.utils import check_arguments
from multilayer_perceptron import regularizers


class Dense:
    def __init__(
        self,
        n_neurons,
        activation="linear",
        input_dim=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
    ):
        self.n_neurons = n_neurons
        self.weights = None
        self.bias = None
        self.z = None
        self.inputs = None
        self.input_dim = input_dim
        self.dW = None
        self.dB = None
        self.kernel_regularizer = None
        self.bias_regularizer = None

        @check_arguments(activations.ACTIVATION_FUNCTIONS)
        def init_activation(self, argument):
            return argument

        @check_arguments(initializers.WEIGHTS_INITIALIZERS)
        def init_weights(self, argument):
            return argument

        @check_arguments(regularizers.REGULARIZERS)
        def init_regularizer(self, argument):
            return argument

        self.activation = init_activation(self, activation)
        self.kernel_initializer = init_weights(self, kernel_initializer)
        self.bias_initializer = init_weights(self, bias_initializer)
        if kernel_regularizer is not None:
            self.kernel_regularizer = init_regularizer(self, kernel_regularizer)
        if bias_regularizer is not None:
            self.bias_regularizer = init_regularizer(self, bias_regularizer)

    def set_weights(self, weights):
        if weights.shape != self.weights.shape:
            raise ValueError("Weights shape must be equal to current weights shape.")
        self.weights = weights

    def set_bias(self, bias):
        if bias.shape != self.bias.shape:
            raise ValueError("Bias shape must be equal to current bias shape.")
        self.bias = bias

    def initialize(self, n_inputs):
        # (n_inputs, n_neurons)
        self.weights = self.kernel_initializer((n_inputs, self.n_neurons))

        # (1, n_neurons)
        self.bias = self.bias_initializer((1, self.n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias

        if self.kernel_regularizer is not None:
            self.z += self.kernel_regularizer(self.weights)

        if self.bias_regularizer is not None:
            self.z += self.bias_regularizer(self.bias)

        return self.activation(self.z)

    def backward(self, grad):
        grad = grad * self.activation.derivative(self.z)

        self.dW = np.dot(self.inputs.T, grad)
        if self.kernel_regularizer is not None:
            self.dW += self.kernel_regularizer.derivative(self.dW)

        self.dB = np.sum(grad, axis=0, keepdims=True)
        if self.bias_regularizer is not None:
            self.dB += self.bias_regularizer.derivative(self.dB)

        grad_inputs = np.dot(grad, self.weights.T)

        return grad_inputs
