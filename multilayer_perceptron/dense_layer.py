import numpy as np
from multilayer_perceptron import initializers
from multilayer_perceptron import activations
from multilayer_perceptron.utils import check_arguments


class Dense:
    def __init__(
        self,
        n_neurons,
        activation="linear",
        input_dim=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    ):
        self.n_neurons = n_neurons
        self.weights = None
        self.bias = None
        self.z = None
        self.inputs = None
        self.input_dim = input_dim
        self.dW = None
        self.dB = None

        @check_arguments(activations.ACTIVATION_FUNCTIONS)
        def set_activation(self, argument):
            self.activation = argument

        @check_arguments(initializers.WEIGHTS_INITIALIZERS)
        def set_kernel_initializer(self, argument):
            self.kernel_initializer = argument

        @check_arguments(initializers.WEIGHTS_INITIALIZERS)
        def set_bias_initializer(self, argument):
            self.bias_initializer = argument

        set_activation(self, activation)
        set_kernel_initializer(self, kernel_initializer)
        set_bias_initializer(self, bias_initializer)

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
        self.z = np.dot(self.inputs, self.weights) + self.bias
        return self.activation(self.z)

    def backward(self, grad):
        grad = grad * self.activation.derivative(self.z)
        self.dW = np.dot(self.inputs.T, grad)
        self.dB = np.sum(grad, axis=0, keepdims=True)
        grad_inputs = np.dot(grad, self.weights.T)
        return grad_inputs
