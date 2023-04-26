import numpy as np
from initializers import *
import activations


class DenseLayer:
    def __init__(self, n_neurons, activation, weights_initializer, bias_initializer):
        ACTIVATION_FUNCTIONS = {
            "relu": activations.relu,
            "sigmoid": activations.sigmoid,
            "softmax": activations.softmax,
            "softplus": activations.softplus,
            "softsign": activations.softsign,
            "tanh": activations.tanh,
            "exponential": activations.exponential,
        }
        WEIGHTS_INITIALIZERS = {
            "random_normal": RandomNormal(),
            "random_uniform": RandomUniform(),
            "truncated_normal": TruncatedNormal(),
            "zeros": Zeros(),
            "ones": Ones(),
            "glorot_normal": GlorotNormal(),
            "glorot_uniform": GlorotUniform(),
            "he_normal": HeNormal(),
            "he_uniform": HeUniform(),
        }

        self.activation = ACTIVATION_FUNCTIONS[activation]
        self.weights_initializer = WEIGHTS_INITIALIZERS[weights_initializer]
        self.bias_initializer = WEIGHTS_INITIALIZERS[bias_initializer]
        self.n_neurons = n_neurons
        self.weights = None
        self.bias = None

    def initialize(self, shape):
        # self.weights = self.weights_initializer.initialize(shape)
        # self.bias = self.bias_initializer.initialize(shape)
        pass
