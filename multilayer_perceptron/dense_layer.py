import numpy as np
from activation_functions import *
from initializers import *


class DenseLayer:
    def __init__(self, n_neurons, activation, weights_initializer, bias_initializer):
        ACTIVATION_FUNCTIONS = {
            "relu": ReLU(),
            "sigmoid": Sigmoid(),
            "softmax": Softmax(),
            "softplus": SoftPlus(),
            "softsign": SoftSign(),
            "tanh": Tanh(),
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
        self.bias_initializer = bias_initializer
        self.n_neurons = n_neurons
