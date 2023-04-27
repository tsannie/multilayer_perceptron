import numpy as np
from initializers import *
from activations import ACTIVATION_FUNCTIONS
from initializers import WEIGHTS_INITIALIZERS


def check_arguments(valid_options):
    def decorator(func):
        def wrapper(self, argument):
            if isinstance(argument, str):
                if argument not in valid_options:
                    raise ValueError(
                        "Invalid argument. {} not in list of valid options.".format(
                            argument
                        )
                    )
                argument = valid_options[argument]
            elif not callable(argument):
                raise ValueError("Invalid argument. ")
            return func(self, argument)

        return wrapper

    return decorator


class DenseLayer:
    def __init__(self, n_neurons, activation, weights_initializer, bias_initializer):
        self.n_neurons = n_neurons
        self.weights = None
        self.bias = None

        @check_arguments(ACTIVATION_FUNCTIONS)
        def set_activation(self, argument):
            self.activation = argument

        @check_arguments(WEIGHTS_INITIALIZERS)
        def set_weights_initializer(self, argument):
            self.weights_initializer = argument

        @check_arguments(WEIGHTS_INITIALIZERS)
        def set_bias_initializer(self, argument):
            self.bias_initializer = argument

        set_activation(self, activation)
        set_weights_initializer(self, weights_initializer)
        set_bias_initializer(self, bias_initializer)

    def initialize(self, shape):
        self.weights = self.weights_initializer(shape)
        self.bias = self.bias_initializer((1, self.n_neurons))
