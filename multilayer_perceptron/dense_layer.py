import numpy as np
import initializers
import activations


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
    def __init__(
        self,
        n_neurons,
        activation=None,
        input_dim=None,
        weights_initializer=None,
        bias_initializer=None,
    ):
        self.n_neurons = n_neurons
        self.weights = None
        self.bias = None
        self.z = None
        self.inputs = None
        self.input_dim = input_dim

        @check_arguments(activations.ACTIVATION_FUNCTIONS)
        def set_activation(self, argument):
            self.activation = argument

        @check_arguments(initializers.WEIGHTS_INITIALIZERS)
        def set_weights_initializer(self, argument):
            self.weights_initializer = argument

        @check_arguments(initializers.WEIGHTS_INITIALIZERS)
        def set_bias_initializer(self, argument):
            self.bias_initializer = argument

        set_activation(self, activation)
        set_weights_initializer(self, weights_initializer)
        set_bias_initializer(self, bias_initializer)

    def initialize(self, n_inputs):
        if self.input_dim is not None:
            n_inputs = self.input_dim

        # (n_inputs, n_neurons)
        self.weights = self.weights_initializer((n_inputs, self.n_neurons))

        # (1, n_neurons)
        self.bias = self.bias_initializer((1, self.n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(self.inputs, self.weights) + self.bias
        return self.activation()(self.z)

    def backward(self, dA, learning_rate):
        m = self.inputs.shape[0]
        dA = dA * self.activation()(self.z, derivative=True)
        dW = 1 / m * np.dot(self.inputs.T, dA)
        db = 1 / m * np.sum(dA, axis=0, keepdims=True)
        dZ = np.dot(dA, self.weights.T)

        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db

        return dZ
