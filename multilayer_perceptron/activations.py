import numpy as np


class Activation:
    def __call__(self, x, derivative=False):
        if derivative:
            return self.derivative(x)
        else:
            return self.activate(x)

    def activate(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class ReLU(Activation):
    """ReLU activation activate. range [0, inf)"""

    def activate(self, x):
        return np.maximum(x, 0)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)


class Sigmoid(Activation):
    """Sigmoid activation activate. range [0, 1]"""

    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.activate(x) * (1 - self.activate(x))


class Softmax(Activation):
    """Softmax activation activate. range [0, 1]"""

    def activate(self, x):
        reduce_sum = np.sum(np.exp(x), axis=-1, keepdims=True)
        return np.exp(x) / reduce_sum

    def derivative(self, x):
        return self.activate(x) * (1 - self.activate(x))


class Softplus(Activation):
    """Softplus activation activate. range [0, inf)"""

    def activate(self, x):
        return np.log(1 + np.exp(x))

    def derivative(self, x):
        return 1 / (1 + np.exp(-x))


class Softsign(Activation):
    """Softsign activation activate. range [-1, 1]"""

    def activate(self, x):
        return x / (1 + np.abs(x))

    def derivative(self, x):
        return 1 / (1 + np.abs(x)) ** 2


class Tanh(Activation):
    """Tanh activation activate. range [-1, 1]"""

    def activate(self, x):
        sinh = np.exp(x) - np.exp(-x)
        cosh = np.exp(x) + np.exp(-x)
        return sinh / cosh

    def derivative(self, x):
        return 1 - self.activate(x) ** 2


class Exponential(Activation):
    """Exponential activation activate. range [0, inf)"""

    def activate(self, x):
        return np.exp(x)

    def derivative(self, x):
        return self.activate(x)


ACTIVATION_FUNCTIONS = {
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "softmax": Softmax,
    "softplus": Softplus,
    "softsign": Softsign,
    "tanh": Tanh,
    "exponential": Exponential,
}
