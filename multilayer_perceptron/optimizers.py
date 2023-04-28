import numpy as np


class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def __call__(self, grads, params):
        raise NotImplementedError

    def get_config(self):
        return {"learning_rate": self.learning_rate}


class SGD(Optimizer):
    """Stochastic gradient descent optimizer"""

    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def __call__(self, grads, params):
        for i in range(len(params)):
            params[i] = params[i] - self.learning_rate * grads[i]


class RMSprop(Optimizer):
    """RMSprop optimizer"""

    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-7):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.cache = {}

    def __call__(self, grads, params):
        for i in range(len(params)):
            if params[i] not in self.cache:
                self.cache[params[i]] = np.zeros_like(params[i])
            self.cache[params[i]] = (
                self.rho * self.cache[params[i]] + (1 - self.rho) * grads[i] ** 2
            )
            params[i] = params[i] - self.learning_rate * grads[i] / (
                np.sqrt(self.cache[params[i]]) + self.epsilon
            )
        return params

    def get_config(self):
        return {
            "learning_rate": self.learning_rate,
            "rho": self.rho,
            "epsilon": self.epsilon,
        }


class Adam(Optimizer):
    """Adam optimizer"""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def __call__(self, grads, params):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1
        learning_rate_t = (
            self.learning_rate
            * np.sqrt(1 - self.beta2**self.t)
            / (1 - self.beta1**self.t)
        )

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2
            params[i] = params[i] - learning_rate_t * self.m[i] / (
                np.sqrt(self.v[i]) + self.epsilon
            )

    def get_config(self):
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
        }
