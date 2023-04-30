import numpy as np


class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.iterations = 0

    def apply_gradients(self, grads_and_vars):
        if not grads_and_vars:
            return None
        grads, variables = zip(*grads_and_vars)

        if len(grads) != len(variables):
            raise TypeError(
                "number of gradients is different from number of parameters"
            )

        agg_grads_var = self._aggregate_gradients(zip(grads, variables))
        self._apply_gradients(agg_grads_var)

        self.iterations += 1
        return self.iterations

    def variables(self):
        return []

    def _aggregate_gradients(self, grads_and_vars):
        raise NotImplementedError

    def _apply_gradients(self, grads_and_vars):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def _aggregate_gradients(self, grads_and_vars):
        return grads_and_vars

    def _apply_gradients(self, grads_and_vars):
        for g, v in grads_and_vars:
            v -= self.learning_rate * g


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-7):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.cache = {}

    def _aggregate_gradients(self, grads_and_vars):
        if self.iterations == 0:
            self.cache = [np.zeros_like(v) for _, v in grads_and_vars]
        agg_grads_var = []
        for i, (g, v) in enumerate(grads_and_vars):
            self.cache[i] = self.rho * self.cache[i] + (1 - self.rho) * g**2
            agg_grads_var.append(g / (np.sqrt(self.cache[i]) + self.epsilon), v)
        return agg_grads_var

    def _apply_gradients(self, grads_and_vars):
        for g, v in grads_and_vars:
            v -= self.learning_rate * g


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def _aggregate_gradients(self, grads_and_vars):
        if self.m is None:
            self.m = [np.zeros_like(v) for _, v in grads_and_vars]
        if self.v is None:
            self.v = [np.zeros_like(v) for _, v in grads_and_vars]

        self.t += 1
        learning_rate_t = (
            self.learning_rate
            * np.sqrt(1 - self.beta2**self.t)
            / (1 - self.beta1**self.t)
        )

        agg_grads_var = []
        for i, (g, v) in enumerate(grads_and_vars):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
            v -= learning_rate_t * self.m[i] / (np.sqrt(self.v[i]) + self.epsilon)
            agg_grads_var.append((g, v))
        return agg_grads_var

    def _apply_gradients(self, grads_and_vars):
        for g, v in grads_and_vars:
            v -= g
