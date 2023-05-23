import numpy as np


class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def apply_gradients(self, grads, variables):
        if len(grads) != len(variables):
            raise TypeError(
                "number of gradients is different from number of parameters"
            )

        grads, variables = self._aggregate_gradients(grads, variables)
        new_variables = self._apply_gradients(grads, variables)

        return new_variables

    def _aggregate_gradients(self, grads, variables):
        raise NotImplementedError

    def _apply_gradients(self, grads, variables):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity = None

    def _aggregate_gradients(self, grads, variables):
        return grads, variables

    def _apply_gradients(self, grads, variables):
        if self.velocity is None:
            self.velocity = [np.zeros_like(v) for v in variables]

        for i in range(len(grads)):
            self.velocity[i] = (
                self.momentum * self.velocity[i] - self.learning_rate * grads[i]
            )
            if self.nesterov:
                variables[i] += (
                    self.momentum * self.velocity[i] - self.learning_rate * grads[i]
                )
            else:
                variables[i] += self.velocity[i]

        return variables


""" class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-7):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.cache = {}

    def _aggregate_gradients(self, grads, variables):
        if self.iterations == 0:
            self.cache = [np.zeros_like(v) for _, v in grads, variables]
        agg_grads_var = []
        for i, (g, v) in enumerate(grads, variables):
            self.cache[i] = self.rho * self.cache[i] + (1 - self.rho) * g**2
            agg_grads_var.append(g / (np.sqrt(self.cache[i]) + self.epsilon), v)
        return agg_grads_var

    def _apply_gradients(self, grads, variables):
        for g, v in grads, variables:
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

    def _aggregate_gradients(self, grads, variables):
        if self.m is None:
            self.m = [np.zeros_like(v) for _, v in grads, variables]
        if self.v is None:
            self.v = [np.zeros_like(v) for _, v in grads, variables]

        self.t += 1
        learning_rate_t = (
            self.learning_rate
            * np.sqrt(1 - self.beta2**self.t)
            / (1 - self.beta1**self.t)
        )

        agg_grads_var = []
        for i, (g, v) in enumerate(grads, variables):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
            v -= learning_rate_t * self.m[i] / (np.sqrt(self.v[i]) + self.epsilon)
            agg_grads_var.append((g, v))
        return agg_grads_var

    def _apply_gradients(self, grads, variables):
        for g, v in grads, variables:
            v -= g """


OPTIMIZERS = {
    "sgd": SGD(),
}
