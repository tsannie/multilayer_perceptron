import numpy as np


class Optimizer:
    def __init__(self, learning_rate=0.01, name=None):
        self.learning_rate = learning_rate
        self.name = name

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

    def reset(self):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False):
        super().__init__(learning_rate, "sgd")
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = None

    def _aggregate_gradients(self, grads, variables):
        return grads, variables

    def _apply_gradients(self, grads, variables):
        for i in range(len(grads)):
            if self.momentum > 0:
                if self.velocities is None:
                    self.velocities = [np.zeros_like(v) for v in variables]
                self.velocities[i] = (
                    self.momentum * self.velocities[i] - self.learning_rate * grads[i]
                )
                if self.nesterov:
                    variables[i] += (
                        self.momentum * self.velocities[i]
                        - self.learning_rate * grads[i]
                    )
                else:
                    variables[i] += self.velocities[i]
            else:
                variables[i] -= self.learning_rate * grads[i]
        return variables

    def reset(self):
        self.velocities = None


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-7):
        super().__init__(learning_rate, "rmsprop")
        self.rho = rho
        self.epsilon = epsilon
        self.cache = None

    def _aggregate_gradients(self, grads, variables):
        if not self.cache:
            self.cache = [np.zeros_like(v) for v in variables]
        for i in range(len(grads)):
            self.cache[i] = self.rho * self.cache[i] + (1 - self.rho) * grads[i] ** 2
            grads[i] /= np.sqrt(self.cache[i] + self.epsilon)
        return grads, variables

    def _apply_gradients(self, grads, variables):
        for i in range(len(grads)):
            variables[i] -= self.learning_rate * grads[i]
        return variables

    def reset(self):
        self.cache = None


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        super().__init__(learning_rate, "adam")
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.momentums = None
        self.velocities = None
        self.timestep = 0

    def _aggregate_gradients(self, grads, variables):
        if self.momentums is None:
            self.momentums = [np.zeros_like(v) for v in variables]
            self.velocities = [np.zeros_like(v) for v in variables]

        for i in range(len(grads)):
            self.momentums[i] = (
                self.beta1 * self.momentums[i] + (1 - self.beta1) * grads[i]
            )
            self.velocities[i] = (
                self.beta2 * self.velocities[i] + (1 - self.beta2) * grads[i] ** 2
            )

        return grads, variables

    def _apply_gradients(self, grads, variables):
        self.timestep += 1
        learning_rate = (
            self.learning_rate
            * np.sqrt(1 - self.beta2**self.timestep)
            / (1 - self.beta1**self.timestep)
        )

        for i in range(len(grads)):
            variables[i] -= (
                learning_rate
                * self.momentums[i]
                / (np.sqrt(self.velocities[i]) + self.epsilon)
            )

        return variables

    def reset(self):
        self.momentums = None
        self.velocities = None
        self.timestep = 0


OPTIMIZERS = {
    "sgd": SGD,
    "rmsprop": RMSprop,
    "adam": Adam,
}
