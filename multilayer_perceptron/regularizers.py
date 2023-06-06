import numpy as np


class Regularizer:
    def __init__(self, name):
        self.name = name

    def __call__(self, weights):
        return self.call(weights)

    def call(self, weights):
        raise NotImplementedError

    def derivative(self, weights):
        raise NotImplementedError


class L1(Regularizer):
    def __init__(self, l1=0.01):
        super().__init__("l1")
        self.l1 = l1

    def call(self, x):
        return self.l1 * np.sum(np.abs(x))

    def derivative(self, x):
        return self.l1 * np.sign(x)


class L2(Regularizer):
    def __init__(self, l2=0.01):
        super().__init__("l2")
        self.l2 = l2

    def call(self, x):
        return self.l2 * np.sum(np.square(x))

    def derivative(self, x):
        return self.l2 * 2 * x


class L1L2(Regularizer):
    def __init__(self, l1=0.01, l2=0.01):
        super().__init__("l1l2")
        self.l1 = l1
        self.l2 = l2

    def call(self, x):
        return self.l1 * np.sum(np.abs(x)) + self.l2 * np.sum(np.square(x))

    def derivative(self, x):
        return self.l1 * np.sign(x) + self.l2 * 2 * x


REGULARIZERS = {"l1": L1, "l2": L2, "l1l2": L1L2}
