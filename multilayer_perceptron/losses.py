import numpy as np


class Loss:
    def __init__(self, name):
        self.name = name

    def __call__(self, y_true, y_pred):
        return self.call(y_true, y_pred)

    def call(self, y_true, y_pred):
        raise NotImplementedError

    def derivative(self, y_true, y_pred):
        raise NotImplementedError


# Probabilistic losses


class BinaryCrossentropy(Loss):
    def __init__(self, from_logits=False, epsilon=1e-7):
        super().__init__("binary_crossentropy")
        self.from_logits = from_logits
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        n = y_true.shape[0]
        if not self.from_logits:
            y_pred = 1 / (1 + np.exp(-y_pred))
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return (
            -1 / n * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        )

    def derivative(self, y_true, y_pred):
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))


# Regression losses


class MeanSquaredError(Loss):
    def __init__(self):
        super().__init__("mean_squared_error")

    def call(self, y_true, y_pred):
        return np.mean(np.sum(np.square(y_pred - y_true), axis=1))

    def derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true)


LOSSES = {
    "binary_crossentropy": BinaryCrossentropy,
    "mean_squared_error": MeanSquaredError,
}
