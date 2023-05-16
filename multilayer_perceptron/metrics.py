import numpy as np


class Metric:
    def reset_state(self):
        raise NotImplementedError

    def result(self):
        raise NotImplementedError


class MeanMetric(Metric):
    def __init__(self):
        self.total = 0
        self.count = 0

    def update_state(self, value):
        self.total += value
        self.count += 1

    def reset_state(self):
        self.total = 0
        self.count = 0

    def result(self):
        print("Total: {}, Count: {}".format(self.total, self.count))
        return self.total / self.count


class Accuracy(MeanMetric):
    def __init__(self):
        super().__init__()

    def update_state(self, y_true, y_pred):
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        print("y_true: {}, y_pred: {}".format(y_true[:5], y_pred[:5]))
        super().update_state(np.sum(y_true == y_pred) / len(y_true))


class BinaryAccuracy(MeanMetric):
    def __init__(self):
        super().__init__()

    def update_state(self, y_true, y_pred):
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        super().update_state(np.sum(y_true == y_pred) / len(y_true))


METRICS = {
    "accuracy": Accuracy(),
    "binary_accuracy": BinaryAccuracy(),
}
