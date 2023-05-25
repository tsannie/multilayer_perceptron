import numpy as np


class Metric:
    def __init__(self, name):
        self.name = name

    def reset_state(self):
        raise NotImplementedError

    def result(self):
        raise NotImplementedError


class Precision(Metric):
    def __init__(self):
        super().__init__("precision")
        self.true_positives = 0
        self.false_positives = 0

    def update_state(self, y_true, y_pred):
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
        self.true_positives += np.sum(y_true * y_pred)
        self.false_positives += np.sum((1 - y_true) * y_pred)

    def reset_state(self):
        self.true_positives = 0
        self.false_positives = 0

    def result(self):
        return self.true_positives / (self.true_positives + self.false_positives)


class Recall(Metric):
    def __init__(self):
        super().__init__("recall")
        self.true_positives = 0
        self.false_negatives = 0

    def update_state(self, y_true, y_pred):
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
        self.true_positives += np.sum(y_true * y_pred)
        self.false_negatives += np.sum(y_true * (1 - y_pred))

    def reset_state(self):
        self.true_positives = 0
        self.false_negatives = 0

    def result(self):
        return self.true_positives / (self.true_positives + self.false_negatives)


class MeanMetric(Metric):
    def __init__(self, name):
        super().__init__(name)
        self.total = 0
        self.count = 0

    def update_state(self, value):
        self.total += value
        self.count += 1

    def reset_state(self):
        self.total = 0
        self.count = 0

    def result(self):
        return self.total / self.count


class Accuracy(MeanMetric):
    def __init__(self):
        super().__init__("accuracy")

    def update_state(self, y_true, y_pred):
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        super().update_state(np.mean(y_true == y_pred))


class BinaryAccuracy(MeanMetric):
    def __init__(self):
        super().__init__("binary_accuracy")

    def update_state(self, y_true, y_pred):
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
        super().update_state(np.mean(y_true == y_pred))


METRICS = {
    "accuracy": Accuracy,
    "binary_accuracy": BinaryAccuracy,
}
