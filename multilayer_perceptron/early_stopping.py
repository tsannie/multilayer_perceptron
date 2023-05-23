import numpy as np


class Callback:
    def __init__(self):
        self.model = None

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass


class EarlyStopping(Callback):
    def __init__(self, patience=0, mode="auto"):
        super().__init__()
        self.patience = patience
        self.mode = mode
        self.best_epoch = 0
        self.best_value = None
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_value = self.model.history[-1][self.model.monitor]
        if self.best_value is None:
            self.best_value = current_value
        elif self.mode == "min" and current_value < self.best_value:
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0
        elif self.mode == "max" and current_value > self.best_value:
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
