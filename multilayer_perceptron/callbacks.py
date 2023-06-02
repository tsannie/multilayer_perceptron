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
    def __init__(self, monitor="loss", patience=0, mode="min", start_from_epoch=0):
        super().__init__()
        if mode not in ["min", "max"]:
            raise ValueError("mode must be either min, or max.")
        self.mode = mode
        self.patience = patience
        self.monitor = monitor
        self.best_value = None
        self.wait = 0
        self.best_epoch = None
        self.start_from_epoch = start_from_epoch

    def on_epoch_end(self, epoch, logs=None):
        if self.monitor not in logs:
            print("WARNING: Early stopping requires {} available!".format(self.monitor))
            return

        if epoch < self.start_from_epoch:
            return
        current_value = logs.get(self.monitor)[-1]
        if self.best_value is None:
            self.best_value = current_value
        if (self.mode == "min" and current_value < self.best_value) or (
            self.mode == "max" and current_value > self.best_value
        ):
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True


class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor="val_loss", mode="min"):
        super().__init__()
        if mode not in ["min", "max"]:
            raise ValueError("mode must be either min, or max.")
        self.mode = mode
        self.monitor = monitor
        self.filepath = filepath
        self.best_value = None

    def on_epoch_end(self, epoch, logs=None):
        if self.monitor not in logs:
            print(
                "WARNING: Model checkpoint requires {} available!".format(self.monitor)
            )
            return

        current_value = logs.get(self.monitor)[-1]
        if self.best_value is None:
            self.best_value = current_value
        if (self.mode == "min" and current_value < self.best_value) or (
            self.mode == "max" and current_value > self.best_value
        ):
            self.best_value = current_value
            self.model.save(self.filepath)
            print("Saved model to {}".format(self.filepath))


CALLBACKS = {
    "early_stopping": EarlyStopping,
    "model_checkpoint": ModelCheckpoint,
}
