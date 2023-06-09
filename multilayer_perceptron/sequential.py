from multilayer_perceptron import dense_layer
from multilayer_perceptron import optimizers
from multilayer_perceptron.utils import (
    check_arguments,
    History,
    shuffle_dataset,
    split_dataset,
)
from multilayer_perceptron import losses
from multilayer_perceptron import metrics as metrics_module
from multilayer_perceptron import callbacks as callbacks_module


import numpy as np
from tqdm import tqdm


class Sequential:
    def __init__(self):
        self.layers = []
        self.metrics = []
        self.loss = None
        self.optimizer = None
        self.compiled = False
        self.stop_training = False
        self.history = History()
        self.callbacks = []

    def add(self, layer):
        if not isinstance(layer, dense_layer.Dense):
            raise TypeError("Layer must be a Dense object.")
        if len(self.layers) == 0 and layer.input_dim is None:
            raise ValueError("Input dimension must be specified for first layer.")
        elif len(self.layers) > 0 and layer.input_dim is not None:
            raise ValueError("Input dimension must not be specified after first layer.")
        self.layers.append(layer)

    def compile(self, optimizer="rmsprop", loss=None, metrics=None):
        @check_arguments(optimizers.OPTIMIZERS)
        def set_optimizer(self, argument):
            self.optimizer = argument

        @check_arguments(losses.LOSSES)
        def set_loss(self, argument):
            self.loss = argument

        @check_arguments(metrics_module.METRICS)
        def set_metrics(self, argument):
            self.metrics.append(argument)

        set_optimizer(self, optimizer)
        set_loss(self, loss)

        if metrics is not None:
            for metric in metrics:
                set_metrics(self, metric)

        for layer in range(len(self.layers)):
            if layer == 0:
                self.layers[layer].initialize(self.layers[layer].input_dim)
            else:
                self.layers[layer].initialize(self.layers[layer - 1].n_neurons)

        self.history.append("loss")
        for metric in self.metrics:
            self.history.append(metric.name)

        self.compiled = True

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
    ):
        if not self.compiled:
            raise RuntimeError("Model must be compiled before fitting.")

        if self.loss is None:
            raise RuntimeError("Loss function must be specified for fitting.")

        if x.shape[1] != self.layers[0].input_dim:
            raise ValueError("Input dimension of first layer must match input data.")

        if x is None or y is None:
            raise ValueError("x and y must be specified for fitting.")

        if validation_split < 0 or validation_split > 1:
            raise ValueError("Validation split must be between 0 and 1.")

        if batch_size is None:
            batch_size = x.shape[0]

        if validation_data is not None and validation_split > 0:
            raise ValueError(
                "validation_data and validation_split cannot be specified at the same time."
            )

        if validation_data is not None:
            if not isinstance(validation_data, tuple) or len(validation_data) != 2:
                raise TypeError(
                    "validation_data must be a tuple of numpy arrays (x_val, y_val)."
                )

        @check_arguments(callbacks_module.CALLBACKS)
        def set_callbacks(self, argument):
            argument.set_model(self)
            self.callbacks.append(argument)

        if callbacks is not None:
            for callback in callbacks:
                set_callbacks(self, callback)

        if validation_split > 0:
            x, y, x_val, y_val = split_dataset(x, y, ratio_train=(1 - validation_split))

        if validation_data is not None:
            x_val, y_val = validation_data

        self.save_metrics(x, y, batch_size)
        if validation_split > 0 or validation_data is not None:
            self.save_metrics(x_val, y_val, batch_size, validation=True)

        for epoch in range(epochs):
            if self.stop_training:
                break

            for callback in self.callbacks:
                callback.on_epoch_begin(epoch, self.history.history)

            metrics_format = ""
            for history in self.history.history:
                metrics_format += "{}: {:.4f} - ".format(
                    history,
                    0
                    if len(self.history.history[history]) == 0
                    else self.history.history[history][-1],
                )

            if shuffle:
                x, y = shuffle_dataset(x, y)

            print("Epoch {}/{}".format(epoch + 1, epochs))
            for i in tqdm(
                range(0, x.shape[0], batch_size),
                desc=metrics_format,
            ):
                for callback in self.callbacks:
                    callback.on_batch_begin(i, self.history.history)
                x_batch = x[i : i + batch_size]
                y_batch = y[i : i + batch_size]

                output = x_batch
                for layer in self.layers:
                    output = layer.forward(output)

                grad = self.loss.derivative(y_batch, output)

                for layer in reversed(self.layers):
                    grad = layer.backward(grad)

                for layer in self.layers:
                    layer.weights = self.optimizer.apply_gradients(
                        layer.dW, layer.weights
                    )
                    layer.bias = self.optimizer.apply_gradients(layer.dB, layer.bias)
                    self.optimizer.reset()

                self.save_metrics(x, y, batch_size)
                if validation_split > 0 or validation_data is not None:
                    self.save_metrics(x_val, y_val, batch_size, validation=True)

                for callback in self.callbacks:
                    callback.on_batch_end(epoch, self.history.history)

            for callback in self.callbacks:
                callback.on_epoch_end(epoch, self.history.history)
            print()

        return self.history

    def save_metrics(self, x=None, y=None, batch_size=None, validation=False):
        scores = self.evaluate(x, y, batch_size)
        if validation:
            self.history.append("val_loss", scores[0])
        else:
            self.history.append("loss", scores[0])
        for i in range(1, len(scores)):
            if validation:
                self.history.append("val_" + self.metrics[i - 1].name, scores[i])
            else:
                self.history.append(self.metrics[i - 1].name, scores[i])

    def evaluate(self, x=None, y=None, batch_size=None):
        if x is None or y is None:
            raise ValueError("x and y must be specified for evaluation.")

        if batch_size is None:
            batch_size = x.shape[0]

        scores = []

        predictions = self.predict(x, batch_size)
        scores.append(self.loss(y, predictions))
        for metric in self.metrics:
            metric.reset_state()
            metric.update_state(y, predictions)
            scores.append(metric.result())

        return scores

    def predict(self, x=None, batch_size=None):
        if x is None:
            raise ValueError("x must be specified for prediction.")

        if batch_size is None:
            batch_size = x.shape[0]

        predictions = []

        for i in range(0, x.shape[0], batch_size):
            x_batch = x[i : i + batch_size]

            output = x_batch
            for layer in self.layers:
                output = layer.forward(output)

            predictions.append(output)

        return np.concatenate(predictions)

    def save(self, path):
        import json

        topology = {
            "network_type": "multilayer_perceptron",
            "n_layers": len(self.layers),
            "optimizer": self.optimizer.name,
            "loss": self.loss.name,
        }

        layers_config = []
        for layer in self.layers:
            type_layer = "hidden"
            if layer.input_dim is not None:
                type_layer = "input"
            if layer == self.layers[-1]:
                type_layer = "output"
            layers_config.append(
                {
                    "type": type_layer,
                    "n_neurons": layer.n_neurons,
                    "activation": layer.activation.name,
                }
            )
            if layer.input_dim is not None:
                layers_config[-1]["input_dim"] = layer.input_dim

        topology["layers"] = layers_config

        weights = []
        biases = []
        for layer in self.layers:
            weights.append(layer.weights.tolist())
            biases.append(layer.bias.tolist())
        topology["weights"] = weights
        topology["biases"] = biases

        with open(path, "w") as f:
            json.dump(topology, f, indent=4)

    def load(self, path):
        import json

        with open(path, "r") as f:
            model_json = json.load(f)

        for layer in model_json["layers"]:
            if layer["type"] == "input":
                self.add(
                    dense_layer.Dense(
                        layer["n_neurons"],
                        input_dim=layer["input_dim"],
                        activation=layer["activation"],
                    )
                )
            else:
                self.add(
                    dense_layer.Dense(
                        layer["n_neurons"], activation=layer["activation"]
                    )
                )

        self.compile(
            optimizer=model_json["optimizer"],
            loss=model_json["loss"],
        )

        for i in range(len(self.layers)):
            w = np.array(model_json["weights"][i]).astype(float)
            b = np.array(model_json["biases"][i]).astype(float)
            self.layers[i].set_weights(w)
            self.layers[i].set_bias(b)

    def summary(self):
        print("Model: Sequential")
        print("Optimizer: {}".format(self.optimizer.name))
        print("Loss: {}".format(self.loss.name))
        print("Metrics: {}".format(", ".join([metric.name for metric in self.metrics])))
        print("Layers:")
        for layer in self.layers:
            print(
                "\t{}: {} neurons, {} activation".format(
                    layer.name, layer.n_neurons, layer.activation.name
                )
            )
