from multilayer_perceptron import dense_layer
from multilayer_perceptron import optimizers
from multilayer_perceptron.utils import check_arguments, History, shuffle_dataset
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

        self.history.append("loss", 0)
        for metric in self.metrics:
            self.history.append(metric.name, 0)

        self.compiled = True

    def fit(self, x=None, y=None, batch_size=None, epochs=1, callbacks=None):
        if not self.compiled:
            raise RuntimeError("Model must be compiled before fitting.")

        if self.loss is None:
            raise RuntimeError("Loss function must be specified for fitting.")

        if x.shape[1] != self.layers[0].input_dim:
            raise ValueError("Input dimension of first layer must match input data.")

        if x is None or y is None:
            raise ValueError("x and y must be specified for fitting.")

        if batch_size is None:
            batch_size = x.shape[0]

        @check_arguments(callbacks_module.CALLBACKS)
        def set_callbacks(self, argument):
            argument.set_model(self)
            self.callbacks.append(argument)

        if callbacks is not None:
            for callback in callbacks:
                set_callbacks(self, callback)

        for epoch in range(epochs):
            if self.stop_training:
                break
            total_loss = 0

            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)
            for metric in self.metrics:
                metric.reset_state()

            metrics_format = ""
            # browse self.history.history to get all metrics
            for history in self.history.history:
                metrics_format += "{}: {:.4f} - ".format(
                    history, self.history.history[history][-1]
                )

            print("Epoch {}/{}".format(epoch + 1, epochs))
            for i in tqdm(
                range(0, x.shape[0], batch_size),
                desc=metrics_format,
            ):
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

                for metric in self.metrics:
                    metric.update_state(y_batch, output)

                # compute loss

                total_loss += self.loss(y_batch, output)
                training_loss = total_loss / (i + batch_size)

            self.history.append("loss", training_loss)
            for metric in self.metrics:
                self.history.append(metric.name, metric.result())

            for callback in self.callbacks:
                callback.on_epoch_end(epoch, self.history.history)

            print()

        return self.history

    def evaluate(self, x=None, y=None, batch_size=None):
        if x is None or y is None:
            raise ValueError("x and y must be specified for evaluation.")

        if batch_size is None:
            batch_size = x.shape[0]

        scores = []
        scores.append(0)

        # TODO better gestion of batch_size

        for metric in self.metrics:
            metric.reset_state()

        print("Batch size: {}".format(batch_size))
        for i in range(0, x.shape[0], batch_size):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]

            output = x_batch
            for layer in self.layers:
                output = layer.forward(output)
            print("output {}".format(output[:5]))

            scores[0] += self.loss(y_batch, output)

            for metric in self.metrics:
                print("y_batch {}".format(y_batch[:5]))
                metric.update_state(y_batch, output)

        for metric in self.metrics:
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
        # save the network topology
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
