from multilayer_perceptron import dense_layer
from multilayer_perceptron import optimizers
from multilayer_perceptron.utils import check_arguments
from multilayer_perceptron import losses
from multilayer_perceptron import metrics as metrics_module

import numpy as np


class Sequential:
    def __init__(self):
        self.layers = []
        self.metrics = []
        self.loss = None
        self.optimizer = None
        self.compiled = False

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

        for metric in metrics:
            set_metrics(self, metric)

        # Initialize all layers
        for layer in range(len(self.layers)):
            if layer == 0:
                self.layers[layer].initialize(self.layers[layer].input_dim)
            else:
                self.layers[layer].initialize(self.layers[layer - 1].n_neurons)

        self.compiled = True

    def fit(self, x=None, y=None, batch_size=None, epochs=1):
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

        print("Batch size: {}".format(batch_size))

        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, x.shape[0], batch_size):
                x_batch = x[i : i + batch_size]
                y_batch = y[i : i + batch_size]

                # print("x_batch shape {}".format(x_batch.shape))
                output = x_batch
                for layer in self.layers:
                    output = layer.forward(output)

                # print("LAYER 0 WEIGHTS", self.layers[0].weights)
                # print("LAYER 0 SHAPE", self.layers[0].weights.shape)
                # print("output", output)
                grad = self.loss.derivative(y_batch, output)
                # print("grad after loss derivate {}".format(grad[:5]))

                for layer in reversed(self.layers):
                    grad = layer.backward(grad)

                for layer in self.layers:
                    layer.weights = self.optimizer.apply_gradients(
                        layer.dW, layer.weights
                    )
                    layer.bias = self.optimizer.apply_gradients(layer.dB, layer.bias)

                # compute loss
                total_loss += self.loss(y_batch, output)
                training_loss = total_loss / (i + 1)
                print(
                    "Loss: {:.4f} - Epoch: {}/{} - Batch: {}/{}".format(
                        training_loss,
                        epoch + 1,
                        epochs,
                        i + batch_size,
                        x.shape[0],
                    ),
                    end="\r",
                )

            print()

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

            scores[0] += self.loss(y_batch, output)

            for metric in self.metrics:
                print("y_batch {}".format(y_batch[:5]))
                metric.update_state(y_batch, output)

        for metric in self.metrics:
            scores.append(metric.result())

        return scores
