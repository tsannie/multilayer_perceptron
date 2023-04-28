import numpy as np
from dense_layer import DenseLayer


class Sequential:
    def __init__(self):
        self.layers = []
        self.metrics = []
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        if not isinstance(layer, DenseLayer):
            raise TypeError("Layer must be a DenseLayer object.")
        if len(self.layers) == 0 and layer.input_dim is None:
            raise ValueError("Input dimension must be specified for first layer.")
        elif len(self.layers) > 0 and layer.input_dim is not None:
            raise ValueError("Input dimension must not be specified after first layer.")
        self.layers.append(layer)

    def compile(self, loss, optimizer, metrics=None):
        self.loss = loss
        self.optimizer = optimizer
        if metrics is not None:
            self.metrics = metrics

        # Initialize all layers
        for layer in range(len(self.layers)):
            if layer == 0:
                self.layers[layer].initialize(self.layers[layer].input_dim)
            else:
                self.layers[layer].initialize(self.layers[layer - 1].n_neurons)
