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
        self.layers.append(layer)

    def compile(self, loss, optimizer, metrics=None):
        self.loss = loss
        self.optimizer = optimizer
        if metrics is not None:
            self.metrics = metrics

        # Initialize all layers
        for i in range(1, len(self.layers)):
            self.layers[i].initialize((self.layers[i - 1].n_neurons,))
