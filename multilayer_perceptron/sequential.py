from multilayer_perceptron import dense_layer
from multilayer_perceptron import optimizers
from multilayer_perceptron.utils import check_arguments


class Sequential:
    def __init__(self):
        self.layers = []
        self.metrics = []
        self.loss = None
        self.optimizer = None
        self.compiled = False

    def add(self, layer):
        if not isinstance(layer, dense_layer.DenseLayer):
            raise TypeError("Layer must be a DenseLayer object.")
        if len(self.layers) == 0 and layer.input_dim is None:
            raise ValueError("Input dimension must be specified for first layer.")
        elif len(self.layers) > 0 and layer.input_dim is not None:
            raise ValueError("Input dimension must not be specified after first layer.")
        self.layers.append(layer)

    def compile(self, optimizer="rmsprop", loss=None, metrics=None):
        @check_arguments(optimizers.OPTIMIZERS)
        def set_optimizer(self, argument):
            self.optimizer = argument

        set_optimizer(self, optimizer)
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

        self.compiled = True

    def fit(self, x=None, y=None, batch_size=None, epochs=1):
        if not self.compiled:
            raise RuntimeError("Model must be compiled before fitting.")

        if x is None or y is None:
            raise ValueError("x and y must be specified for fitting.")

        if batch_size is None:
            batch_size = x.shape[0]

        for epoch in range(epochs):
            current_loss = 0
            for i in range(0, x.shape[0], batch_size):
                x_batch = x[i : i + batch_size]
                y_batch = y[i : i + batch_size]

                # Forward pass
                output = x_batch
                for layer in self.layers:
                    output = layer.forward(output)

                # Backward pass
                grad = self.loss.derivative()(output, y_batch)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)

                # Update weights
                for layer in self.layers:
                    self.optimizer.apply_gradients(layer.weights, layer.dW)
                    self.optimizer.apply_gradients(layer.bias, layer.dB)

                current_loss += self.loss(output, y_batch)
