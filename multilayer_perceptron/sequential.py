from multilayer_perceptron import dense_layer
from multilayer_perceptron import optimizers
from multilayer_perceptron.utils import check_arguments
from multilayer_perceptron import losses


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

        @check_arguments(losses.LOSSES)
        def set_loss(self, argument):
            self.loss = argument

        set_optimizer(self, optimizer)
        set_loss(self, loss)
        if metrics is not None:
            self.metrics = metrics
        # TODO METRICS

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
            current_loss = 0
            for i in range(0, x.shape[0], batch_size):
                x_batch = x[i : i + batch_size]
                y_batch = y[i : i + batch_size]

                output = x_batch
                for layer in self.layers:
                    output = layer.forward(output)

                print("outputs before loss derivate {}".format(output[:5]))
                grad = self.loss.derivative(output, y_batch)

                print("grad after loss derivate {}".format(grad[:5]))
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)

                for layer in self.layers:
                    layer.weights = self.optimizer.apply_gradients(
                        layer.dW, layer.weights
                    )
                    # layer.bias = self.optimizer.apply_gradients(layer.dB, layer.bias)

                current_loss += self.loss(output, y_batch)
                print(
                    "Epoch: {} Loss: {}".format(
                        epoch, current_loss / x.shape[0] * batch_size
                    ),
                    end="\r",
                )
            print()
