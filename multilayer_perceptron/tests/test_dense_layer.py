import sys

sys.path.append("..")
import unittest
from dense_layer import DenseLayer
import activations
import initializers
import numpy as np


class TestDenseLayer(unittest.TestCase):
    def setUp(self):
        self.activation = "relu"
        self.weights_initializer = initializers.GlorotNormal(seed=42)
        self.bias_initializer = initializers.Zeros()
        self.n_neurons = 10
        self.dense_layer = DenseLayer(
            n_neurons=self.n_neurons,
            activation=self.activation,
            weights_initializer=self.weights_initializer,
            bias_initializer=self.bias_initializer,
        )

    def test_init(self):
        self.assertEqual(self.dense_layer.activation, activations.ReLU)
        self.assertEqual(self.dense_layer.weights_initializer, self.weights_initializer)
        self.assertEqual(self.dense_layer.bias_initializer, self.bias_initializer)
        self.assertEqual(self.dense_layer.n_neurons, self.n_neurons)
        self.assertIsNone(self.dense_layer.weights)
        self.assertIsNone(self.dense_layer.bias)

    def test_init_with_str_or_callable_activation(self):
        self.dense_layer = DenseLayer(
            n_neurons=self.n_neurons,
            activation="relu",
            weights_initializer=self.weights_initializer,
            bias_initializer=self.bias_initializer,
        )
        self.assertEqual(self.dense_layer.activation, activations.ReLU)

        self.dense_layer = DenseLayer(
            n_neurons=self.n_neurons,
            activation=activations.ReLU(),
            weights_initializer=self.weights_initializer,
            bias_initializer=self.bias_initializer,
        )
        self.assertTrue(isinstance(self.dense_layer.activation, activations.ReLU))

    def test_init_with_str_or_callable_weights_initializer(self):
        self.dense_layer = DenseLayer(
            n_neurons=self.n_neurons,
            activation="relu",
            weights_initializer="glorot_normal",
            bias_initializer="zeros",
        )
        self.assertEqual(
            self.dense_layer.weights_initializer, initializers.GlorotNormal
        )
        self.assertEqual(self.dense_layer.bias_initializer, initializers.Zeros)

        self.assertEqual(self.dense_layer.activation, activations.ReLU)

        self.dense_layer = DenseLayer(
            n_neurons=self.n_neurons,
            activation=activations.ReLU(),
            weights_initializer=initializers.GlorotNormal(),
            bias_initializer=initializers.Zeros(),
        )
        self.assertTrue(
            isinstance(self.dense_layer.weights_initializer, initializers.GlorotNormal)
        )
        self.assertTrue(
            isinstance(self.dense_layer.bias_initializer, initializers.Zeros)
        )
        self.assertTrue(isinstance(self.dense_layer.activation, activations.ReLU))

    def test_init_with_invalid_arg(self):
        try:
            self.dense_layer = DenseLayer(
                n_neurons=self.n_neurons,
                activation="invalid",
                weights_initializer=self.weights_initializer,
                bias_initializer=self.bias_initializer,
            )
            self.fail("Should have raised a ValueError")
        except ValueError:
            pass

        try:
            self.dense_layer = DenseLayer(
                n_neurons=self.n_neurons,
                activation=42,
                weights_initializer=self.weights_initializer,
                bias_initializer=self.bias_initializer,
            )
            self.fail("Should have raised a ValueError")
        except ValueError:
            pass

    def test_initialize(self):
        n_inputs = 21
        self.dense_layer.initialize(n_inputs)

        self.assertEqual(self.dense_layer.weights.shape, (n_inputs, self.n_neurons))
        self.assertEqual(self.dense_layer.bias.shape, (1, self.n_neurons))

        self.assertTrue(np.all(self.dense_layer.weights >= -1))
        self.assertTrue(np.all(self.dense_layer.weights <= 1))
        self.assertTrue(np.all(self.dense_layer.bias == 0))

    def test_forward(self):
        n_inputs = 21
        m = 50
        self.dense_layer.initialize(n_inputs)

        inputs = np.random.randn(m, n_inputs)
        outputs = self.dense_layer.forward(inputs)

        self.assertEqual(outputs.shape, (50, self.n_neurons))

    def test_backward(self):
        n_inputs = 21
        m = 50
        self.dense_layer.initialize(n_inputs)

        inputs = np.random.randn(m, n_inputs)
        outputs = self.dense_layer.forward(inputs)

        grad_outputs = np.random.randn(*outputs.shape)
        grad_inputs = self.dense_layer.backward(grad_outputs, 0.1)

        self.assertEqual(grad_inputs.shape, inputs.shape)


if __name__ == "__main__":
    unittest.main()
