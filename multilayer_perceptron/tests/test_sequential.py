import sys

sys.path.append("..")
import unittest
from dense_layer import DenseLayer
from sequential import Sequential


class TestSequential(unittest.TestCase):
    def test_add_layer(self):
        model = Sequential()
        dense_layer = DenseLayer(10, input_dim=5)
        model.add(dense_layer)
        self.assertEqual(len(model.layers), 1)
        self.assertIsInstance(model.layers[0], DenseLayer)

    def test_add_invalid_layer(self):
        model = Sequential()
        with self.assertRaises(TypeError):
            model.add("invalid_layer")

    def test_add_input_dim_error(self):
        model = Sequential()
        dense_layer1 = DenseLayer(n_neurons=10, input_dim=5)
        model.add(dense_layer1)
        dense_layer2 = DenseLayer(n_neurons=20, input_dim=10)
        with self.assertRaises(ValueError):
            model.add(dense_layer2)

    def test_compile(self):
        model = Sequential()
        model.add(DenseLayer(n_neurons=10, input_dim=5))
        model.add(DenseLayer(n_neurons=1000))
        model.compile(loss="mse", optimizer="sgd", metrics=["accuracy"])
        self.assertEqual(model.loss, "mse")
        self.assertEqual(model.optimizer, "sgd")
        self.assertEqual(model.metrics, ["accuracy"])

    def test_initialize_layers(self):
        model = Sequential()
        model.add(DenseLayer(n_neurons=10, input_dim=5))
        model.add(DenseLayer(n_neurons=1))
        model.compile(loss="mse", optimizer="sgd")
        self.assertEqual(model.layers[0].weights.shape, (5, 10))
        self.assertEqual(model.layers[0].bias.shape, (1, 10))
        self.assertEqual(model.layers[1].weights.shape, (10, 1))
        self.assertEqual(model.layers[1].bias.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
