import sys

sys.path.append("..")
import numpy as np
import unittest
import activations as my_activations
from tensorflow.keras import activations as keras_activations
import tensorflow as tf


class TestActivations(unittest.TestCase):
    def setUp(self):
        self.inputs = []
        for _ in range(100):
            input_array = np.random.rand(10, 10)
            self.inputs.append(input_array)
        self.inputs.append(np.zeros((10, 10)))
        self.inputs.append(np.ones((10, 10)))
        self.inputs.append(np.full((10, 10), -1.0))
        self.inputs.append(np.full((10, 10), -100.0))
        self.inputs.append(np.full((10, 10), 100.0))

    def test_relu(self):
        keras_relu = keras_activations.relu
        my_relu = my_activations.ReLU()

        for x in self.inputs:
            keras_output = keras_relu(x)
            my_output = my_relu(x)

            np.testing.assert_array_almost_equal(keras_output, my_output)

    def test_sigmoid(self):
        keras_sigmoid = keras_activations.sigmoid
        my_sigmoid = my_activations.Sigmoid()

        for x in self.inputs:
            keras_output = keras_sigmoid(x)
            my_output = my_sigmoid(x)

            np.testing.assert_array_almost_equal(keras_output, my_output)

    def test_softmax(self):
        keras_softmax = keras_activations.softmax
        my_softmax = my_activations.Softmax()

        for x in self.inputs:
            scores = x.reshape(1, -1)
            tensor = tf.convert_to_tensor(scores)

            keras_output = keras_softmax(tensor)
            my_output = my_softmax(scores)
            np.testing.assert_array_almost_equal(keras_output, my_output)

    def test_softplus(self):
        keras_softplus = keras_activations.softplus
        my_softplus = my_activations.Softplus()

        for x in self.inputs:
            keras_output = keras_softplus(x)
            my_output = my_softplus(x)

            np.testing.assert_array_almost_equal(keras_output, my_output)

    def test_softsign(self):
        keras_softsign = keras_activations.softsign
        my_softsign = my_activations.Softsign()

        for x in self.inputs:
            keras_output = keras_softsign(x)
            my_output = my_softsign(x)

            np.testing.assert_array_almost_equal(keras_output, my_output)

    def test_tanh(self):
        keras_tanh = keras_activations.tanh
        my_tanh = my_activations.Tanh()

        for x in self.inputs:
            keras_output = keras_tanh(x)
            my_output = my_tanh(x)

            np.testing.assert_array_almost_equal(keras_output, my_output)

    def test_exponential(self):
        keras_exponential = keras_activations.exponential
        my_exponential = my_activations.Exponential()

        for x in self.inputs:
            keras_output = keras_exponential(x)
            my_output = my_exponential(x)

            np.testing.assert_array_almost_equal(keras_output, my_output)


if __name__ == "__main__":
    unittest.main()
