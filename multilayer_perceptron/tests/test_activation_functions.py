import sys

sys.path.append("..")
import numpy as np
import unittest
import activation_functions as my_activations
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
        my_relu = my_activations.relu

        for x in self.inputs:
            keras_output = keras_relu(x)
            my_output = my_relu(x)

            np.testing.assert_array_almost_equal(keras_output, my_output)

    def test_sigmoid(self):
        keras_sigmoid = keras_activations.sigmoid
        my_sigmoid = my_activations.sigmoid

        for x in self.inputs:
            keras_output = keras_sigmoid(x)
            my_output = my_sigmoid(x)

            np.testing.assert_array_almost_equal(keras_output, my_output)

    def test_softmax(self):
        keras_softmax = keras_activations.softmax
        my_softmax = my_activations.softmax

        for x in self.inputs:
            scores = x.reshape(1, -1)
            tensor = tf.convert_to_tensor(scores)

            keras_output = keras_softmax(tensor)
            my_output = my_softmax(scores)
            print(keras_output, " == ", my_output)
            np.testing.assert_array_almost_equal(keras_output, my_output)


if __name__ == "__main__":
    unittest.main()
