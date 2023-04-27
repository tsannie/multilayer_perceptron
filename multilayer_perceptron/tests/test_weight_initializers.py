import sys

sys.path.append("..")

import unittest
import numpy as np
import initializers as my_initializers
from tensorflow.keras import initializers as keras_initializers


class TestInitializers(unittest.TestCase):
    def setUp(self):
        n = 1
        self.shape = []
        while n <= 1024:
            self.shape.append((n, n))
            n *= 2
        self.shape.append((1, 1024))
        self.shape.append((1024, 2048))

    def test_shape(self):
        keras_random_normal = keras_initializers.RandomNormal(seed=42)
        my_random_normal = my_initializers.RandomNormal(seed=42)

        for shape in self.shape:
            keras_output = keras_random_normal(shape)
            my_output = my_random_normal(shape)

            np.testing.assert_array_almost_equal(keras_output.shape, my_output.shape)

    def test_random_normal(self):
        keras_random_normal = keras_initializers.RandomNormal(
            seed=42, stddev=0.5, mean=15
        )
        my_random_normal = my_initializers.RandomNormal(seed=42, stddev=0.5, mean=15)

        for shape in self.shape:
            keras_output = keras_random_normal(shape)
            my_output = my_random_normal(shape)

            np.testing.assert_array_almost_equal(keras_output.shape, my_output.shape)

            self.assertAlmostEqual(np.mean(my_output), 15, places=0)
            if shape != (1, 1):
                self.assertAlmostEqual(np.std(my_output), 0.5, places=0)

    def test_random_uniform(self):
        my_random_uniform = my_initializers.RandomUniform(
            seed=42, minval=-21, maxval=21
        )

        for shape in self.shape:
            my_output = my_random_uniform(shape)

            self.assertTrue(np.all(my_output >= -21))
            self.assertTrue(np.all(my_output <= 21))

    def test_truncated_normal(self):
        mean = 21
        stddev = 10
        my_truncated_normal = my_initializers.TruncatedNormal(
            seed=42, mean=mean, stddev=stddev
        )

        for shape in self.shape:
            my_output = my_truncated_normal(shape)

            limit = 2 * stddev
            self.assertTrue(np.all(my_output >= mean - limit))
            self.assertTrue(np.all(my_output <= mean + limit))

    def test_zeros(self):
        my_zeros = my_initializers.Zeros()

        for shape in self.shape:
            my_output = my_zeros(shape)

            self.assertTrue(np.all(my_output == 0))

    def test_ones(self):
        my_ones = my_initializers.Ones()

        for shape in self.shape:
            my_output = my_ones(shape)

            self.assertTrue(np.all(my_output == 1))

    def test_glorot_normal(self):
        my_glorot_normal = my_initializers.GlorotNormal(seed=42)

        for shape in self.shape:
            my_output = my_glorot_normal(shape)

            fan_in, fan_out = shape
            std = np.sqrt(2 / (fan_in + fan_out))

            self.assertAlmostEqual(np.mean(my_output), 0, places=0)
            if shape != (1, 1):
                self.assertAlmostEqual(np.std(my_output), std, places=0)

    def test_glorot_uniform(self):
        my_glorot_uniform = my_initializers.GlorotUniform(seed=42)

        for shape in self.shape:
            my_output = my_glorot_uniform(shape)

            fan_in, fan_out = shape
            limit = np.sqrt(6 / (fan_in + fan_out))

            self.assertTrue(np.all(my_output >= -limit))
            self.assertTrue(np.all(my_output <= limit))

    def test_he_normal(self):
        my_he_normal = my_initializers.HeNormal(seed=42)

        for shape in self.shape:
            my_output = my_he_normal(shape)

            fan_in, _ = shape
            std = np.sqrt(2 / fan_in)

            self.assertAlmostEqual(np.mean(my_output), 0, places=0)
            if shape != (1, 1):
                self.assertAlmostEqual(np.std(my_output), std, places=0)

    def test_he_uniform(self):
        my_he_uniform = my_initializers.HeUniform(seed=42)

        for shape in self.shape:
            my_output = my_he_uniform(shape)

            fan_in, _ = shape
            limit = np.sqrt(6 / fan_in)

            self.assertTrue(np.all(my_output >= -limit))
            self.assertTrue(np.all(my_output <= limit))

    def test_identity(self):
        my_identity = my_initializers.Identity()

        for shape in self.shape:
            try:
                my_output = my_identity(shape)
                self.assertTrue(np.all(my_output == np.eye(shape[0])))
            except ValueError:
                if shape[0] == shape[1]:
                    self.fail("Identity initializer should work for square matrices")


if __name__ == "__main__":
    unittest.main()
