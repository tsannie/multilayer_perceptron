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
            self.assertAlmostEqual(np.std(my_output), 0.5, places=0)

    def test_random_uniform(self):
        keras_random_uniform = keras_initializers.RandomUniform(
            seed=42, minval=-21, maxval=21
        )
        my_random_uniform = my_initializers.RandomUniform(
            seed=42, minval=-21, maxval=21
        )

        for shape in self.shape:
            keras_output = keras_random_uniform(shape)
            my_output = my_random_uniform(shape)

            np.testing.assert_array_almost_equal(keras_output.shape, my_output.shape)

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


if __name__ == "__main__":
    unittest.main()
