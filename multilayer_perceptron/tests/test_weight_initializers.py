import sys

sys.path.append("..")

import unittest
import numpy as np
from initializers import *


class TestRandomNormal(unittest.TestCase):
    def test_initialize(self):
        initializer = RandomNormal("random_normal", (3, 3))
        output = initializer.initialize()
        self.assertEqual(output.shape, (3, 3))


class TestRandomUniform(unittest.TestCase):
    def test_initialize(self):
        initializer = RandomUniform("random_uniform", (3, 3))
        output = initializer.initialize(minval=-0.1, maxval=0.1, seed=42)
        self.assertEqual(output.shape, (3, 3))
        self.assertLessEqual(output.min(), 0.1)
        self.assertGreaterEqual(output.max(), -0.1)


class TestTruncatedNormal(unittest.TestCase):
    def test_initialize(self):
        initializer = TruncatedNormal("truncated_normal", (3, 3))
        output = initializer.initialize(mean=0.0, stddev=0.05, seed=42)
        self.assertEqual(output.shape, (3, 3))


class TestZeros(unittest.TestCase):
    def test_initialize(self):
        initializer = Zeros("zeros", (3, 3))
        output = initializer.initialize()
        self.assertEqual(output.shape, (3, 3))
        self.assertEqual(output.sum(), 0)


class TestOnes(unittest.TestCase):
    def test_initialize(self):
        initializer = Ones("ones", (3, 3))
        output = initializer.initialize()
        self.assertEqual(output.shape, (3, 3))
        self.assertEqual(output.sum(), 9)


class TestGlorotNormal(unittest.TestCase):
    def test_initialize(self):
        initializer = GlorotNormal("glorot_normal", (3, 3))
        output = initializer.initialize(seed=42)
        self.assertEqual(output.shape, (3, 3))


class TestGlorotUniform(unittest.TestCase):
    def test_initialize(self):
        initializer = GlorotUniform("glorot_uniform", (3, 3))
        output = initializer.initialize(seed=42)
        self.assertEqual(output.shape, (3, 3))


class TestHeNormal(unittest.TestCase):
    def test_initialize(self):
        initializer = HeNormal("he_normal", (3, 3))
        output = initializer.initialize(seed=42)
        self.assertEqual(output.shape, (3, 3))


class TestHeUniform(unittest.TestCase):
    def test_initialize(self):
        initializer = HeUniform("he_uniform", (3, 3))
        output = initializer.initialize(seed=42)
        self.assertEqual(output.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
