import sys

sys.path.append("..")
import numpy as np
import unittest

import metrics as my_metrics
import tensorflow.keras.metrics as keras_metrics


class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Test data
        self.y_true1 = np.array([0, 1, 1, 0, 1])
        self.y_pred1 = np.array([0, 1, 0, 1, 1])

        self.y_true2 = np.array([[0, 1], [0, 1], [1, 0], [0, 1], [0, 1]])
        self.y_pred2 = np.array([[0, 1], [1, 0], [0, 1], [1, 0], [1, 0]])

        self.y_true3 = np.array([[0, 1], [1, 0], [0, 1], [1, 0], [1, 0]])
        self.y_pred3 = np.array([[0, 1], [1, 0], [0, 1], [1, 0], [1, 0]])

    def test_accuracy(self):
        # TEST 1
        custom_acc = my_metrics.Accuracy()
        custom_acc.update_state(self.y_true1, self.y_pred1)
        custom_acc_result = custom_acc.result()

        keras_acc = keras_metrics.Accuracy()
        keras_acc.update_state(self.y_true1, self.y_pred1)
        keras_acc_result = keras_acc.result()

        self.assertEqual(custom_acc_result, keras_acc_result)

        custom_acc.reset_state()
        keras_acc.reset_state()

        # TEST 2
        custom_acc.update_state(self.y_true2, self.y_pred2)
        custom_acc_result = custom_acc.result()

        keras_acc.update_state(self.y_true2, self.y_pred2)
        keras_acc_result = keras_acc.result()

        self.assertEqual(custom_acc_result, keras_acc_result)

        # TEST 3
        custom_acc.reset_state()
        keras_acc.reset_state()

        custom_acc.update_state(self.y_true3, self.y_pred3)
        custom_acc_result = custom_acc.result()

        keras_acc.update_state(self.y_true3, self.y_pred3)
        keras_acc_result = keras_acc.result()

        self.assertEqual(custom_acc_result, keras_acc_result)

    def test_binary_accuracy(self):
        # TEST 1
        custom_acc = my_metrics.BinaryAccuracy()
        custom_acc.update_state(self.y_true1, self.y_pred1)
        custom_acc_result = custom_acc.result()

        keras_acc = keras_metrics.BinaryAccuracy()
        keras_acc.update_state(self.y_true1, self.y_pred1)
        keras_acc_result = keras_acc.result()

        self.assertEqual(custom_acc_result, keras_acc_result)

        custom_acc.reset_state()
        keras_acc.reset_state()

        # TEST 2
        custom_acc.update_state(self.y_true2, self.y_pred2)
        custom_acc_result = custom_acc.result()

        keras_acc.update_state(self.y_true2, self.y_pred2)
        keras_acc_result = keras_acc.result()

        self.assertEqual(custom_acc_result, keras_acc_result)

        # TEST 3
        custom_acc.reset_state()
        keras_acc.reset_state()

        custom_acc.update_state(self.y_true3, self.y_pred3)
        custom_acc_result = custom_acc.result()

        keras_acc.update_state(self.y_true3, self.y_pred3)
        keras_acc_result = keras_acc.result()

        self.assertEqual(custom_acc_result, keras_acc_result)


if __name__ == "__main__":
    unittest.main()
