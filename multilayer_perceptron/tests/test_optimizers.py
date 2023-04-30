import sys

sys.path.append("..")

import unittest
import numpy as np
from tensorflow.keras import optimizers as keras_optimizers
import tensorflow as tf
import optimizers as my_optimizers


class TestOptimizers(unittest.TestCase):
    def setUp(self):
        self.model = tf.keras.Sequential(
            [tf.keras.layers.Dense(2, input_shape=(2,), activation="linear")]
        )
        self.x = np.array([[1, 1], [2, 2]])
        self.y = np.array([[0, 1], [1, 0]])
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.optimizer = my_optimizers.SGD(lr=0.1)
        self.keras_optimizer = keras_optimizers.SGD(learning_rate=0.1)

    def test_SGD(self):
        with tf.GradientTape() as tape:
            predictions = self.model(self.x)
            loss = self.loss_fn(self.y, predictions)
        grads = tape.gradient(loss, self.model.trainable_variables)

        keras_grads = [
            tf.convert_to_tensor(g) for g in grads
        ]  # Conversion en tenseurs TensorFlow
        keras_params = self.model.trainable_variables

        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )  # Utilisation de la classe optimizer personnalisée
        self.keras_optimizer.apply_gradients(
            zip(keras_grads, keras_params)
        )  # Utilisation de l'optimizer Keras

        # Vérification des mises à jour des poids du modèle
        for i, layer in enumerate(self.model.layers):
            np.testing.assert_allclose(layer.get_weights()[0], keras_params[i].numpy())
            np.testing.assert_allclose(layer.get_weights()[0], keras_params[i].numpy())


"""         my_sgd = my_optimizers.SGD(learning_rate=0.01)
        my_sgd(self.grads, self.params)

        for i in range(len(self.params)):
            np.testing.assert_array_almost_equal(
                keras_params[i].numpy(),
                self.params[i].numpy(),
                decimal=5,
            ) """


"""     def test_RMSprop(self):
        my_rmsprop = my_optimizers.RMSprop(
            learning_rate=self.learning_rate, rho=self.rho, epsilon=self.epsilon
        )
        keras_rmsprop = keras_optimizers.RMSprop(
            learning_rate=self.learning_rate, rho=self.rho, epsilon=self.epsilon
        )
        my_rmsprop(self.grads, self.params)
        keras_rmsprop.apply_gradients(zip(self.grads, self.params))
        for i in range(len(self.params)):
            np.testing.assert_allclose(self.params[i], keras_rmsprop.get_weights()[i])

    def test_Adam(self):
        my_adam = my_optimizers.Adam(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
        )
        keras_adam = keras_optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta1,
            beta_2=self.beta2,
            epsilon=self.epsilon,
        )
        my_adam(self.grads, self.params)
        keras_adam.apply_gradients(zip(self.grads, self.params))
        for i in range(len(self.params)):
            np.testing.assert_allclose(self.params[i], keras_adam.get_weights()[i]) """


if __name__ == "__main__":
    unittest.main()
