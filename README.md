# multilayer_perceptron 🧮

![image](https://i.imgur.com/18CCG1y.png)

## 📝 Description

Welcome to my project of a MLP (Multilayer Perceptron) library developed from scratch, similar to [Keras](https://keras.io/api/). This library allows you to create, train, and evaluate multilayer perceptron models.

For this project, I am using predictive data to diagnose cancer in cells, with M representing malignant and B representing benign

## 📦 Features

### Sequential model

The Sequential model is a linear stack of layers.

```python
    Sequential()
```

**Methods:**

- _add_

  Add a layer to the model.

```python
    def add(self, layer):
```

- _compile_

  Configures the model for training.

```python
    def compile(self, optimizer="rmsprop", loss=None, metrics=None):
```

- _fit_

  Trains the model for a fixed number of epochs (iterations on a dataset).

```python
    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
    ):
```

- _evaluate_

  Returns the loss value & metrics values for the model.

```python
    def evaluate(self, x=None, y=None, batch_size=None):
```

- _predict_

  Generates output predictions for the input samples.

```python
    def predict(self, x=None, batch_size=None):
```

- _save_

  Saves the model to a json file.

```python
    def save(self, path):
```

- _load_

  Loads the model from a json file.

```python
    def load(self, path):
```

- _summary_

  Prints a summary of the model.

```python
    def summary(self):
```

### Layers

Layers are the basic building blocks of neural networks.

- [x] Dense

```python
    Dense(
        n_neurons,
        activation="linear",
        input_dim=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
    )
```

### Activation Functions

Activation functions are mathematical equations that determine the output of a neural network. The function is attached to each neuron in the network, and determines whether it should be activated or not, based on whether each neuron's input is relevant for the model's prediction.

- [x] Linear
- [x] ReLU
- [x] Leaky ReLU
- [x] Sigmoid
- [x] Softmax
- [x] Softplus
- [x] Softsign
- [x] Tanh
- [x] Exponential

### Initializers

Initializers define the way to set the initial random weights of neurons in a layer.

- [x] Random Normal
- [x] Random Uniform
- [x] Truncated Normal
- [x] Zeros
- [x] Ones
- [x] Glorot Normal
- [x] Glorot Uniform
- [x] He Normal
- [x] He Uniform
- [x] Identity

### Optimizers

Optimizers play a crucial role by helping to minimize the loss function and find the optimal values for the model's parameters. They guide the learning process by adjusting the parameters iteratively based on the gradients, leading to improved accuracy and faster convergence during training.

Test for all optimizers with "random" structure of neural network:

```bash
python train_model.py -o data.csv
```

![image](https://i.imgur.com/CyaNEkp.png)

- [x] Adam
- [x] Stochastic Gradient Descent
- [x] Stochastic Gradient Descent with Nesterov Momentum
- [x] RMSprop

### Loss Functions

Loss function measures the discrepancy between the predicted output of a model and the true output.

- [x] Binary Cross Entropy
- [x] Mean Squared Error

### Callbacks

Callbacks are functions that can be applied at certain stages of the training process, such as at the end of each epoch.

- [x] Early Stopping

  The early stopping callback is used to stop the training process if the model stops improving on the validation data.

```python
    EarlyStopping(monitor="loss", min_delta=0, patience=0, mode="min", start_from_epoch=0)
```

- [x] Model Checkpoint

  The model checkpoint callback is used to save the model after every epoch.

```python
    ModelCheckpoint(filepath, monitor="val_loss", mode="min"):
```

### Metrics

Metrics are used to evaluate the performance of your model.

Here an image of some metrics on a model trained:

![image](https://i.imgur.com/97CIGzk.png)

- [x] Accuracy
- [x] Binary Accuracy
- [x] Precision
- [x] Recall
- [x] Mean Squared Error

## 📌 TODO

- [ ] End regularization implementation

## 👨‍💻 Author

- [@tsannie](https://github.com/tsannie)

```

```
