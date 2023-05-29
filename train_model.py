import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

from multilayer_perceptron.dense_layer import Dense
from multilayer_perceptron.sequential import Sequential
from multilayer_perceptron.callbacks import EarlyStopping
from multilayer_perceptron.optimizers import SGD, RMSprop, Adam
from multilayer_perceptron.initializers import HeUniform

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.callbacks import EarlyStopping

file_name = "./data/data.csv"


def one_hot_encoding(y):
    """One hot encoding for y"""

    n_classes = len(np.unique(y))
    classes = np.unique(y)
    one_hot = np.zeros((y.shape[0], n_classes))
    for i, c in enumerate(classes):
        idx = np.where(y == c)
        one_hot[idx, i] = 1

    return one_hot


def standardize(X):
    """Standardize the dataset X"""

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X


def train_model(X, y, graph):
    print("Training model most suitable for the dataset")

    model = Sequential()
    model.add(Dense(2, input_dim=X.shape[1], activation="sigmoid"))
    model.add(Dense(24, activation="sigmoid", kernel_initializer="he_uniform"))
    model.add(Dense(42, activation="sigmoid", kernel_initializer="he_uniform"))
    model.add(Dense(24, activation="sigmoid", kernel_initializer="he_uniform"))
    model.add(Dense(2, activation="softmax", kernel_initializer="he_uniform"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(),
        metrics=["accuracy", "binary_accuracy", "precision", "recall"],
    )

    early_stopping = EarlyStopping(monitor="loss", patience=5, mode="min")
    history = model.fit(X, y, batch_size=8, epochs=64, callbacks=[early_stopping])

    if graph:
        # TODO plot the graph
        pass

    model.save("./data/model.json")


def train_optimizer(X, y, optimizer):
    """Train the model with the given optimizer"""

    model = Sequential()
    model.add(Dense(2, input_dim=X.shape[1], activation="sigmoid"))
    model.add(Dense(24, activation="sigmoid"))
    model.add(Dense(42, activation="sigmoid"))
    model.add(Dense(24, activation="sigmoid"))
    model.add(Dense(2, activation="softmax"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy", "binary_accuracy", "precision", "recall"],
    )

    history = model.fit(X, y, batch_size=8, epochs=64)

    return history


def test_all_optimizers(X, y):
    """Test all optimizers"""

    optimizers = [
        SGD(learning_rate=0.01),
        SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
        RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-7),
        Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7),
    ]
    optimizer_name = [
        "SGD",
        "SGD nesterov momentum",
        "RMSprop",
        "Adam",
    ]

    histories = []
    for i in range(len(optimizers)):
        print("Training with {}".format(optimizer_name[i]))
        history = train_optimizer(X, y, optimizers[i])
        histories.append(history)
        print()

    plt.figure(figsize=(12, 8))
    for i, history in enumerate(histories):
        plt.plot(history.history["loss"], label=optimizer_name[i])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset", type=str, help="Path to the dataset", metavar="dataset_path"
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        help="Optimizer benchmark",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    with open(args.dataset, "r") as f:
        df = pd.read_csv(f, header=None)

    y = one_hot_encoding(df.values[:, 1])
    X = standardize(df.values[:, 2:].astype(float))

    if args.optimizer:
        test_all_optimizers(X, y)

    train_model(X, y, graph=True)
