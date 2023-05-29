import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

from multilayer_perceptron.utils import (
    one_hot_encoding,
    standardize,
    shuffle_dataset,
    split_dataset,
)
from multilayer_perceptron.dense_layer import Dense
from multilayer_perceptron.sequential import Sequential
from multilayer_perceptron.callbacks import EarlyStopping
from multilayer_perceptron.optimizers import SGD, RMSprop, Adam

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.callbacks import EarlyStopping

file_name = "./data/data.csv"


def train_model(X, y, graph=False):
    print("Training model most suitable for the dataset")

    model = Sequential()
    model.add(Dense(2, input_dim=X.shape[1], activation="sigmoid"))
    model.add(Dense(24, activation="sigmoid", kernel_initializer="he_uniform"))
    model.add(Dense(32, activation="tanh", kernel_initializer="he_uniform"))
    model.add(Dense(42, activation="sigmoid", kernel_initializer="he_uniform"))
    model.add(Dense(32, activation="tanh", kernel_initializer="he_uniform"))
    model.add(Dense(24, activation="sigmoid", kernel_initializer="he_uniform"))
    model.add(Dense(2, activation="softmax", kernel_initializer="he_uniform"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(),
        metrics=["binary_accuracy", "mse", "precision"],
    )

    early_stopping = EarlyStopping(monitor="loss", patience=3, mode="min")
    history = model.fit(
        X, y, batch_size=8, epochs=64, callbacks=[early_stopping], validation_split=0.2
    )

    if graph:
        metrics = history.history.keys()
        n_metrics = len(metrics)

        fig, axs = plt.subplots(n_metrics, 1, figsize=(8, 6 * n_metrics))
        fig.subplots_adjust(hspace=0.5)

        for i, metric_name in enumerate(metrics):
            ax = axs[i]
            ax.plot(history.history[metric_name])
            ax.set_title(metric_name)
            ax.set_xlabel("Epochs")
            ax.set_ylabel(metric_name)

        plt.show()

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

    history = model.fit(X, y, batch_size=8, epochs=42)

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
    parser.add_argument(
        "-g", "--graph", help="Display graph", default=False, action="store_true"
    )

    args = parser.parse_args()

    with open(args.dataset, "r") as f:
        df = pd.read_csv(f, header=None)

    y = one_hot_encoding(df.values[:, 1])
    X = standardize(df.values[:, 2:].astype(float))

    if args.optimizer:
        test_all_optimizers(X, y)
    else:
        train_model(X, y, args.graph)
