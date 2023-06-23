import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

from multilayer_perceptron.utils import read_dataset
from multilayer_perceptron.dense_layer import Dense
from multilayer_perceptron.sequential import Sequential
from multilayer_perceptron.callbacks import EarlyStopping, ModelCheckpoint
from multilayer_perceptron.optimizers import SGD, RMSprop, Adam
from multilayer_perceptron.losses import BinaryCrossentropy

file_name = "./data/data.csv"


def train_model(X, y, X_test=None, y_test=None, graph=False):
    print("Training model most suitable for the dataset")

    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], activation="relu"))
    model.add(Dense(256, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(128, activation="leaky_relu"))
    model.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(64, activation="tanh"))
    model.add(Dense(2, activation="softmax"))

    loss = BinaryCrossentropy(from_logits=True)

    model.compile(
        loss=loss,
        optimizer=SGD(learning_rate=0.0001, momentum=0.9, nesterov=True),
        metrics=["binary_accuracy", "precision"],
    )

    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=3)
    checkpoint = ModelCheckpoint("./data/model.json", monitor="val_loss", mode="min")

    if X_test is not None and y_test is not None:
        history = model.fit(
            X,
            y,
            batch_size=8,
            epochs=128,
            callbacks=[early_stopping, checkpoint],
            validation_data=(X_test, y_test),
        )
    else:
        history = model.fit(
            X,
            y,
            batch_size=8,
            epochs=128,
            callbacks=[early_stopping, checkpoint],
            validation_split=0.2,
        )

    if graph:
        metrics = history.history.keys()

        metrics = history.history.keys()
        num_rows = (len(metrics) + 1) // 2

        fig, axs = plt.subplots(num_rows, figsize=(15, 6 * num_rows))
        fig.tight_layout(pad=3.0)
        fig.subplots_adjust(top=0.95)
        fig.suptitle("Model metrics")

        # Plot training & validation accuracy values on the same plot
        for i, metric in enumerate(metrics):
            if metric.startswith("val_"):
                continue
            axs[i].plot(history.history[metric])
            axs[i].plot(history.history["val_" + metric])
            axs[i].set_ylabel(metric)
            axs[i].legend(["Train", "Validation"], loc="upper left")

        plt.show()


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
        metrics=["accuracy"],
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
        "dataset",
        type=str,
        help="Path to the dataset",
        metavar="dataset_path",
    )
    parser.add_argument(
        "-dt",
        "--datatest",
        help="Test the dataset",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        help="Optimizer benchmark",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-g",
        "--graph",
        help="Display graph",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    X, y = read_dataset(args.dataset)
    if args.datatest:
        X_test, y_test = read_dataset(args.datatest, test=True)

    if args.optimizer:
        test_all_optimizers(X, y)
    else:
        if args.datatest:
            train_model(X, y, X_test, y_test, graph=args.graph)
        else:
            train_model(X, y, graph=args.graph)
