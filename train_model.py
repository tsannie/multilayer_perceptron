import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from multilayer_perceptron.dense_layer import DenseLayer
from multilayer_perceptron.sequential import Sequential

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

file_name = "./data/data.csv"


def standardize(X):
    """Standardize the dataset X"""

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X


if __name__ == "__main__":
    # prediction = NeuralNetwork(file_name)
    try:
        df = pd.read_csv(file_name, header=None)
    except FileNotFoundError:
        exit("File not found")

    df[1] = df[1].replace("M", 1)
    df[1] = df[1].replace("B", 0)
    df[1] = df[1].astype(int)

    y = df.values[:, 1].reshape(-1, 1)
    X = df.values[:, 2:]

    X = standardize(X)
    print(X)

    print(X.shape)
    print(y.shape)

    model = Sequential()
    model.add(DenseLayer(1, input_dim=X.shape[1], activation="sigmoid"))
    # model.add(DenseLayer(64, activation="relu"))
    # model.add(DenseLayer(128, activation="relu"))
    # model.add(DenseLayer(64, activation="relu"))
    # model.add(DenseLayer(1, activation="sigmoid"))

    """     model = Sequential()
    model.add(Dense(32, input_dim=X.shape[1], activation="relu")) """
    """     model.add(Dense(64, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid")) """
    model.compile(loss="binary_crossentropy", optimizer="sgd")
    # model.summary()
    model.fit(X, y, batch_size=1, epochs=100)
"""    history = model.fit(X, y, epochs=20, batch_size=10)

    df_test = pd.read_csv("./data/data_test.csv", header=None)
    df_test[1] = df_test[1].replace("M", 1)
    df_test[1] = df_test[1].replace("B", 0)
    df_test[1] = df_test[1].astype(int)

    y_test = df_test.values[:, 1].reshape(-1, 1)
    X_test = df_test.values[:, 2:]

    X_test = standardize(X_test)

    print(X_test.shape)
    print(y_test.shape)

    scores = model.evaluate(X_test, y_test)
    print("Test loss:", scores[0])
    print("Test accuracy:", scores[1])

    plt.plot(history.history["loss"])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.show()

    # Plot accuracy vs. epoch
    plt.plot(history.history["accuracy"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.show() """
