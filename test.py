import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


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


def binary_cross_entropy(y_true, y_pred):
    """Binary cross entropy loss function"""

    n = y_true.shape[0]
    epsilon = 1e-7

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -1 / n * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def train_model(X, y):
    print("Training model most suitable for the dataset")

    model = Sequential()
    model.add(Dense(2, input_dim=X.shape[1], activation="sigmoid"))
    model.add(Dense(24, activation="sigmoid", kernel_initializer="he_uniform"))
    model.add(Dense(42, activation="sigmoid", kernel_initializer="he_uniform"))
    model.add(Dense(24, activation="sigmoid", kernel_initializer="he_uniform"))
    model.add(Dense(2, activation="softmax", kernel_initializer="he_uniform"))

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy", "binary_accuracy"],
    )

    history = model.fit(X, y, batch_size=8, epochs=40, validation_split=0.2)

    with open("./data/data_test.csv", "r") as f:
        df_test = pd.read_csv(f, header=None)

    print("Testing model on test dataset")
    y_test = one_hot_encoding(df_test.values[:, 1])
    X_test = standardize(df_test.values[:, 2:].astype(float))

    predict = model.predict(X_test)
    predict = np.round(predict)

    score = binary_cross_entropy(y_test, predict)
    print("Binary cross entropy loss: ", score)

    print(
        "Accuracy: ",
        np.sum(np.argmax(predict, axis=1) == np.argmax(y_test, axis=1))
        / y_test.shape[0],
    )


if __name__ == "__main__":
    with open("./data/data.csv", "r") as f:
        df = pd.read_csv(f, header=None)

    y = one_hot_encoding(df.values[:, 1])
    X = standardize(df.values[:, 2:].astype(float))

    model = train_model(X, y)
