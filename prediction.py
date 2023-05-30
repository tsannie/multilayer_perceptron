import pandas as pd
import numpy as np
import json
import argparse

from multilayer_perceptron.sequential import Sequential
from multilayer_perceptron.dense_layer import Dense
from train_model import standardize, one_hot_encoding


def binary_cross_entropy(y_true, y_pred):
    """Binary cross entropy loss function"""

    n = y_true.shape[0]
    epsilon = 1e-7

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -1 / n * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def load_model(model_json):
    """Load the model from the json file"""

    model = Sequential()

    for layer in model_json["layers"]:
        if layer["type"] == "input":
            model.add(
                Dense(
                    layer["n_neurons"],
                    input_dim=layer["input_dim"],
                    activation=layer["activation"],
                )
            )
        else:
            model.add(Dense(layer["n_neurons"], activation=layer["activation"]))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    for i in range(len(model.layers)):
        w = np.array(model_json["weights"][i]).astype(float)
        b = np.array(model_json["biases"][i]).astype(float)
        model.layers[i].set_weights(w)
        model.layers[i].set_bias(b)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", type=str, help="Path to the model", metavar="model_path"
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to the dataset to predict",
        metavar="dataset_path",
    )

    args = parser.parse_args()

    with open(args.model, "r") as f:
        model = json.load(f)

    print("Loading dataset")
    with open(args.dataset, "r") as f:
        df = pd.read_csv(f, header=None)

    X = standardize(df.values[:, 2:].astype(float))
    y = one_hot_encoding(df.values[:, 1])

    # load the model
    model = load_model(model)

    # predictions on the dataset
    predictions = model.predict(X)
    predictions = np.round(predictions)
    score = binary_cross_entropy(y, predictions)
    print("Binary cross entropy loss: ", score)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y, axis=1)) / len(y)
    print("Accuracy: ", accuracy)
