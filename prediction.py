import pandas as pd
import numpy as np
import json
import argparse

from multilayer_perceptron.sequential import Sequential
from multilayer_perceptron.dense_layer import Dense
from train_model import standardize, one_hot_encoding, read_dataset
from multilayer_perceptron.losses import BinaryCrossentropy


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

    X, y = read_dataset(args.dataset)

    # load the model
    model = load_model(model)

    # predictions on the dataset
    predictions = model.predict(X)
    # predictions = np.round(predictions)

    loss = BinaryCrossentropy(from_logits=True)
    x = loss(y, predictions)
    print("Binary cross entropy loss: {:.2f}".format(x))

    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y, axis=1)) / len(y)
    print("Accuracy: {:.2f}".format(accuracy))
