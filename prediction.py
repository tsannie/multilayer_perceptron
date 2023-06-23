import pandas as pd
import numpy as np
import json
import argparse

from multilayer_perceptron.sequential import Sequential
from multilayer_perceptron.dense_layer import Dense
from multilayer_perceptron.utils import read_dataset
from multilayer_perceptron.losses import BinaryCrossentropy


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

    X, y = read_dataset(args.dataset, test=True)

    # load the model
    model = Sequential()

    model.load(args.model)

    # predictions on the dataset
    predictions = model.predict(X)

    loss = BinaryCrossentropy(from_logits=True)
    x = loss(y, predictions)
    print("Binary cross entropy loss: {:.4f}".format(x))

    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y, axis=1)) / len(y)
    print("Accuracy: {:.2f}".format(accuracy))
