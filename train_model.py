import pandas as pd

file_name = "./data/data.csv"


class NeuralNetwork:
    def __init__(self, file_name):
        pass


if __name__ == "__main__":
    # prediction = NeuralNetwork(file_name)
    try:
        df = pd.read_csv(file_name, header=None)
    except FileNotFoundError:
        exit("File not found")

    df[1] = df[1].replace('M', 1)
    df[1] = df[1].replace('B', 0)
    df[1] = df[1].astype(int)

    X = df.values[:, 1].reshape(-1, 1)
    y = df.values[:, 2:]

    print(X.shape)
    print(y.shape)

