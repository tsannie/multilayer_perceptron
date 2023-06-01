import pandas as pd
import matplotlib.pyplot as plt
import math
import argparse

default_file_path = "../data/dataset.csv"


def histogram(df):
    header = ["diagnostic"]
    for i in range(1, len(df.columns)):
        header.append("param" + str(i))

    df = pd.DataFrame(df.values, columns=header)

    n_params = len(df.columns) - 1
    n_rows = math.ceil(math.sqrt(n_params))
    n_cols = math.ceil(n_params / n_rows)

    figure = plt.figure(figsize=(16, 12))
    figure.suptitle("Histograms of the {} params".format(n_params), fontsize=20)

    for i, param in enumerate(df.columns[1:]):
        ax = figure.add_subplot(n_rows, n_cols, i + 1)
        ax.set_title(param, fontsize=15)
        for diagnostic in df["diagnostic"].unique():
            ax.hist(
                df[df["diagnostic"] == diagnostic][param],
                bins=20,
                alpha=0.5,
                label=diagnostic,
            )

    plt.legend(loc="upper right")
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="Path to the csv file")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv_file, index_col=0)

        histogram(df)
    except FileNotFoundError:
        print("Invalid file")
