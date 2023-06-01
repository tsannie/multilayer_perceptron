import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import numpy as np

default_file_path = "../data/data.csv"


def pair_plot(df):
    scatter_kws = {"s": 5, "alpha": 0.5}
    diag_kws = {"bins": 20, "alpha": 0.5}

    header = ["diagnostic"]
    for i in range(1, len(df.columns)):
        header.append("param" + str(i))

    df = pd.DataFrame(df.values, columns=header)

    pairplot = sns.pairplot(
        df,
        hue="diagnostic",
        diag_kind="hist",
        plot_kws=scatter_kws,
        diag_kws=diag_kws,
        height=2,
    )

    for i, col in enumerate(pairplot.axes):
        for j, axes in enumerate(col):
            param1 = axes.get_xlabel()
            param2 = axes.get_ylabel()

            if i == 0:
                axes.set_title(
                    param1[:20] + "..." if len(param1) > 20 else param1, rotation=20
                )
            if j == 0:
                axes.set_ylabel(
                    param2[:20] + "..." if len(param2) > 20 else param2,
                    ha="right",
                    rotation=0,
                )
            axes.set_xticks([])
            axes.set_yticks([])

    plt.subplots_adjust(
        top=0.90, bottom=0, left=0.1, right=0.93, hspace=0.3, wspace=0.3
    )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_file", type=str, default=default_file_path, help="csv file path"
    )
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv_file, index_col=0)

        pair_plot(df)
    except FileNotFoundError:
        print("Invalid file")
