import numpy as np


def check_arguments(valid_options):
    def decorator(func):
        def wrapper(self, argument):
            if isinstance(argument, str):
                if argument not in valid_options:
                    raise ValueError(
                        "Invalid argument. {} not in list of valid options.".format(
                            argument
                        )
                    )
                argument = valid_options[argument]
                argument = argument()
            else:
                is_valid = False
                for _, v in valid_options.items():
                    if isinstance(argument, v):
                        is_valid = True
                        break
                if not is_valid:
                    raise TypeError("Invalid argument type.")

            return func(self, argument)

        return wrapper

    return decorator


class History:
    def __init__(self):
        self.history = {}

    def append(self, key, value=None):
        if key not in self.history:
            self.history[key] = []
        if value is not None:
            self.history[key].append(value)


def shuffle_dataset(X, y):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]


def split_dataset(X, y, ratio_train=0.8):
    n_samples = X.shape[0]
    n_train = int(n_samples * ratio_train)

    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    return X_train, y_train, X_test, y_test


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
