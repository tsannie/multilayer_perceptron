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

    def append(self, key, value):
        if key not in self.history:
            self.history[key] = []
        self.history[key].append(value)

def shuffle_dataset(X, y):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]
