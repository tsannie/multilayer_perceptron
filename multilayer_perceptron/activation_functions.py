import numpy as np


def relu(x):
    """ReLU activation function. range [0, inf)"""
    return np.maximum(x, 0)


def sigmoid(x):
    """Sigmoid activation function. range [0, 1]"""
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1):
    """Softmax activation function. range [0, 1]"""
    reduce_sum = np.sum(np.exp(x), axis=axis, keepdims=True)
    return np.exp(x) / reduce_sum
