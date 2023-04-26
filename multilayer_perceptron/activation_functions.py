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


def softplus(x):
    """Softplus activation function. range [0, inf)"""
    return np.log(1 + np.exp(x))


def softsign(x):
    """Softsign activation function. range [-1, 1]"""
    return x / (1 + np.abs(x))


def tanh(x):
    """Tanh activation function. range [-1, 1]"""
    sinh = np.exp(x) - np.exp(-x)
    cosh = np.exp(x) + np.exp(-x)
    return sinh / cosh


def exponential(x):
    """Exponential activation function. range [0, inf)"""
    return np.exp(x)
