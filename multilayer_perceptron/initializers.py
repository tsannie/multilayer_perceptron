import numpy as np


class WeightInitializer:
    """Base class for weight initializers"""

    def __init__(self, name, shape):
        self.name = name
        self.shape = shape

    def __call__(self, *args, **kwargs):
        return self.initialize(*args, **kwargs)

    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return self.name


class RandomNormal(WeightInitializer):
    """Random normal distribution"""

    def initialize(self, mean=0.0, stddev=0.05, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(mean, stddev, self.shape)


class RandomUniform(WeightInitializer):
    """Random uniform distribution"""

    def initialize(self, minval=-0.05, maxval=0.05, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(minval, maxval, self.shape)


class TruncatedNormal(WeightInitializer):
    """Truncated normal distribution"""

    def initialize(self, mean=0.0, stddev=0.05, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(mean, stddev, self.shape)


class Zeros(WeightInitializer):
    """Zeros"""

    def initialize(self):
        return np.zeros(self.shape)


class Ones(WeightInitializer):
    """Ones"""

    def initialize(self):
        return np.ones(self.shape)


class GlorotNormal(WeightInitializer):
    """Glorot normal distribution"""

    def initialize(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        fan_in, fan_out = self.shape
        stddev = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0.0, stddev, self.shape)


class GlorotUniform(WeightInitializer):
    """Glorot uniform distribution"""

    def initialize(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        fan_in, fan_out = self.shape
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, self.shape)


class HeNormal(WeightInitializer):
    """He normal distribution"""

    def initialize(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        fan_in, _ = self.shape
        stddev = np.sqrt(2 / fan_in)
        return np.random.normal(0.0, stddev, self.shape)


class HeUniform(WeightInitializer):
    """He uniform distribution"""

    def initialize(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        fan_in, _ = self.shape
        limit = np.sqrt(6 / fan_in)
        return np.random.uniform(-limit, limit, self.shape)
