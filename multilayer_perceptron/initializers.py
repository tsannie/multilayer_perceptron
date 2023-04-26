import numpy as np


class RandomNormal:
    """Random normal distribution"""

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, shape):
        return self.rng.normal(loc=self.mean, scale=self.stddev, size=shape)

    def get_config(self):
        return {
            "mean": self.mean,
            "stddev": self.stddev,
            "seed": self.seed,
        }


class RandomUniform:
    """Random uniform distribution"""

    def __init__(self, minval=-0.05, maxval=0.05, seed=None):
        self.minval = minval
        self.maxval = maxval
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, shape):
        return self.rng.uniform(low=self.minval, high=self.maxval, size=shape)

    def get_config(self):
        return {
            "minval": self.minval,
            "maxval": self.maxval,
            "seed": self.seed,
        }


class TruncatedNormal:
    """Truncated normal distribution"""

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, shape):
        r = self.rng.normal(loc=self.mean, scale=self.stddev, size=shape)

        limit = 2 * self.stddev
        return np.clip(r, self.mean - limit, self.mean + limit)

    def get_config(self):
        return {
            "mean": self.mean,
            "stddev": self.stddev,
            "seed": self.seed,
        }
