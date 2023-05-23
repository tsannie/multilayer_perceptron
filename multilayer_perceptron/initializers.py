import numpy as np


class Initializer:
    """Base class for all initializers"""

    def __call__(self, shape):
        raise NotImplementedError

    def get_config(self):
        return {}


class RandomNormal(Initializer):
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


class RandomUniform(Initializer):
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


class TruncatedNormal(Initializer):
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


class Zeros(Initializer):
    """Zeros initializer"""

    def __call__(self, shape):
        return np.zeros(shape)


class Ones(Initializer):
    """Ones initializer"""

    def __call__(self, shape):
        return np.ones(shape)


class GlorotNormal(Initializer):
    """Glorot normal initializer (Xavier normal initializer))"""

    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, shape):
        fan_in, fan_out = shape
        stddev = np.sqrt(2 / (fan_in + fan_out))
        return self.rng.normal(loc=0.0, scale=stddev, size=shape)

    def get_config(self):
        return {
            "seed": self.seed,
        }


class GlorotUniform(Initializer):
    """Glorot uniform initializer (Xavier uniform initializer))"""

    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, shape):
        fan_in, fan_out = shape
        limit = np.sqrt(6 / (fan_in + fan_out))
        return self.rng.uniform(low=-limit, high=limit, size=shape)

    def get_config(self):
        return {
            "seed": self.seed,
        }


class HeNormal(Initializer):
    """He normal initializer"""

    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, shape):
        fan_in, _ = shape
        stddev = np.sqrt(2 / fan_in)
        return self.rng.normal(loc=0.0, scale=stddev, size=shape)

    def get_config(self):
        return {
            "seed": self.seed,
        }


class HeUniform(Initializer):
    """He uniform initializer"""

    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, shape):
        fan_in, _ = shape
        limit = np.sqrt(6 / fan_in)
        return self.rng.uniform(low=-limit, high=limit, size=shape)

    def get_config(self):
        return {
            "seed": self.seed,
        }


class Identity(Initializer):
    """Identity initializer"""

    def __call__(self, shape):
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(
                "Identity initializer can only be used for 2D square matrices"
            )
        return np.eye(shape[0])


WEIGHTS_INITIALIZERS = {
    "random_normal": RandomNormal,
    "random_uniform": RandomUniform,
    "truncated_normal": TruncatedNormal,
    "zeros": Zeros,
    "ones": Ones,
    "glorot_normal": GlorotNormal,
    "glorot_uniform": GlorotUniform,
    "he_normal": HeNormal,
    "he_uniform": HeUniform,
    "identity": Identity,
}
