from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple
from scipy.stats import norm


class MetaLearningDistribution(ABC):
    def __init__(self, d_x: int, batch_shape: Tuple):
        self.d_x = d_x
        self.batch_shape = batch_shape
        assert self.d_x > 0

    @abstractmethod
    def _sample(self, n_samples: int) -> np.ndarray:
        """
        Sample a batch of n_samples from the distribution
        return.shape == (n_samples,) + self.batch_dim + (self.d_x,)
        """
        pass

    @abstractmethod
    def _log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the log prob for a batch of samples x. x is guaranteed to have
        a single sample dimension.
        x.shape == (n_samples,) + self.batch_dim + (self.d_x)
        return.shape == (n_samples,) + self.batch_dim
        """
        pass

    def sample(self, n_samples: int) -> np.ndarray:
        # check input
        assert n_samples > 0

        # compute samples
        x = self._sample(n_samples)

        # check output
        assert x.shape == (n_samples,) + self.batch_shape + (self.d_x,)
        return x

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        # check input
        sample_shape = x.shape[: -(len(self.batch_shape) + 1)]
        assert len(sample_shape) in [0, 1]
        assert x.shape == sample_shape + self.batch_shape + (self.d_x,)

        ## compute log prob
        # add dummy sample dim
        if len(sample_shape) == 0:
            lp = self._log_prob(x[None]).squeeze(0)
        else:
            lp = self._log_prob(x)

        # check output
        assert lp.shape == sample_shape + self.batch_shape
        return lp


class MetaLearningFactorizedNormal(MetaLearningDistribution):
    def __init__(self, mu: np.ndarray, scale: np.ndarray):
        # check inputs
        batch_dims = mu.shape[:-1]
        d_x = mu.shape[-1]
        assert mu.shape == batch_dims + (d_x,)
        assert scale.shape == batch_dims + (d_x,)
        assert (scale > 0.0).all()

        # set fields
        super().__init__(d_x=d_x, batch_shape=batch_dims)
        self.mu = mu
        self.scale = scale

    def _sample(self, n_samples: int) -> np.ndarray:
        s = norm.rvs(
            loc=self.mu,
            scale=self.scale,
            size=(n_samples,) + self.batch_shape + (self.d_x,),
        )
        return s

    def _log_prob(self, x: np.ndarray) -> np.ndarray:
        return np.sum(norm.logpdf(x, loc=self.mu, scale=self.scale), axis=-1)
