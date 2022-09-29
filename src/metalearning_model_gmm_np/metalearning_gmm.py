import numpy as np
import tensorflow as tf
from experiment_util.metalearning_distributions import MetaLearningDistribution
from gmm_util.gmm import GMM


def get_diagonal(a: np.ndarray):
    """
    Extract the diagonals of the batch of matrices in a.
    The batch dimensions are defined to be the first n-2 dimensions where n = a.ndim.
    """
    # check that matrices are square
    d_x = a.shape[-1]
    assert d_x > 0
    assert a.shape[-2] == d_x

    # make last two dimensions the first two
    a_transposed = np.transpose(a, np.roll(np.arange(a.ndim), shift=2))
    # extract diagonal
    a_diag = a_transposed.diagonal()
    return a_diag


def has_positive_diagonal(a: np.ndarray):
    """
    Check that the diagonals of the batch of matrices in a are all positive.
    The batch dimensions are defined to be the first n-2 dimensions where n = a.ndim.
    """
    # check that matrices are square
    d_x = a.shape[-1]
    assert d_x > 0
    assert a.shape[-2] == d_x

    a_diag = get_diagonal(a)
    # check positivity
    return (a_diag > 0).all()


def is_lower_triangular(a: np.ndarray):
    """
    Check that the batch of matrices in a are all lower triangular.
    The batch dimensions are defined to be the first n-2 dimensions where n = a.ndim.
    """
    # check that matrices are square
    d_x = a.shape[-1]
    assert d_x > 0
    assert a.shape[-2] == d_x

    return (a - np.tril(a) == 0).all()


class RunEagerlyContextManager:
    """
    Context manager that runs code in tensorflow's eager mode and resets the eager
    setting afterwards to what it was before.
    """

    def __enter__(self):
        self.eager_setting = tf.config.functions_run_eagerly()
        tf.config.run_functions_eagerly(True)

    def __exit__(self, exc_type, exc_vale, exc_tb):
        tf.config.run_functions_eagerly(self.eager_setting)


class MetaLearningGMM(MetaLearningDistribution):
    def __init__(
        self,
        log_w: np.ndarray,
        mu: np.ndarray,
        scale_tril: np.ndarray,
    ):
        # check inputs
        batch_shape = log_w.shape[:-1]
        n_comp = log_w.shape[-1]
        d_x = mu.shape[-1]
        assert log_w.shape == batch_shape + (n_comp,)
        assert mu.shape == batch_shape + (n_comp, d_x)
        assert scale_tril.shape == batch_shape + (n_comp, d_x, d_x)
        # check that diagonals of scale_tril are positive
        assert has_positive_diagonal(scale_tril)
        # check that scale_tril are lower triangular matrices
        assert is_lower_triangular(scale_tril)
        super().__init__(d_x=d_x, batch_shape=batch_shape)

        ## generate GMM
        with RunEagerlyContextManager():
            self._gmm = GMM(
                log_w=tf.convert_to_tensor(log_w, dtype=tf.float32),
                loc=tf.convert_to_tensor(mu, dtype=tf.float32),
                scale_tril=tf.convert_to_tensor(scale_tril, dtype=tf.float32),
            )

    @property
    def n_components(self):
        return self._gmm.n_components

    @property
    def log_w(self):
        return self._gmm.log_w.numpy()

    @property
    def mu(self):
        return self._gmm.loc.numpy()

    @property
    def scale_tril(self):
        return self._gmm.scale_tril.numpy()

    @log_w.setter
    def log_w(self, value):
        raise NotImplementedError

    @mu.setter
    def loc(self, value):
        raise NotImplementedError

    @scale_tril.setter
    def scale_tril(self, value):
        raise NotImplementedError

    def _sample(self, n_samples: int) -> np.ndarray:
        with RunEagerlyContextManager():
            return self._gmm.sample(n_samples).numpy()

    def _log_prob(self, x: np.ndarray) -> np.ndarray:
        with RunEagerlyContextManager():
            return self._gmm.log_density(x)[0].numpy()


class MetaLearningMultivariateNormal(MetaLearningGMM):
    def __init__(self, mu: np.ndarray, scale_tril: np.ndarray):
        # check inputs
        batch_dims = mu.shape[:-1]
        d_x = mu.shape[-1]
        assert mu.shape == batch_dims + (d_x,)
        assert scale_tril.shape == batch_dims + (d_x, d_x)
        super().__init__(
            log_w=np.zeros(batch_dims + (1,)),
            mu=mu[..., None, :],
            scale_tril=scale_tril[..., None, :, :],
        )

    @property
    def mu(self):
        return super().mu[..., 0, :]  # mean of zeroth component

    @property
    def scale_tril(self):
        return super().scale_tril[..., 0, :, :]  # scale_tril of zeroth component
