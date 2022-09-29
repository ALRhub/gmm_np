import os
from typing import Iterable, Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gmm_util.gmm import GMM

from metalearning_model_gmm_np.decoder import decoder_builder
from metalearning_model_gmm_np.util import assert_shape


class NP:
    """
    p(D^t | theta) = \int p(D^t | z, theta) p(z | theta) dz
    """

    def __init__(
        self,
        d_x: int,
        d_y: int,
        d_z: int,
        gmm_prior_scale: float,
        gmm_prior_n_components: int,
        gmm_posterior_n_components: int,
        decoder_n_hidden: int,
        decoder_d_hidden: int,
        decoder_activation: str,
        decoder_std_y_features: str,
        decoder_std_y_lower_bound: Optional[float] = None,
        decoder_fixed_std_y_value: Optional[Iterable[float]] = None,
    ):
        ## constants
        self.d_x = d_x
        self.d_y = d_y
        self.d_z = d_z
        self.prior_init_scale = gmm_prior_scale
        self.prior_n_comps = gmm_prior_n_components
        self.post_n_comps = gmm_posterior_n_components

        ## variables
        # will be set in self.reset_posteriors()
        self._n_tasks_total = tf.Variable(shape=None, initial_value=-1, dtype=tf.int32)
        self._gmm_post_log_w = tf.Variable(
            shape=[None, self.post_n_comps],
            initial_value=tf.ones((0, self.post_n_comps)),
            dtype=tf.float32,
        )
        self._gmm_post_loc = tf.Variable(
            shape=[None, self.post_n_comps, self.d_z],
            initial_value=tf.ones((0, self.post_n_comps, self.d_z)),
            dtype=tf.float32,
        )
        self._gmm_post_prec = tf.Variable(
            shape=[None, self.post_n_comps, self.d_z, self.d_z],
            initial_value=tf.ones((0, self.post_n_comps, self.d_z, self.d_z)),
            dtype=tf.float32,
        )
        # will be set in self.reset_batch()
        self._task_ids_batch = tf.Variable(
            shape=[None], initial_value=tf.ones((0,), dtype=tf.int32), dtype=tf.int32
        )

        ## model and approximate posterior
        self.decoder = decoder_builder(
            d_x=self.d_x,
            d_y=self.d_y,
            d_z=self.d_z,
            n_hidden=decoder_n_hidden,
            d_hidden=decoder_d_hidden,
            activation=decoder_activation,
            std_y_features=decoder_std_y_features,
            std_y_lower_bound=decoder_std_y_lower_bound,
            fixed_std_y=decoder_fixed_std_y_value,
        )
        prior_w, prior_loc, prior_cov = create_initial_gmm_parameters(
            n_tasks=1,
            d_z=self.d_z,
            n_components=self.prior_n_comps,
            prior_scale=self.prior_init_scale,
        )
        self.prior = GMM(
            log_w=tf.math.log(prior_w),
            loc=prior_loc,
            prec=tf.linalg.inv(prior_cov),  
        )
        # will be set in self.reset_batch()
        self.post_batch = None

        # trackables (for saving/loading)
        self._trackables = {
            "gmm_post_log_w": self._gmm_post_log_w,
            "gmm_post_loc": self._gmm_post_loc,
            "gmm_post_prec": self._gmm_post_prec,
            "decoder": self.decoder.trackables,
        }

    @property
    def n_tasks_total(self):
        return self._n_tasks_total

    @n_tasks_total.setter
    def n_tasks_total(self, value):
        self._n_tasks_total.assign(value)

    @property
    def batch_task_ids(self):
        return self._task_ids_batch

    @batch_task_ids.setter
    def batch_task_ids(self, value: tf.Tensor):
        assert tf.rank(value) == 1
        assert tf.reduce_max(value) < self._n_tasks_total
        assert tf.reduce_min(value) >= 0
        self._task_ids_batch.assign(value)

    @property
    def post_log_w(self):
        return self._gmm_post_log_w

    @post_log_w.setter
    def post_log_w(self, value):
        self._gmm_post_log_w.assign(value)

    @property
    def post_loc(self):
        return self._gmm_post_loc

    @post_loc.setter
    def post_loc(self, value):
        self._gmm_post_loc.assign(value)

    @property
    def post_prec(self):
        return self._gmm_post_prec

    @post_prec.setter
    def post_prec(self, value):
        self._gmm_post_prec.assign(value)

    @property
    def n_tasks_batch(self):
        return tf.shape(self.batch_task_ids)[0]

    @property
    def trackables(self) -> dict:
        return self._trackables

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        ]
    )
    def _predict(self, x: tf.Tensor, z: tf.Tensor):
        # check input
        n_tasks = tf.shape(x)[0]  # does not have to be equal to self.n_tasks_batch
        n_points = tf.shape(x)[1]
        n_samples = tf.shape(z)[0]
        assert_shape(x, (n_tasks, n_points, self.d_x))
        assert_shape(z, (n_samples, n_tasks, self.d_z))

        mu, var = self.decoder(x=x, z=z)

        # check output
        assert_shape(mu, (n_samples, n_tasks, n_points, self.d_y))
        assert_shape(var, (n_samples, n_tasks, n_points, self.d_y))
        return mu, var

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None], dtype=tf.bool),
        ]
    )
    def _log_likelihood(
        self, x: tf.Tensor, y: tf.Tensor, z: tf.Tensor, ctx_mask: tf.Tensor
    ):
        """
        log p(D^t | z, theta)
        """
        # check input
        n_tasks = tf.shape(x)[0]  # does not have to be equal to self.n_tasks_batch
        n_points = tf.shape(x)[1]
        n_samples = tf.shape(z)[0]
        assert_shape(x, (n_tasks, n_points, self.d_x))
        assert_shape(y, (n_tasks, n_points, self.d_y))
        assert_shape(z, (n_samples, n_tasks, self.d_z))
        assert_shape(ctx_mask, (n_tasks, n_points))

        # compute log likelihood
        mu, var = self._predict(x=x, z=z)
        assert_shape(mu, (n_samples, n_tasks, n_points, self.d_y))
        assert_shape(var, (n_samples, n_tasks, n_points, self.d_y))
        gaussian = tfp.distributions.Independent(
            tfp.distributions.Normal(loc=mu, scale=tf.sqrt(var)),
            reinterpreted_batch_ndims=1,  # sum ll of data dim upon calling log_prob
        )
        log_likelihood = gaussian.log_prob(y[None, ...])
        assert_shape(log_likelihood, (n_samples, n_tasks, n_points))

        # apply mask
        log_likelihood = tf.math.multiply(
            log_likelihood, tf.cast(ctx_mask[None], tf.float32)
        )
        log_likelihood = tf.reduce_sum(log_likelihood, axis=-1)

        # check output
        assert_shape(log_likelihood, (n_samples, n_tasks))
        return log_likelihood

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        ]
    )
    def _log_prior_density(self, z: tf.Tensor):
        """
        log p(z | theta)
        """
        # check input
        n_tasks = tf.shape(z)[1]  # does not have to be equal to self.n_tasks_batch
        n_samples = tf.shape(z)[0]
        assert_shape(z, (n_samples, n_tasks, self.d_z))

        # compute log conditional prior density
        flat_z = tf.reshape(z, (n_samples * n_tasks, 1, self.d_z))
        log_prior_density, _, _ = self.prior.log_density(flat_z)
        log_prior_density = tf.reshape(log_prior_density, (n_samples, n_tasks))

        # check output
        assert_shape(log_prior_density, (n_samples, n_tasks))
        return log_prior_density

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None], dtype=tf.bool),
        ]
    )
    def _log_unnormalized_posterior_density(
        self, x: tf.Tensor, y: tf.Tensor, z: tf.Tensor, ctx_mask: tf.Tensor
    ):
        """
        log p(D^t | z, theta) + log p(z | theta)
        """
        # check input
        n_tasks = tf.shape(x)[0]
        n_points = tf.shape(x)[1]
        n_samples = tf.shape(z)[0]
        assert_shape(x, (n_tasks, n_points, self.d_x))
        assert_shape(y, (n_tasks, n_points, self.d_y))
        assert_shape(z, (n_samples, n_tasks, self.d_z))
        assert_shape(ctx_mask, (n_tasks, n_points))

        # compute log_density
        log_likelihood = self._log_likelihood(x=x, y=y, z=z, ctx_mask=ctx_mask)
        assert_shape(log_likelihood, (n_samples, n_tasks))
        log_prior_density = self._log_prior_density(z=z)
        assert_shape(log_prior_density, (n_samples, n_tasks))
        log_density = log_likelihood + log_prior_density

        # check output
        assert_shape(log_density, (n_samples, n_tasks))
        return log_density

    # @tf.function(
    #     input_signature=[
    #         tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    #     ]
    # )
    def _log_approximate_posterior_density(self, z: tf.Tensor):
        # check input
        n_samples = tf.shape(z)[0]
        assert_shape(z, (n_samples, self.n_tasks_batch, self.d_z))

        log_density, _, _ = self.post_batch.log_density(
            z=z, compute_grad=False, compute_hess=False
        )

        # check output
        assert_shape(log_density, (n_samples, self.n_tasks_batch))
        return log_density

    # @tf.function(
    #     input_signature=[
    #         tf.TensorSpec(shape=None, dtype=tf.int32),
    #     ]
    # )
    def _sample_approximate_posterior(self, n_samples: int):
        z = self.post_batch.sample(n_samples=n_samples)

        # check output
        assert_shape(z, (n_samples, self.n_tasks_batch, self.d_z))
        return z

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.int32),
        ]
    )
    def _sample_prior(self, n_samples: int):
        z = self.prior.sample(n_samples)

        # check output
        assert_shape(z, (n_samples, 1, self.d_z))
        return z

    def reset_posteriors(self, n_tasks_total: int):
        """
        Create n_tasks_total uninitialized posterior GMMs.
        To initialize parameters, use initialize_posteriors() afterwards.
        """
        assert tf.executing_eagerly()

        self.n_tasks_total = n_tasks_total
        self.post_log_w = tf.zeros((n_tasks_total, self.post_n_comps))
        self.post_loc = tf.zeros((n_tasks_total, self.post_n_comps, self.d_z))
        self.post_prec = tf.zeros(
            (n_tasks_total, self.post_n_comps, self.d_z, self.d_z)
        )

    def set_batch(self, task_ids: tf.Tensor):
        """
        Choose posterior GMMs for current batch.
        """
        assert tf.executing_eagerly()

        self.batch_task_ids = task_ids
        self.post_batch = GMM(
            log_w=tf.gather(self.post_log_w, self.batch_task_ids),
            loc=tf.gather(self.post_loc, self.batch_task_ids),
            prec=tf.gather(self.post_prec, self.batch_task_ids),
        )

    def store_params_from_current_batch(
        self,
        task_ids: Optional[tf.Tensor] = None,
        log_w: Optional[tf.Tensor] = None,
        loc: Optional[tf.Tensor] = None,
        prec: Optional[tf.Tensor] = None,
    ):
        """
        Update parameters from current posterior GMM.
        """
        assert tf.executing_eagerly()

        if task_ids is None:  # update from self.gmm_post_batch
            assert log_w is None
            assert loc is None
            assert prec is None
            task_ids = self.batch_task_ids
            log_w = self.post_batch.log_w
            loc = self.post_batch.loc
            prec = self.post_batch.prec

        self.post_log_w.scatter_nd_update(task_ids[:, None], log_w)
        self.post_loc.scatter_nd_update(task_ids[:, None], loc)
        self.post_prec.scatter_nd_update(task_ids[:, None], prec)

    def initialize_posteriors(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        ctx_mask: tf.Tensor,
        task_ids: tf.Tensor,
        n_samples_comp: int,
    ):
        """
        Choose self.n_post_comp best prior components for each task in batch and
        initialize posterior with them.
        This also works for empty input x/y: then, the target density contains only
        the prior density and we choose those components that have highest prior density,
        which should make sense.
        """
        assert tf.executing_eagerly()

        ## check input
        n_tasks = tf.shape(x)[0]
        n_points = tf.shape(x)[1]
        assert_shape(x, (n_tasks, n_points, self.d_x))
        assert_shape(y, (n_tasks, n_points, self.d_y))
        assert_shape(ctx_mask, (n_tasks, n_points))
        assert_shape(task_ids, (n_tasks,))

        if self.prior_n_comps > self.post_n_comps:
            ## set parameters of current posterior to the best components of the prior
            # sample prior
            z = self.prior.sample_all_components(n_samples_per_component=n_samples_comp)
            assert_shape(z, (n_samples_comp, 1, self.prior_n_comps, self.d_z))
            # repeat samples for each task
            z = tf.broadcast_to(
                z, (n_samples_comp, n_tasks, self.prior_n_comps, self.d_z)
            )
            z = tf.transpose(z, (0, 2, 1, 3))
            z = tf.reshape(z, (n_samples_comp * self.prior_n_comps, n_tasks, self.d_z))
            assert_shape(z, (n_samples_comp * self.prior_n_comps, n_tasks, self.d_z))
            # evaluate target distribution on
            log_p_tgt = self._log_unnormalized_posterior_density(
                x=x, y=y, z=z, ctx_mask=ctx_mask
            )
            assert_shape(log_p_tgt, (n_samples_comp * self.prior_n_comps, n_tasks))
            log_p_tgt = tf.reshape(
                log_p_tgt, (n_samples_comp, self.prior_n_comps, n_tasks)
            )
            log_p_tgt = tf.transpose(log_p_tgt, (0, 2, 1))
            log_p_tgt = tf.reduce_mean(log_p_tgt, axis=0)
            assert_shape(log_p_tgt, (n_tasks, self.prior_n_comps))
            # choose self.post_n_comp best prior components per task
            sort_idx = tf.argsort(log_p_tgt, axis=1, direction="DESCENDING")
            best_idx = sort_idx[:, : self.post_n_comps]
        else:
            ## choose all prior components
            assert self.prior_n_comps == self.post_n_comps
            best_idx = np.repeat(
                np.arange(self.prior_n_comps).reshape(1, -1), n_tasks, axis=0
            )

        log_w = tf.gather(tf.squeeze(self.prior.log_w, 0), best_idx)
        loc = tf.gather(tf.squeeze(self.prior.loc, 0), best_idx)
        prec = tf.gather(tf.squeeze(self.prior.prec, 0), best_idx)

        ## set gmm parameters
        self.store_params_from_current_batch(
            task_ids=task_ids, log_w=log_w, loc=loc, prec=prec
        )

    def predict(
        self, x: np.ndarray, n_samples: int, sample_from: str = "approximate_posterior"
    ):
        assert tf.executing_eagerly()

        # check input
        n_tasks = tf.shape(x)[0]
        n_points = tf.shape(x)[1]
        assert_shape(x, (n_tasks, n_points, self.d_x))

        # convert input to tf
        x = tf.constant(x, dtype=tf.float32)

        # sample z
        z = self.sample_z(n_tasks=n_tasks, n_samples=n_samples, sample_from=sample_from)
        z = tf.constant(z, dtype=tf.float32)

        # perform prediction
        y_pred, var_y = self._predict(x=x, z=z)

        # convert output back to numpy
        y_pred, var_y = y_pred.numpy(), var_y.numpy()

        # check output
        assert_shape(y_pred, (n_samples, n_tasks, n_points, self.d_y))
        assert_shape(var_y, (n_samples, n_tasks, n_points, self.d_y))
        return y_pred, var_y

    def sample_z(
        self,
        n_samples: int,
        sample_from: str = "approximate_posterior",
        n_tasks: Optional[int] = None,
    ):
        # if sample_from == "approximate_posterior", n_tasks is redundant (as it
        #  has to be equal to self.n_tasks_batch), and so we don't force the user to
        #  provide it
        if n_tasks is None:
            n_tasks = self.n_tasks_batch
        if not sample_from in ["approximate_posterior", "prior"]:
            raise ValueError(f"Unknown value of argument 'sample_from' = {sample_from}")
        if sample_from == "approximate_posterior":
            assert n_tasks == self.n_tasks_batch

        # sample z
        if sample_from == "approximate_posterior":
            z = self._sample_approximate_posterior(n_samples=n_samples)
        elif sample_from == "prior":
            # n_samples prior samples per task
            z = self._sample_prior(n_samples=n_samples * n_tasks)
            z = tf.reshape(z, (n_samples, n_tasks, self.d_z))

        # convert output to numpy
        z = z.numpy()

        # check output
        assert_shape(z, (n_samples, n_tasks, self.d_z))

        return z

    def predict_at_z(self, x: np.ndarray, z: np.ndarray):
        assert tf.executing_eagerly()

        # check input
        n_tasks = tf.shape(x)[0]
        n_points = tf.shape(x)[1]
        n_samples = tf.shape(z)[0]
        assert_shape(x, (n_tasks, n_points, self.d_x))
        assert_shape(z, (n_samples, n_tasks, self.d_z))

        # convert input to tf
        x = tf.constant(x, dtype=tf.float32)
        z = tf.constant(z, dtype=tf.float32)

        # perform prediction
        y_pred, var_y = self._predict(x=x, z=z)

        # convert output back to numpy
        y_pred, var_y = y_pred.numpy(), var_y.numpy()

        # check output
        assert_shape(y_pred, (n_samples, n_tasks, n_points, self.d_y))
        assert_shape(var_y, (n_samples, n_tasks, n_points, self.d_y))
        return y_pred, var_y


def create_initial_gmm_parameters(
    d_z: int,
    n_tasks: int,
    n_components: int,
    prior_scale: float,
):
    prior = tfp.distributions.Normal(
        loc=tf.zeros(d_z), scale=prior_scale * tf.ones(d_z)
    )
    initial_cov = prior_scale**2 * tf.eye(d_z)  # same as prior covariance

    weights = tf.ones((n_tasks, n_components)) / n_components
    means = prior.sample((n_tasks, n_components))
    covs = tf.stack([initial_cov] * n_components, axis=0)
    covs = tf.stack([covs] * n_tasks, axis=0)

    # check output
    assert weights.shape == (n_tasks, n_components)
    assert means.shape == (n_tasks, n_components, d_z)
    assert covs.shape == (n_tasks, n_components, d_z, d_z)
    return weights, means, covs
