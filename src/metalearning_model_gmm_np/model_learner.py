import tensorflow as tf
from tensorflow import keras
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from metalearning_model_gmm_np.np import NP
from metalearning_model_gmm_np.util import assert_shape


class ModelLearner:
    def __init__(
        self,
        np_model: NP,
        lr_likelihood: float,
        n_samples: int,
        learn_prior: bool,
        prior_diagonal_cov: bool,
        n_tasks_batch: int,
        n_tasks_total: int,
    ):
        self.np = np_model
        self.n_samples = n_samples
        self.lr_lhd = lr_likelihood
        self.learn_prior = learn_prior
        self.prior_diagonal_cov = prior_diagonal_cov

        if learn_prior:
            self.sample_buffer_prior = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                data_spec=tf.TensorSpec([n_tasks_batch, self.np.d_z], dtype=tf.float32),
                batch_size=n_samples,
                max_length=n_tasks_total // n_tasks_batch,
            )  # stores n_tasks_total batches of n_samples
        else:
            self.sample_buffer_prior = None
        self.optim_lhd = keras.optimizers.Adam(learning_rate=self.lr_lhd)

    @property
    def trackables(self) -> dict:
        return {"optim_lhd": self.optim_lhd}

    def step(self, x: tf.Tensor, y: tf.Tensor, ctx_mask: tf.Tensor):
        assert tf.executing_eagerly()

        # check input
        n_points = tf.shape(x)[1]
        assert_shape(x, (self.np.n_tasks_batch, n_points, self.np.d_x))
        assert_shape(y, (self.np.n_tasks_batch, n_points, self.np.d_y))
        assert_shape(ctx_mask, (self.np.n_tasks_batch, n_points))

        # step likelihood and prior
        z_post = self.np._sample_approximate_posterior(n_samples=self.n_samples)
        loss_likelihood = self.step_likelihood(x=x, y=y, z=z_post, ctx_mask=ctx_mask)
        if self.learn_prior:
            self.sample_buffer_prior.add_batch(z_post)
            z = tf.reshape(
                self.sample_buffer_prior.gather_all(),
                (-1, self.np.n_tasks_batch, self.np.d_z),
            )
            self.step_prior(z)

        # return metrics
        metrics = {"loss_likelihood": loss_likelihood.numpy()}
        return metrics

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None], dtype=tf.bool),
        ]
    )
    def step_likelihood(
        self, x: tf.Tensor, y: tf.Tensor, z: tf.Tensor, ctx_mask: tf.Tensor
    ):
        """
        Perform one gradient step on the likelihood parameters.
        """
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(
                self.np._log_likelihood(x=x, y=y, z=z, ctx_mask=ctx_mask)
            )
        # step optimizer
        grads = tape.gradient(target=loss, sources=self.np.decoder.trainable_weights)
        self.optim_lhd.apply_gradients(zip(grads, self.np.decoder.trainable_weights))

        return loss

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)]
    )
    def step_prior(self, z: tf.Tensor):
        """
        Perform one EM step on the prior parameters
        """
        ## check input
        n_samples = tf.shape(z)[0]
        n_z = n_samples * self.np.n_tasks_batch
        n_comp = self.np.prior.n_components
        d_z = self.np.d_z
        assert_shape(z, (n_samples, self.np.n_tasks_batch, self.np.d_z))

        ### perform step
        # flatten z
        z = tf.reshape(z, (n_z, self.np.d_z))
        ## compute responsibilities
        log_resp = self.np.prior.log_responsibilities(z=z[:, None, :])
        log_resp = tf.squeeze(log_resp, axis=1)
        assert_shape(log_resp, (n_z, self.np.prior.n_components))
        n_k = tf.exp(tf.math.reduce_logsumexp(log_resp, axis=0))
        assert_shape(n_k, (self.np.prior.n_components,))
        # update weights
        new_log_w = tf.math.log(n_k) - tf.math.log(tf.cast(n_z, tf.float32))
        ## update means
        resp = tf.math.exp(log_resp)
        new_loc = resp[:, :, None] * z[:, None, :]
        assert_shape(new_loc, (n_z, n_comp, d_z))
        new_loc = 1.0 / n_k[:, None] * tf.reduce_sum(new_loc, axis=0)
        assert_shape(new_loc, (n_comp, d_z))
        ## update covariances
        diff = z[:, None, :] - new_loc[None, :, :]
        assert_shape(diff, (n_z, n_comp, d_z))
        new_cov = tf.einsum("...i,...j->...ij", diff, diff)
        assert_shape(new_cov, (n_z, n_comp, d_z, d_z))
        new_cov = resp[:, :, None, None] * new_cov
        assert_shape(new_cov, (n_z, n_comp, d_z, d_z))
        new_cov = 1.0 / n_k[:, None, None] * tf.reduce_sum(new_cov, axis=0)
        if self.prior_diagonal_cov:
            new_cov = tf.linalg.diag(tf.linalg.diag_part(new_cov))

        ## check outputs
        if tf.executing_eagerly():
            assert tf.reduce_all(tf.shape(new_log_w) == (n_comp,))
            assert tf.reduce_all(tf.shape(new_loc) == (n_comp, d_z))
            assert tf.reduce_all(tf.shape(new_cov) == (n_comp, d_z, d_z))
        self.np.prior.log_w = new_log_w[None, :]
        self.np.prior.loc = new_loc[None, :, :]
        self.np.prior.prec = tf.linalg.inv(new_cov)[None, :, :, :]
