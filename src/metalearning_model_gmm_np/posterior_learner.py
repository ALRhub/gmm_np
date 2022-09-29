from abc import ABC, abstractmethod

import tensorflow as tf
from multi_daft_vi.multi_daft_vi import step as multi_daft_step

from metalearning_model_gmm_np.np import NP
from metalearning_model_gmm_np.util import assert_shape
from gmm_util.util import LNPDF


class ApproximatePosteriorLearner(ABC):
    def __init__(self, np_model: NP):
        self.np_model = np_model
        self.target_dist = LNPDF_NP(np_model=self.np_model)

    def step(self, x: tf.Tensor, y: tf.Tensor, ctx_mask: tf.Tensor):
        assert tf.executing_eagerly()

        ## check input
        n_points_tgt = tf.shape(x)[1]
        assert_shape(x, (self.np_model.n_tasks_batch, n_points_tgt, self.np_model.d_x))
        assert_shape(y, (self.np_model.n_tasks_batch, n_points_tgt, self.np_model.d_y))

        ## step
        self.target_dist.condition_on(x=x, y=y, ctx_mask=ctx_mask)
        # calls the specific update method of the gmm learner
        model, n_feval, mean_log_tgt_density = self.gmm_learner_step()
        self.np_model.store_params_from_current_batch()

        # return some metrics
        metrics = {"mean_log_tgt_density": mean_log_tgt_density, "n_feval": n_feval}
        return metrics

    @abstractmethod
    def gmm_learner_step(self):
        raise NotImplementedError

    @property
    def trackables(self):  # may be overwritten if trackables exists
        return {}


class MultiDaftLearner(ApproximatePosteriorLearner):
    def __init__(
        self,
        np_model,
        n_samples_per_comp,
        component_kl_bound,
        dual_conv_tol,
        global_upper_bound,
        use_warm_starts,
        diagonal_cov,
    ):
        super().__init__(np_model=np_model)
        self.n_samples_per_comp = n_samples_per_comp
        self.component_kl_bound = component_kl_bound
        self.dual_conv_tol = dual_conv_tol
        self.global_upper_bound = global_upper_bound
        self.use_warm_starts = use_warm_starts
        self.diagonal_cov = diagonal_cov
        if self.use_warm_starts:
            raise NotImplementedError(
                "Warm start functionality not integrated yet. "
                "Set warm_start_interval_size correctly!"
            )

    def gmm_learner_step(self):
        return multi_daft_step(
            target_dist=self.target_dist,
            model=self.np_model.post_batch,
            n_samples_per_comp=self.n_samples_per_comp,
            component_kl_bound=self.component_kl_bound,
            dual_conv_tol=self.dual_conv_tol,
            global_upper_bound=self.global_upper_bound,
            use_warm_starts=self.use_warm_starts,
            warm_start_interval_size=-1,  # Not used yet, tbd
            more_optimizer=None,  # Has to do with warm starts, tbd
            diagonal_cov=self.diagonal_cov,
        )


class LNPDF_NP(LNPDF):
    def __init__(self, np_model: NP):
        self.np = np_model

        # will be set in self.condition_on()
        self._x = None
        self._y = None
        self._ctx_mask = None

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def ctx_mask(self):
        return self._ctx_mask

    def condition_on(self, x: tf.Tensor, y: tf.Tensor, ctx_mask: tf.Tensor):
        if self._x is None:
            assert self._y is None
            assert self._ctx_mask is None
            self._x = tf.Variable(
                shape=[None, None, None], initial_value=x, dtype=tf.float32
            )
            self._y = tf.Variable(
                shape=[None, None, None], initial_value=y, dtype=tf.float32
            )
            self._ctx_mask = tf.Variable(
                shape=[None, None], initial_value=ctx_mask, dtype=tf.bool
            )
        else:
            self._x.assign(x)
            self._y.assign(y)
            self._ctx_mask.assign(ctx_mask)

    def can_sample(self):
        return False

    def get_num_dimensions(self):
        return self.np.d_z

    def log_density(
        self, z: tf.Tensor, compute_grad: bool = False, compute_hess: bool = False
    ):
        assert tf.executing_eagerly()

        # check input
        n_samples = tf.shape(z)[0]
        assert_shape(z, (n_samples, self.np.n_tasks_batch, self.np.d_z))
        assert not (compute_hess and not compute_grad)

        # compute log_density (and possibly gradient/hessian)
        if compute_hess:
            raise NotImplementedError
        if compute_grad:
            log_density_hess = None
            log_density, log_density_grad = self._eval_grad_np_latent_distribution(
                x=self.x, y=self.y, z=z, ctx_mask=self.ctx_mask
            )
        else:
            log_density_hess, log_density_grad = None, None
            log_density = self._eval_np_latent_distribution(
                x=self.x, y=self.y, z=z, ctx_mask=self.ctx_mask
            )

        # check output
        assert_shape(log_density, (n_samples, self.np.n_tasks_batch))
        if compute_grad:
            assert_shape(
                log_density_grad,
                (n_samples, self.np.n_tasks_batch, self.np.d_z),
            )
        if compute_hess:
            assert_shape(
                log_density_hess,
                (n_samples, self.np.n_tasks_batch, self.np.d_z, self.np.d_z),
            )
        return log_density, log_density_grad, log_density_hess

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None], dtype=tf.bool),
        ]
    )
    def _eval_np_latent_distribution(
        self, x: tf.Tensor, y: tf.Tensor, z: tf.Tensor, ctx_mask: tf.Tensor
    ):
        return self.np._log_unnormalized_posterior_density(
            x=x, y=y, z=z, ctx_mask=ctx_mask
        )

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None], dtype=tf.bool),
        ]
    )
    def _eval_grad_np_latent_distribution(
        self, x: tf.Tensor, y: tf.Tensor, z: tf.Tensor, ctx_mask: tf.Tensor
    ):
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(z)
            f_z = self.np._log_unnormalized_posterior_density(
                x=x, y=y, z=z, ctx_mask=ctx_mask
            )
        f_z_grad = g.gradient(f_z, z)
        return f_z, f_z_grad
