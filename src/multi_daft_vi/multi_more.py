import tensorflow as tf


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # eta
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),  # old_lin_term
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),  # old_prec
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),  # reward_lin_term
        tf.TensorSpec(
            shape=[None, None, None, None], dtype=tf.float32
        ),  # reward_quad_term
    ]
)
def get_natural_params_simple_reward(
    eta, old_lin_term, old_precision, reward_lin_term, reward_quad_term
):
    """
    Uses the reformulation of the reward with log(target_densitiy) - log(model_density). This leads to the parameterization
    new_prec = old_prec - 1/eta * reward_quad_term
    new_lin = old_lin + 1/eta * reward_lin_term
    """
    expanded_eta = tf.expand_dims(eta, -1)
    new_lin = old_lin_term + reward_lin_term / expanded_eta
    twice_expanded_eta = tf.expand_dims(expanded_eta, -1)
    new_precision = old_precision - reward_quad_term / twice_expanded_eta
    return new_lin, new_precision


def get_natural_params_max_entropy(
    eta, old_lin_term, old_precision, reward_lin_term, reward_quad_term
):
    expanded_eta = tf.expand_dims(eta, -1)
    new_lin = (expanded_eta * old_lin_term + reward_lin_term) / (expanded_eta + 1)
    twice_expanded_eta = tf.expand_dims(expanded_eta, -1)
    new_precision = (twice_expanded_eta * old_precision - reward_quad_term) / (
        twice_expanded_eta + 1
    )
    return new_lin, new_precision


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),  # chol_old_cov
        tf.TensorSpec(shape=[], dtype=tf.float32),  # dim_z
    ]
)
def get_kl_const_part(chol_old_cov, dim_z):
    old_logdet = 2 * tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(chol_old_cov)), axis=-1
    )
    return old_logdet - dim_z


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # eta
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),  # old_lin_term
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),  # old_mean
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),  # old_prec
        tf.TensorSpec(
            shape=[None, None, None, None], dtype=tf.float32
        ),  # inv_chol_old_cov
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),  # reward_lin_term
        tf.TensorSpec(
            shape=[None, None, None, None], dtype=tf.float32
        ),  # reward_quad_term
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # kl_const_part
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # eye_matrix
    ]
)
def get_kl(
    eta,
    old_lin_term,
    old_mean,
    old_prec,
    transposed_inv_chol_old_cov,
    reward_lin_term,
    reward_quad_term,
    kl_const_part,
    eye_matrix,
):
    new_lin, new_prec = get_natural_params_simple_reward(
        eta, old_lin_term, old_prec, reward_lin_term, reward_quad_term
    )

    chol_new_prec = tf.linalg.cholesky(new_prec)
    # check top right corner of each batch for NaN
    nan_output = tf.math.is_nan(chol_new_prec)[..., 0:1, 0:1]
    # replace nans with identities such that the program does not throw errors.
    safe_chol_new_prec = tf.where(nan_output, eye_matrix, chol_new_prec)
    safe_new_prec = tf.where(nan_output, eye_matrix, new_prec)
    # compute parts for the kl:
    new_logdet = -2 * tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(safe_chol_new_prec)), axis=-1
    )

    # uses that trace(M@M.T) = ||M||^2_2, and that trace has its cyclic property and cholesky identities
    # we can use triangular solve since safe_chol_new_prec is lower triangular
    trace_matrix = tf.linalg.triangular_solve(
        safe_chol_new_prec, transposed_inv_chol_old_cov, lower=True
    )
    trace_term = tf.reduce_sum(tf.square(trace_matrix), axis=(-2, -1))
    # compute the new_mean with the safe_new_prec to make sure that there is no matrix inversion error.
    new_mean = tf.linalg.solve(safe_new_prec, tf.expand_dims(new_lin, -1))[..., 0]
    diff = old_mean - new_mean  # shape batch x dim_z
    mahanalobis_dist = tf.reduce_sum(
        tf.square(
            tf.linalg.matvec(transposed_inv_chol_old_cov, diff, transpose_a=True)
        ),
        axis=-1,
    )
    kl = 0.5 * (kl_const_part - new_logdet + trace_term + mahanalobis_dist)
    # replace the components flagged as nan with real nans
    kl = tf.where(nan_output[..., 0, 0], tf.constant(float("NaN")), kl)
    return kl


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[], dtype=tf.float32),  # eps
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # lower_bound
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # upper_bound
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),  # old_mean
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),  # old_lin_term
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),  # old_prec
        tf.TensorSpec(
            shape=[None, None, None, None], dtype=tf.float32
        ),  # inv_chol_old_cov
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),  # reward_lin_term
        tf.TensorSpec(
            shape=[None, None, None, None], dtype=tf.float32
        ),  # reward_quad_term
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # kl_const_part
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # eye_matrix
    ]
)
def log_space_bracketing_search(
    eps,
    lower_bound,
    upper_bound,
    old_mean,
    old_lin_term,
    old_prec,
    transposed_inv_chol_old_cov,
    reward_lin_term,
    reward_quad_term,
    kl_const_part,
    eye_matrix,
):
    for iters in tf.range(1000):
        eta = 0.5 * (upper_bound + lower_bound)
        # test current eta
        kl = get_kl(
            tf.math.exp(eta),
            old_lin_term,
            old_mean,
            old_prec,
            transposed_inv_chol_old_cov,
            reward_lin_term,
            reward_quad_term,
            kl_const_part,
            eye_matrix,
        )
        converged = tf.math.exp(upper_bound) - tf.exp(lower_bound) < 1e-4
        if tf.math.reduce_all(converged):
            break
        # if kl is nan, condition is also false, which is what we want
        f_eval = eps - kl
        condition = f_eval > 0
        upper_bound = tf.where(condition, eta, upper_bound)
        lower_bound = tf.where(condition, lower_bound, eta)
    # use upper_bound as final value (less greedy)
    return tf.math.exp(upper_bound)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[], dtype=tf.float32),  # eps
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # lower_bound
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # upper_bound
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),  # old_mean
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),  # old_lin_term
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),  # old_prec
        tf.TensorSpec(
            shape=[None, None, None, None], dtype=tf.float32
        ),  # transposed_inv_chol_old_cov
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),  # reward_lin_term
        tf.TensorSpec(
            shape=[None, None, None, None], dtype=tf.float32
        ),  # reward_quad_term
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # kl_const_part
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # eye_matrix
        tf.TensorSpec(shape=[], dtype=tf.float32),  # conv_tol
        tf.TensorSpec(shape=[], dtype=tf.int32),  # max_iter
    ]
)
def tf_bracketing_search(
    eps,
    lower_bound,
    upper_bound,
    old_mean,
    old_lin_term,
    old_prec,
    transposed_inv_chol_old_cov,
    reward_lin_term,
    reward_quad_term,
    kl_const_part,
    eye_matrix,
    conv_tol,
    max_iter,
):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # lower_bound
            tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # upper_bound
            tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # eta
            tf.TensorSpec(shape=[], dtype=tf.int32),  # max_iter
        ]
    )
    def converged_condition(lower_bound_var, upper_bound_var, eta, num_iter):
        converged = tf.reduce_all(
            tf.reduce_any(
                [upper_bound_var - eta < conv_tol, eta - lower_bound_var < conv_tol],
                axis=0,
            )
        )
        # repeat until it is converged
        return tf.reduce_all([not converged, num_iter < max_iter])
        # return num_iter < max_iter

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # lower_bound
            tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # upper_bound
            tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # eta
            tf.TensorSpec(shape=[], dtype=tf.int32),  # num_iter
        ]
    )
    def loop_body(lower_bound_var, upper_bound_var, eta, num_iter):
        # print(f"{tf.reduce_mean(upper_bound_var - lower_bound_var)=}")
        # print(f"{tf.reduce_max(upper_bound_var - lower_bound_var)=}")
        # print("--------------")
        # test current eta
        kl = get_kl(
            eta,
            old_lin_term,
            old_mean,
            old_prec,
            transposed_inv_chol_old_cov,
            reward_lin_term,
            reward_quad_term,
            kl_const_part,
            eye_matrix,
        )
        condition = kl < eps
        upper_bound_var = tf.where(condition, eta, upper_bound_var)
        lower_bound_var = tf.where(condition, lower_bound_var, eta)
        eta = 0.5 * (upper_bound_var + lower_bound_var)
        return lower_bound_var, upper_bound_var, eta, num_iter + 1

    num_iter = tf.constant(0, dtype=tf.int32)
    eta = 0.5 * (upper_bound + lower_bound)
    lower_bound, upper_bound, eta, num_iter = tf.while_loop(
        converged_condition,
        loop_body,
        loop_vars=(lower_bound, upper_bound, eta, num_iter),
    )
    # print(num_iter)
    return upper_bound, int(num_iter)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),  # new_lin_term
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),  # new_prec
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),  # old_lin_term
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),  # old_prec
    ]
)
def satisfy_pd_constraint(new_lin_term, new_prec, old_lin_term, old_prec):
    # test for pd by chol decomp
    chol_new_prec = tf.linalg.cholesky(new_prec)
    nan_output = tf.math.is_nan(chol_new_prec)
    # nan_output only in lower triangular part. We want the whole matrix to be Nan in order to replace it with the old step
    nan_output = nan_output[..., 0, 0]
    nan_output = nan_output[..., tf.newaxis, tf.newaxis]
    # where the update fails it takes the old parameters
    safe_new_prec = tf.where(nan_output, old_prec, new_prec)
    safe_new_lin_term = tf.where(nan_output[..., 0], old_lin_term, new_lin_term)
    return safe_new_lin_term, safe_new_prec, tf.math.logical_not(nan_output[..., 0, 0])


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),  # chol_old_cov
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),  # old_prec
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),  # old_mean_term
        tf.TensorSpec(shape=[], dtype=tf.float32),  # dim_z
        tf.TensorSpec(shape=[], dtype=tf.float32),  # eta_lower_bound
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # old_eta
        tf.TensorSpec(shape=[None, None], dtype=tf.bool),  # old_success
        tf.TensorSpec(shape=[], dtype=tf.float32),  # global_lower_bound
        tf.TensorSpec(shape=[], dtype=tf.float32),  # global_upper_bound
        tf.TensorSpec(shape=[], dtype=tf.bool),  # log_space
        tf.TensorSpec(shape=[], dtype=tf.bool),  # use_warm_starts
        tf.TensorSpec(shape=[], dtype=tf.float32),  # warm_start_interval_size
    ]
)
def init_bracketing_search(
    chol_old_cov,
    old_prec,
    old_mean,
    dim_z,
    eta_lower_bound,
    old_eta,
    old_success,
    global_lower_bound,
    global_upper_bound,
    log_space,
    use_warm_starts,
    warm_start_interval_size,
):
    kl_const_part = get_kl_const_part(chol_old_cov, dim_z)
    old_lin_term = tf.linalg.matvec(old_prec, old_mean)
    transposed_inv_chol_old_cov = tf.transpose(
        tf.linalg.inv(chol_old_cov), perm=[0, 1, 3, 2]
    )

    if use_warm_starts:
        # warm start
        if log_space:
            lower_bound = tf.maximum(
                tf.maximum(0.0, tf.math.log(eta_lower_bound)),
                tf.math.log(old_eta) - 0.1,
            )
            upper_bound = tf.math.log(old_eta) + 0.1
        else:
            lower_bound = tf.maximum(
                tf.maximum(1.0, eta_lower_bound),
                old_eta - warm_start_interval_size / 2.0,
            )
            upper_bound = old_eta + warm_start_interval_size / 2.0

        # select warm start on previous successful updates
        lower_bound = tf.where(old_success, lower_bound, global_lower_bound)
        upper_bound = tf.where(old_success, upper_bound, global_upper_bound)
    else:
        identity = old_eta * 0.0 + 1.0
        lower_bound = identity * global_lower_bound
        upper_bound = identity * global_upper_bound

    return (
        kl_const_part,
        old_lin_term,
        transposed_inv_chol_old_cov,
        lower_bound,
        upper_bound,
    )


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # eta
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),  # old_lin_term
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),  # old_prec
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),  # reward_lin
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),  # reward_quad
    ]
)
def get_distribution(eta, old_lin_term, old_prec, reward_lin_term, reward_quad_term):
    new_lin_term, new_prec = get_natural_params_simple_reward(
        eta, old_lin_term, old_prec, reward_lin_term, reward_quad_term
    )
    # test the output to make sure that precision is positive definite
    new_lin_term, new_prec, success = satisfy_pd_constraint(
        new_lin_term, new_prec, old_lin_term, old_prec
    )
    new_mean = tf.linalg.solve(new_prec, tf.expand_dims(new_lin_term, -1))[..., 0]
    return new_mean, new_prec, success


class MORE:
    def __init__(
        self,
        batch_dim_and_components,
        dim_z,
        log_space,
        conv_tol,
        global_upper_bound,
        use_warm_starts,
        warm_start_interval_size,
        eta_lower_bound=1e-8,
    ):
        self.log_space = tf.constant(log_space, dtype=tf.bool)
        self.eta_lower_bound = tf.constant(eta_lower_bound, dtype=tf.float32)
        self.use_warm_starts = tf.constant(use_warm_starts, dtype=tf.bool)
        self.warm_start_interval_size = tf.constant(
            warm_start_interval_size, dtype=tf.float32
        )
        self.max_dual_steps = 50

        # full range
        if self.log_space:
            self.global_lower_bound = tf.maximum(0.0, tf.math.log(eta_lower_bound))
            self.global_upper_bound = tf.math.log(global_upper_bound)
        else:
            self.global_lower_bound = tf.maximum(1.0, eta_lower_bound)
            self.global_upper_bound = tf.constant(global_upper_bound)

        self.eye_matrix = tf.eye(dim_z)
        # this selects if warm start or big range for dual optimization. In the beginning: no warm start
        self.old_success = tf.constant(False, shape=batch_dim_and_components)
        self.old_eta = tf.constant(global_upper_bound, shape=batch_dim_and_components)
        self.conv_tol = tf.constant(conv_tol)

    @property
    def eta(self):
        return self.old_eta

    @eta.setter
    def eta(self, value):
        self.old_eta = value

    @property
    def success(self):
        return self.old_success

    @success.setter
    def success(self, value):
        self.old_success = value

    def step(
        self, eps, old_mean, chol_old_cov, old_prec, reward_quad_term, reward_lin_term
    ):
        dim_z = tf.constant(reward_lin_term.shape[-1], dtype=tf.float32)
        (
            kl_const_part,
            old_lin_term,
            transposed_inv_chol_old_cov,
            lower_bound,
            upper_bound,
        ) = init_bracketing_search(
            chol_old_cov,
            old_prec,
            old_mean,
            dim_z,
            self.eta_lower_bound,
            self.old_eta,
            self.old_success,
            self.global_lower_bound,
            self.global_upper_bound,
            self.log_space,
            self.use_warm_starts,
            self.warm_start_interval_size,
        )
        # solve dual
        if self.log_space:
            self.old_eta = log_space_bracketing_search(
                eps,
                lower_bound,
                upper_bound,
                old_mean,
                old_lin_term,
                old_prec,
                transposed_inv_chol_old_cov,
                reward_lin_term,
                reward_quad_term,
                kl_const_part,
                self.eye_matrix,
            )
        else:
            self.old_eta, num_iter = tf_bracketing_search(
                eps,
                lower_bound,
                upper_bound,
                old_mean,
                old_lin_term,
                old_prec,
                transposed_inv_chol_old_cov,
                reward_lin_term,
                reward_quad_term,
                kl_const_part,
                self.eye_matrix,
                self.conv_tol,
                max_iter=self.max_dual_steps,
            )
            # print(f"{num_iter=}")
        new_mean, new_prec, self.old_success = get_distribution(
            self.old_eta, old_lin_term, old_prec, reward_lin_term, reward_quad_term
        )
        return new_mean, new_prec
