import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from gmm_util.util import LNPDF
from gmm_util.gmm import GMM as DaftGMM

from multi_daft_vi.multi_more import MORE


def multi_daft_vi(
    config: dict,
    target_dist: LNPDF,
    w_init: tf.Tensor,
    mu_init: tf.Tensor,
    cov_init: tf.Tensor,
    callback,
):
    """
    Basic use of the DAFT VI with multiple tasks.
    """
    batch_shape = w_init.shape[:-1]
    n_components = w_init.shape[-1]
    d_z = mu_init.shape[-1]
    assert w_init.shape == batch_shape + (n_components,)
    assert mu_init.shape == batch_shape + (n_components, d_z)
    assert cov_init.shape == batch_shape + (n_components, d_z, d_z)

    # savepath
    os.makedirs(config["savepath"], exist_ok=True)
    # check compatibility of model and target dist
    assert target_dist.get_num_dimensions() == d_z

    ## instantiate model and target distribution
    model = DaftGMM(
        log_w=tf.math.log(w_init),
        loc=mu_init,
        prec=tf.linalg.inv(cov_init),
    )

    # Much more stable and now optimized with log_space==False. log_space==True is in this version untested!
    more_optimizer = MORE(
        batch_shape + (n_components,),
        d_z,
        log_space=False,
        conv_tol=config["dual_conv_tol"],
        global_upper_bound=config["global_upper_bound"],
        use_warm_starts=config["use_warm_starts"],
        warm_start_interval_size=config["warm_start_interval_size"],
    )

    ## training loop
    n_fevals_total = 0
    with tqdm(total=config["n_iter"]) as pbar:
        for i in range(config["n_iter"]):
            # update parameters
            model, n_fevals, _ = step(
                target_dist=target_dist,
                model=model,
                n_samples_per_comp=config["n_samples_per_comp"],
                component_kl_bound=config["component_kl_bound"],
                dual_conv_tol=config["dual_conv_tol"],
                global_upper_bound=config["global_upper_bound"],
                use_warm_starts=config["use_warm_starts"],
                warm_start_interval_size=config["warm_start_interval_size"],
                diagonal_cov=config["diagonal_cov"],
                more_optimizer=more_optimizer,
            )
            # update n_fevals_total
            n_fevals_total += n_fevals

            # log
            pbar.update()
            if callback is not None and (
                i % config["callback_interval"] == 0 or i == config["n_iter"] - 1
            ):
                callback(model=model)

    return model


def step(
    target_dist,
    model,
    n_samples_per_comp,
    component_kl_bound,
    dual_conv_tol,
    global_upper_bound,
    use_warm_starts,
    warm_start_interval_size,
    diagonal_cov=False,
    more_optimizer=None,
):
    if diagonal_cov:
        d_z = model.prec.shape[-1]
        eye = tf.linalg.eye(d_z)
        diag_prec = model.prec * eye
        model.prec = diag_prec

    samples, rewards, rewards_grad, target_densities = get_samples_and_rewards(
        target_dist, model, n_samples_per_comp
    )
    num_samples_per_component, num_task, num_components, dim_z = samples.shape
    if more_optimizer is None:
        # If step should be called stateless, the more optimizer is created here.
        # However, this is not optimal since this does not allow
        # warm starts in general and is less efficient
        more_optimizer = MORE(
            (num_task, num_components),
            dim_z,
            log_space=False,
            conv_tol=dual_conv_tol,
            global_upper_bound=global_upper_bound,
            use_warm_starts=use_warm_starts,
            warm_start_interval_size=warm_start_interval_size,
        )
    if diagonal_cov:
        quad_term, lin_term = diagonal_model_fitting(
            model.loc, model.prec, samples, rewards_grad, eye
        )
    else:
        quad_term, lin_term = model_fitting(
            model.loc, model.prec, samples, rewards_grad
        )
    # update of components
    new_loc, new_prec = more_optimizer.step(
        tf.constant(component_kl_bound, dtype=tf.float32),
        model.loc,
        model.scale_tril,
        model.prec,
        quad_term,
        lin_term,
    )
    model.loc = new_loc
    model.prec = new_prec
    # update of weights
    new_weights = weight_update(model.log_w, rewards)
    model.log_w = new_weights
    return (
        model,
        num_samples_per_component * num_components * num_task,
        np.mean(target_densities.numpy()),
    )


def get_rewards(target_dist, model, samples_per_comp):
    num_samples, num_tasks, num_components, dim_z = tuple(samples_per_comp.shape)
    samples_flattened = tf.reshape(
        tf.transpose(samples_per_comp, [0, 2, 1, 3]), [-1, num_tasks, dim_z]
    )
    # should now have shape [num_samples_per_comp * num_comp, task, dz]
    # get the target and model log densities + gradients
    target_densities, target_grad, _ = target_dist.log_density(
        samples_flattened, compute_grad=True, compute_hess=False
    )
    model_densities, model_grad, _ = model.log_density(
        samples_flattened, compute_grad=True, compute_hess=False
    )
    # combine to get rewards
    rewards_flatten = target_densities - model_densities
    rewards_grad_flatten = target_grad - model_grad
    # unflatten
    rewards = tf.transpose(
        tf.reshape(rewards_flatten, [num_samples, num_components, num_tasks]), [0, 2, 1]
    )
    rewards_grad = tf.transpose(
        tf.reshape(
            rewards_grad_flatten, [num_samples, num_components, num_tasks, dim_z]
        ),
        [0, 2, 1, 3],
    )
    return rewards, rewards_grad, target_densities


def get_samples_and_rewards(target_dist, model, n_samples_per_comp):
    # sampling
    samples_per_comp = model.sample_all_components(n_samples_per_comp)
    rewards, rewards_grad, target_densities = get_rewards(
        target_dist, model, samples_per_comp
    )
    return samples_per_comp, rewards, rewards_grad, target_densities


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
    ]
)
def model_fitting(mean, prec, samples, rewards_grad):
    # model is 0.5 x^T R x + x^T r + r_0
    # It uses Stein for the quadratic matrix R
    diff = samples - mean
    prec_times_diff = tf.einsum("tcnm, stcm->stcn", prec, diff)
    exp_hessian_per_sample = tf.einsum(
        "stcn, stcm->stcnm", prec_times_diff, rewards_grad
    )
    exp_hessian = tf.reduce_mean(exp_hessian_per_sample, axis=0)
    exp_hessian = 0.5 * (exp_hessian + tf.transpose(exp_hessian, [0, 1, 3, 2]))
    exp_gradient = tf.math.reduce_mean(rewards_grad, axis=0)
    quad_term = exp_hessian
    lin_term = exp_gradient - tf.linalg.matvec(quad_term, mean)

    return quad_term, lin_term


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    ]
)
def diagonal_model_fitting(mean, prec, samples, rewards_grad, eye):
    diff = samples - mean
    prec_times_diff = tf.einsum("tcnm, stcm->stcn", prec, diff)
    exp_hessian_per_sample = tf.einsum(
        "stcn, stcm->stcnm", prec_times_diff, rewards_grad
    )
    exp_hessian = tf.reduce_mean(exp_hessian_per_sample, axis=0)
    # clip to diagonal
    exp_hessian = exp_hessian * eye
    exp_gradient = tf.math.reduce_mean(rewards_grad, axis=0)
    quad_term = exp_hessian
    lin_term = exp_gradient - tf.linalg.matvec(quad_term, mean)
    return quad_term, lin_term


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    ]
)
def weight_update(model_log_w, rewards):
    avg_rewards = tf.reduce_mean(rewards, axis=0)
    unnormalized_log_weights = model_log_w + avg_rewards
    new_log_weights = unnormalized_log_weights - tf.reduce_logsumexp(
        unnormalized_log_weights, axis=-1, keepdims=True
    )
    return new_log_weights
