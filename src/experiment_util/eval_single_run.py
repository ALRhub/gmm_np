import os
import shutil
import tempfile
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from eval_util.util import elbo as compute_elbo
from eval_util.util import (
    log_marginal_likelihood_iw_mc,
    log_marginal_likelihood_naive_mc,
)
from eval_util.util import mse2 as compute_mse
from matplotlib import pyplot as plt
from metalearning_benchmarks import MetaLearningBenchmark
from metalearning_benchmarks.util import collate_benchmark

from experiment_util.metalearning_model import (
    MetaLearningLVM,
    MetaLearningLVMParametric,
)


def _summarize_one_run_by_context_size(
    metrics: pd.DataFrame,
    task_aggegrate_op: str,
):
    return (
        metrics.groupby(["n_ctx", "metric_type"])
        .aggregate(task_aggegrate_op)
        .reset_index()
    )


def _summarize_one_run(
    metrics: pd.DataFrame,
    task_aggregate_op: str,
):
    m_by_n_ctx = _summarize_one_run_by_context_size(
        metrics=metrics, task_aggegrate_op=task_aggregate_op
    )
    return (
        m_by_n_ctx.groupby(["metric_type"])
        .aggregate("mean")  # take mean over different n_ctx
        .reset_index()
        .drop(columns="n_ctx")
    )


def _summarize_one_bo_run_by_iteration(
    metrics: pd.DataFrame,
    task_aggegrate_op: str,
):
    return (
        metrics.groupby(["iter", "metric_type"])
        .aggregate(task_aggegrate_op)
        .reset_index()
        .drop(columns="task_id")  # we store the task ID for BO
    )


def _summarize_one_bo_run(
    metrics: pd.DataFrame,
    task_aggregate_op: str,
):
    m_by_n_iter = _summarize_one_bo_run_by_iteration(
        metrics=metrics, task_aggegrate_op=task_aggregate_op
    )
    return (
        m_by_n_iter.groupby(["metric_type"])
        .aggregate("mean")  # take mean over different n_ctx
        .reset_index()
        .drop(columns="iter")
    )


def _compute_predictions_at_z_in_batches(
    model: MetaLearningLVM,
    x: np.ndarray,
    z: np.ndarray,
    d_y: int,
    batch_size: int,
):
    # check input data
    n_samples = z.shape[0]
    n_tasks = x.shape[0]
    n_points = x.shape[1]
    d_x = x.shape[2]
    d_z = z.shape[2]
    task_ids = np.arange(n_tasks)
    assert x.shape == (n_tasks, n_points, d_x)
    assert z.shape == (n_samples, n_tasks, d_z)
    assert task_ids.shape == (n_tasks,)

    # compute number of minibatches
    n_mb = int(np.ceil(n_tasks / batch_size))

    # compute predictions
    y_pred = np.zeros((n_samples, n_tasks, n_points, d_y))
    var_pred = np.ones((n_samples, n_tasks, n_points, d_y))
    for (x_mb, z_mb, task_ids_mb) in tqdm(
        zip(  # minibatch over tasks
            np.array_split(x, n_mb, axis=0),
            np.array_split(z, n_mb, axis=1),
            np.array_split(task_ids, n_mb, axis=0),
        ),
        desc="Compute predictions",
        total=n_mb,
    ):
        y_pred[:, task_ids_mb], var_pred[:, task_ids_mb] = model.predict_at_z(
            x=x_mb,
            z=z_mb,
            task_ids=task_ids_mb,
        )

    return y_pred, var_pred


def _compute_mse_in_batches(
    y: np.ndarray,
    y_pred_at_z_prior: np.ndarray,
    batch_size: int,
):
    # check input data
    n_samples = y_pred_at_z_prior.shape[0]
    n_tasks = y_pred_at_z_prior.shape[1]
    n_points = y_pred_at_z_prior.shape[2]
    d_y = y_pred_at_z_prior.shape[3]
    task_ids = np.arange(n_tasks)
    assert y.shape == (n_tasks, n_points, d_y)
    assert y_pred_at_z_prior.shape == (n_samples, n_tasks, n_points, d_y)
    assert task_ids.shape == (n_tasks,)

    # compute number of minibatches
    n_mb = int(np.ceil(n_tasks / batch_size))

    # compute metrics
    mse = np.zeros((n_tasks,))
    for (y_mb, y_pred_at_z_prior_mb, task_ids_mb) in tqdm(
        zip(  # minibatch over tasks
            np.array_split(y, n_mb, axis=0),
            np.array_split(y_pred_at_z_prior, n_mb, axis=1),
            np.array_split(task_ids, n_mb, axis=0),
        ),
        desc="Computing MSE...",
        total=n_mb,
    ):
        mse[task_ids_mb] = compute_mse(y=y_mb, y_pred=y_pred_at_z_prior_mb)

    return mse


def _compute_lmlhd_naive_mc_in_batches(
    y: np.ndarray,
    y_pred_at_z_prior: np.ndarray,
    var_pred_at_z_prior: np.ndarray,
    batch_size: int,
):
    # check input data
    n_samples = y_pred_at_z_prior.shape[0]
    n_tasks = y_pred_at_z_prior.shape[1]
    n_points = y_pred_at_z_prior.shape[2]
    d_y = y_pred_at_z_prior.shape[3]
    task_ids = np.arange(n_tasks)
    assert y.shape == (n_tasks, n_points, d_y)
    assert y_pred_at_z_prior.shape == (n_samples, n_tasks, n_points, d_y)
    assert var_pred_at_z_prior.shape == (n_samples, n_tasks, n_points, d_y)
    assert task_ids.shape == (n_tasks,)

    # compute number of minibatches
    n_mb = int(np.ceil(n_tasks / batch_size))

    # compute metrics
    lmlhd_naive_mc = np.zeros((n_tasks,))
    for (y_mb, y_pred_at_z_prior_mb, var_pred_at_z_prior_mb, task_ids_mb) in tqdm(
        zip(  # minibatch over tasks
            np.array_split(y, n_mb, axis=0),
            np.array_split(y_pred_at_z_prior, n_mb, axis=1),
            np.array_split(var_pred_at_z_prior, n_mb, axis=1),
            np.array_split(task_ids, n_mb, axis=0),
        ),
        desc="Computing LMLHD_NAIVE_MC...",
        total=n_mb,
    ):
        lmlhd_naive_mc[task_ids_mb] = log_marginal_likelihood_naive_mc(
            y=y_mb,
            y_pred_at_z_prior=y_pred_at_z_prior_mb,
            var_pred_at_z_prior=var_pred_at_z_prior_mb,
        )

    return lmlhd_naive_mc


def _compute_lmlhd_iw_mc_in_batches(
    y: np.ndarray,
    batch_size: int,
    y_pred_at_z_proposal: np.ndarray = None,
    var_pred_at_z_proposal: np.ndarray = None,
    log_prob_prior_at_z_proposal: np.ndarray = None,
    log_prob_proposal_at_z_proposal: np.ndarray = None,
):
    # check input data
    n_samples = y_pred_at_z_proposal.shape[0]
    n_tasks = y_pred_at_z_proposal.shape[1]
    n_points = y_pred_at_z_proposal.shape[2]
    d_y = y_pred_at_z_proposal.shape[3]
    task_ids = np.arange(n_tasks)
    assert y.shape == (n_tasks, n_points, d_y)
    assert y_pred_at_z_proposal.shape == (n_samples, n_tasks, n_points, d_y)
    assert var_pred_at_z_proposal.shape == (n_samples, n_tasks, n_points, d_y)
    assert log_prob_prior_at_z_proposal.shape == (n_samples, n_tasks)
    assert log_prob_proposal_at_z_proposal.shape == (n_samples, n_tasks)
    assert task_ids.shape == (n_tasks,)

    # compute number of minibatches
    n_mb = int(np.ceil(n_tasks / batch_size))

    # compute metrics
    lmlhd_iw_mc = np.zeros((n_tasks,))
    for (
        y_mb,
        y_pred_at_z_proposal_mb,
        var_pred_at_z_proposal_mb,
        log_prob_prior_at_z_proposal_mb,
        log_prob_proposal_at_z_proposal_mb,
        task_ids_mb,
    ) in tqdm(
        zip(  # minibatch over tasks
            np.array_split(y, n_mb, axis=0),
            np.array_split(y_pred_at_z_proposal, n_mb, axis=1),
            np.array_split(var_pred_at_z_proposal, n_mb, axis=1),
            np.array_split(log_prob_prior_at_z_proposal, n_mb, axis=1),
            np.array_split(log_prob_proposal_at_z_proposal, n_mb, axis=1),
            np.array_split(task_ids, n_mb, axis=0),
        ),
        desc="Computing LMLHD_IW_MC...",
        total=n_mb,
    ):
        lmlhd_iw_mc[task_ids_mb] = log_marginal_likelihood_iw_mc(
            y=y_mb,
            y_pred_at_z_proposal=y_pred_at_z_proposal_mb,
            var_pred_at_z_proposal=var_pred_at_z_proposal_mb,
            log_prob_prior_at_z_proposal=log_prob_prior_at_z_proposal_mb,
            log_prob_proposal_at_z_proposal=log_prob_proposal_at_z_proposal_mb,
        )

    return lmlhd_iw_mc


def _compute_elbo_in_batches(
    y: np.ndarray,
    batch_size: int,
    y_pred_at_z_proposal: np.ndarray = None,
    var_pred_at_z_proposal: np.ndarray = None,
    log_prob_prior_at_z_proposal: np.ndarray = None,
    log_prob_proposal_at_z_proposal: np.ndarray = None,
):
    # check input data
    n_samples = y_pred_at_z_proposal.shape[0]
    n_tasks = y_pred_at_z_proposal.shape[1]
    n_points = y_pred_at_z_proposal.shape[2]
    d_y = y_pred_at_z_proposal.shape[3]
    task_ids = np.arange(n_tasks)
    assert y.shape == (n_tasks, n_points, d_y)
    assert y_pred_at_z_proposal.shape == (n_samples, n_tasks, n_points, d_y)
    assert var_pred_at_z_proposal.shape == (n_samples, n_tasks, n_points, d_y)
    assert log_prob_prior_at_z_proposal.shape == (n_samples, n_tasks)
    assert log_prob_proposal_at_z_proposal.shape == (n_samples, n_tasks)
    assert task_ids.shape == (n_tasks,)

    # compute number of minibatches
    n_mb = int(np.ceil(n_tasks / batch_size))

    # compute metrics
    elbo = np.zeros((n_tasks,))
    for (
        y_mb,
        y_pred_at_z_proposal_mb,
        var_pred_at_z_proposal_mb,
        log_prob_prior_at_z_proposal_mb,
        log_prob_proposal_at_z_proposal_mb,
        task_ids_mb,
    ) in tqdm(
        zip(  # minibatch over tasks
            np.array_split(y, n_mb, axis=0),
            np.array_split(y_pred_at_z_proposal, n_mb, axis=1),
            np.array_split(var_pred_at_z_proposal, n_mb, axis=1),
            np.array_split(log_prob_prior_at_z_proposal, n_mb, axis=1),
            np.array_split(log_prob_proposal_at_z_proposal, n_mb, axis=1),
            np.array_split(task_ids, n_mb, axis=0),
        ),
        desc="Computing ELBO...",
        total=n_mb,
    ):
        elbo[task_ids_mb] = compute_elbo(
            y=y_mb,
            y_pred_at_z_proposal=y_pred_at_z_proposal_mb,
            var_pred_at_z_proposal=var_pred_at_z_proposal_mb,
            log_prob_prior_at_z_proposal=log_prob_prior_at_z_proposal_mb,
            log_prob_proposal_at_z_proposal=log_prob_proposal_at_z_proposal_mb,
        )

    return elbo


def eval_one_model_with_nonparametric_latent_distribution(
    benchmark: MetaLearningBenchmark,
    model: MetaLearningLVM,
    context_sizes: Tuple[int],
    n_samples: int,
    n_epochs_adapt: Optional[int] = None,
    batch_size_eval: Optional[int] = 16,
):
    # prepare data
    x_all_ctx, y_all_ctx = collate_benchmark(benchmark)
    x_all_tgt, y_all_tgt = collate_benchmark(benchmark)
    n_tasks = x_all_ctx.shape[0]
    n_points = x_all_ctx.shape[1]
    d_x = x_all_ctx.shape[2]
    d_y = y_all_ctx.shape[2]
    assert x_all_ctx.shape == (n_tasks, n_points, d_x)
    assert x_all_tgt.shape == (n_tasks, n_points, d_x)
    assert y_all_ctx.shape == (n_tasks, n_points, d_y)
    assert y_all_tgt.shape == (n_tasks, n_points, d_y)

    # compute metrics over context size
    dfs = []
    for n_ctx in context_sizes:
        print(f"\n**** Computing Metrics for n_ctx = {n_ctx:d} ****")
        # choose context set
        x_ctx = x_all_ctx[:, :n_ctx, :]
        y_ctx = y_all_ctx[:, :n_ctx, :]

        # adapt model to current context set and sample z
        print(f" * Computing Conditional Prior Distribution...")
        model.adapt(x=x_ctx, y=y_ctx, n_epochs=n_epochs_adapt)
        z_prior = model.sample_z(n_samples)
        assert z_prior.shape == (n_samples, n_tasks, model._d_z)

        # compute predictions for current context set
        print(f" * Computing Predictions at Samples from Conditional Prior...")
        y_pred_at_z_prior, var_pred_at_z_prior = _compute_predictions_at_z_in_batches(
            model=model,
            x=x_all_tgt,
            z=z_prior,
            d_y=d_y,
            batch_size=batch_size_eval,
        )

        # compute metrics for current context set
        print(f" * Computing Metrics...")
        mse = _compute_mse_in_batches(
            y=y_all_tgt,
            y_pred_at_z_prior=y_pred_at_z_prior,
            batch_size=batch_size_eval,
        )
        lmlhd_naive_mc = _compute_lmlhd_naive_mc_in_batches(
            y=y_all_tgt,
            y_pred_at_z_prior=y_pred_at_z_prior,
            var_pred_at_z_prior=var_pred_at_z_prior,
            batch_size=batch_size_eval,
        )

        # store metrics in DataFrames
        df_mse = pd.DataFrame(
            {
                "n_ctx": n_ctx,
                "metric_type": "MSE",
                "metric_value": mse,
            }
        )
        df_lmlhd_naive_mc = pd.DataFrame(
            {
                "n_ctx": n_ctx,
                "metric_type": "LMLHD_NAIVE_MC",
                "metric_value": lmlhd_naive_mc,
            }
        )
        dfs.append(df_mse)
        dfs.append(df_lmlhd_naive_mc)

    # combine into one DataFrame
    metrics = pd.concat(dfs, ignore_index=True)
    return metrics


def eval_one_model_with_parametric_latent_distribution(
    benchmark: MetaLearningBenchmark,
    model: MetaLearningLVMParametric,
    context_sizes: Tuple[int],
    n_samples: int,
    context_size_proposal: int,
    n_epochs_adapt: Optional[int] = None,
    batch_size_eval: Optional[int] = 16,
):
    # prepare data
    x_all_ctx, y_all_ctx = collate_benchmark(benchmark)
    x_all_tgt, y_all_tgt = collate_benchmark(benchmark)
    n_tasks = x_all_ctx.shape[0]
    n_points = x_all_ctx.shape[1]
    d_x = x_all_ctx.shape[2]
    d_y = y_all_ctx.shape[2]
    assert x_all_ctx.shape == (n_tasks, n_points, d_x)
    assert x_all_tgt.shape == (n_tasks, n_points, d_x)
    assert y_all_ctx.shape == (n_tasks, n_points, d_y)
    assert y_all_tgt.shape == (n_tasks, n_points, d_y)

    # compute proposal distribution (for MC-IW) = approximate posterior (for ELBO)
    print(f"\n**** Preparing Proposal Distribution ****")
    print(f" * Computing Proposal Distribution (n_ctx = {context_size_proposal:d})...")
    x_ctx_proposal = x_all_ctx[:, :context_size_proposal, :]
    y_ctx_proposal = y_all_ctx[:, :context_size_proposal, :]
    model.adapt(x=x_ctx_proposal, y=y_ctx_proposal, n_epochs=n_epochs_adapt)
    proposal_distribution = model.latent_distribution()
    z_proposal = proposal_distribution.sample(n_samples)
    log_prob_proposal_at_z_proposal = proposal_distribution.log_prob(z_proposal)
    assert z_proposal.shape == (n_samples, n_tasks, model._d_z)
    assert log_prob_proposal_at_z_proposal.shape == (n_samples, n_tasks)

    # compute predictions at samples from proposal distribtuion
    print(f" * Computing Predictions at Samples from Proposal Distribution...")
    y_pred_at_z_proposal, var_pred_at_z_proposal = _compute_predictions_at_z_in_batches(
        model=model,
        x=x_all_tgt,
        z=z_proposal,
        d_y=d_y,
        batch_size=batch_size_eval,
    )

    # compute metrics over context size
    dfs = []
    for n_ctx in context_sizes:
        print(f"\n**** Computing Metrics for n_ctx = {n_ctx:d} ****")
        # choose context set
        x_ctx = x_all_ctx[:, :n_ctx, :]
        y_ctx = y_all_ctx[:, :n_ctx, :]

        # adapt model to current context set and sample z
        print(f" * Computing Conditional Prior Distribution...")
        model.adapt(x=x_ctx, y=y_ctx, n_epochs=n_epochs_adapt)
        prior = model.latent_distribution()
        z_prior = prior.sample(n_samples)
        log_prob_prior_at_z_proposal = prior.log_prob(z_proposal)

        # check shapes
        assert z_prior.shape == (n_samples, n_tasks, model._d_z)
        assert log_prob_prior_at_z_proposal.shape == (n_samples, n_tasks)

        # compute predictions for current context set
        print(f" * Computing Predictions at Samples from Conditional Prior...")
        y_pred_at_z_prior, var_pred_at_z_prior = _compute_predictions_at_z_in_batches(
            model=model,
            x=x_all_tgt,
            z=z_prior,
            d_y=d_y,
            batch_size=batch_size_eval,
        )

        # compute metrics for current context set
        print(f" * Computing Metrics...")
        mse = _compute_mse_in_batches(
            y=y_all_tgt,
            y_pred_at_z_prior=y_pred_at_z_prior,
            batch_size=batch_size_eval,
        )
        lmlhd_naive_mc = _compute_lmlhd_naive_mc_in_batches(
            y=y_all_tgt,
            y_pred_at_z_prior=y_pred_at_z_prior,
            var_pred_at_z_prior=var_pred_at_z_prior,
            batch_size=batch_size_eval,
        )
        lmlhd_iw_mc = _compute_lmlhd_iw_mc_in_batches(
            y=y_all_tgt,
            y_pred_at_z_proposal=y_pred_at_z_proposal,
            var_pred_at_z_proposal=var_pred_at_z_proposal,
            log_prob_prior_at_z_proposal=log_prob_prior_at_z_proposal,
            log_prob_proposal_at_z_proposal=log_prob_proposal_at_z_proposal,
            batch_size=batch_size_eval,
        )
        elbo = _compute_elbo_in_batches(
            y=y_all_tgt,
            y_pred_at_z_proposal=y_pred_at_z_proposal,
            var_pred_at_z_proposal=var_pred_at_z_proposal,
            log_prob_prior_at_z_proposal=log_prob_prior_at_z_proposal,
            log_prob_proposal_at_z_proposal=log_prob_proposal_at_z_proposal,
            batch_size=batch_size_eval,
        )

        # store metrics in DataFrames
        df_mse = pd.DataFrame(
            {
                "n_ctx": n_ctx,
                "metric_type": "MSE",
                "metric_value": mse,
            }
        )
        df_lmlhd_naive_mc = pd.DataFrame(
            {
                "n_ctx": n_ctx,
                "metric_type": "LMLHD_NAIVE_MC",
                "metric_value": lmlhd_naive_mc,
            }
        )
        df_lmlhd_iw_mc = pd.DataFrame(
            {
                "n_ctx": n_ctx,
                "n_ctx_proposal": context_size_proposal,
                "metric_type": "LMLHD_IW_MC",
                "metric_value": lmlhd_iw_mc,
            }
        )
        df_elbo = pd.DataFrame(
            {
                "n_ctx": n_ctx,
                "n_ctx_proposal": context_size_proposal,
                "metric_type": "ELBO",
                "metric_value": elbo,
            }
        )
        dfs.append(df_mse)
        dfs.append(df_lmlhd_naive_mc)
        dfs.append(df_lmlhd_iw_mc)
        dfs.append(df_elbo)

    # combine into one DataFrame
    metrics = pd.concat(dfs, ignore_index=True)
    return metrics


def eval_one_model(
    benchmark: MetaLearningBenchmark,
    model: MetaLearningLVM,
    context_sizes: Tuple[int],
    n_samples: int,
    context_size_proposal: int,
    n_epochs_adapt: Optional[int] = None,
    batch_size_eval: Optional[int] = 16,
):
    if isinstance(model, MetaLearningLVMParametric):
        return eval_one_model_with_parametric_latent_distribution(
            benchmark=benchmark,
            model=model,
            context_sizes=context_sizes,
            n_samples=n_samples,
            context_size_proposal=context_size_proposal,
            n_epochs_adapt=n_epochs_adapt,
            batch_size_eval=batch_size_eval,
        )
    elif isinstance(model, MetaLearningLVM):
        return eval_one_model_with_nonparametric_latent_distribution(
            benchmark=benchmark,
            model=model,
            context_sizes=context_sizes,
            n_samples=n_samples,
            n_epochs_adapt=n_epochs_adapt,
            batch_size_eval=batch_size_eval,
        )
    else:
        raise NotImplementedError
