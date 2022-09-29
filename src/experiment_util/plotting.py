from tkinter import Image
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from eval_util.util import log_marginal_likelihood_naive_mc
from eval_util.util import mse2 as compute_mse
from matplotlib import pyplot as plt
from metalearning_benchmarks import MetaLearningBenchmark

from experiment_util.metalearning_model import MetaLearningLVM


def _predictions_plotting_available(benchmark: MetaLearningBenchmark):
    if benchmark.d_x == 1 and benchmark.d_y == 1:
        return True

    print(
        f"Plotting not available for "
        + f"d_x = {benchmark.d_x:d}, d_y = {benchmark.d_y:d}, "
        f"Skipping..."
    )
    return False


def plot_function_samples_for_one_task(
    ax,
    model: MetaLearningLVM,
    x_lim: Tuple[float, float],
    x_ctx: np.ndarray,
    y_ctx: np.ndarray,
    x_tgt: np.ndarray,
    y_tgt: np.ndarray,
    z: np.ndarray,
    task_id: int,
    plot_std_y: bool,
):
    """
    Plot function samples on one task.
    model.adapt() is **not** called to allow doing this externally.
    """

    # check input
    n_ctx = x_ctx.shape[0]
    n_tgt = x_tgt.shape[0]
    n_samples = z.shape[0]
    d_x = x_ctx.shape[1]
    d_y = y_ctx.shape[1]
    d_z = z.shape[1]
    assert d_x == 1
    assert d_y == 1
    assert x_ctx.shape == (n_ctx, d_x)
    assert y_ctx.shape == (n_ctx, d_y)
    assert x_tgt.shape == (n_tgt, d_x)
    assert y_tgt.shape == (n_tgt, d_y)
    assert z.shape == (n_samples, d_z)
    x = np.concatenate([x_ctx, x_tgt], axis=0)
    y = np.concatenate([y_ctx, y_tgt], axis=0)
    task_ids = np.array([task_id])

    # obtain predictions on x_plot
    n_plot = 100
    x_plot = np.linspace(x_lim[0, 0], x_lim[0, 1], n_plot)[:, None]
    y_pred_plot, var_pred_plot = model.predict_at_z(
        x=x_plot[None], z=z[:, None], task_ids=task_ids
    )

    # plot predictions
    assert x_plot.shape == (n_plot, d_x)
    assert y_pred_plot.shape == (n_samples, 1, n_plot, d_y)
    assert var_pred_plot.shape == (n_samples, 1, n_plot, d_y)
    x_plot = x_plot.squeeze(1)
    y_pred_plot, var_pred_plot = y_pred_plot.squeeze((1, 3)), var_pred_plot.squeeze((1, 3))  # fmt: skip
    x_plot = np.broadcast_to(x_plot, y_pred_plot.shape)
    ax.plot(x_plot.T, y_pred_plot.T, color="C0", alpha=0.3, zorder=1)
    if plot_std_y:
        std_y_pred_plot = np.sqrt(var_pred_plot)
        low, high = y_pred_plot - std_y_pred_plot, y_pred_plot + std_y_pred_plot
        for s in range(n_samples):
            ax.fill_between(x_plot[s], low[s], high[s], color="C0", alpha=0.1, zorder=1)

    # plot context and target sets
    ax.scatter(x_ctx, y_ctx, marker="x", color="C1", zorder=3)
    ax.scatter(x_tgt, y_tgt, marker=".", color="C2", zorder=2)

    # compute and print metrics (predict again, now at x instead of x_plot)
    y_pred, var_pred = model.predict_at_z(x=x[None], z=z[:, None], task_ids=task_ids)
    mse = compute_mse(
        y=y[None],
        y_pred=y_pred,
    )
    lmlhd = log_marginal_likelihood_naive_mc(
        y=y[None],
        y_pred_at_z_prior=y_pred,
        var_pred_at_z_prior=var_pred,
    )
    text = f"{'LMLHD':5} = {lmlhd.squeeze():+.2f}\n{'MSE':5} = {mse.squeeze():+.2f}"
    ax.text(
        x=0.05,
        y=0.05,
        transform=ax.transAxes,
        s=text,
        font="monospace",
        fontsize="x-small",
    )

    # labelling
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_function_samples(
    model: MetaLearningLVM,
    benchmark: MetaLearningBenchmark,
    task_ids: Tuple[int],
    context_sizes: Tuple[int],
    n_samples: int,
    n_epochs_adapt: int,
    plot_std_y: bool,
):
    assert not benchmark.is_dynamical_system

    # prepare inputs
    n_tasks = len(task_ids)
    n_ctx_sets = len(context_sizes)

    # prepare plot
    x_size = 4
    golden = (1 + np.sqrt(5)) / 2
    fig, axes = plt.subplots(
        nrows=n_ctx_sets,
        ncols=n_tasks,
        figsize=(n_tasks * x_size, n_tasks * x_size / golden),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    # collate data
    x = np.zeros((n_tasks, benchmark.n_datapoints_per_task, benchmark.d_x))
    y = np.zeros((n_tasks, benchmark.n_datapoints_per_task, benchmark.d_y))
    for i, task_id in enumerate(task_ids):
        task = benchmark.get_task_by_index(task_id)
        x[i], y[i] = task.x, task.y

    # plot
    for j, n_ctx in enumerate(context_sizes):
        # split tasks
        x_ctx, y_ctx = x[:, :n_ctx], y[:, :n_ctx]
        x_tgt, y_tgt = x[:, n_ctx:], y[:, n_ctx:]

        # adapt to all tasks at once and sample latent representations
        model.adapt(x=x_ctx, y=y_ctx, n_epochs=n_epochs_adapt)
        z = model.sample_z(n_samples=n_samples)

        # plot for each task
        for i, task_id in enumerate(task_ids):
            # initialize
            ax = axes[j, i]
            task_id = task_ids[i]

            # plot
            plot_function_samples_for_one_task(
                ax=ax,
                model=model,
                x_lim=benchmark.x_bounds,
                x_ctx=x_ctx[i],
                y_ctx=y_ctx[i],
                x_tgt=x_tgt[i],
                y_tgt=y_tgt[i],
                z=z[:, i],
                task_id=task_id,
                plot_std_y=plot_std_y,
            )

            # label
            if j == 0:
                ax.set_title(f"Task-ID = {task_id:03d}")
            ax.label_outer()

    fig.tight_layout()
    return fig


def plot_predictions(
    model: MetaLearningLVM,
    benchmark: MetaLearningBenchmark,
    task_ids: Tuple[int],
    context_sizes: Tuple[int],
    n_samples: int,
    n_epochs_adapt: int,
    plot_std_y: bool,
):
    if not _predictions_plotting_available(benchmark):
        return

    # plot standard regression predictions
    return plot_function_samples(
        model=model,
        benchmark=benchmark,
        task_ids=task_ids,
        context_sizes=context_sizes,
        n_samples=n_samples,
        n_epochs_adapt=n_epochs_adapt,
        plot_std_y=plot_std_y,
    )


def set_log_yscale_for_facet_grid(g, log_scale_labels: List[str]):
    for label, ax in g.axes_dict.items():
        if label in log_scale_labels:
            ax.set_yscale("log")


def set_sharey_for_facet_grid(g, sharey_labels: List):
    # determine one axis with which y-axes are shared
    share_ax = None
    for label in sharey_labels:
        if label in g.axes_dict:
            share_ax = g.axes_dict[sharey_labels[0]]
    if share_ax is None:  # none of the axes is present
        return

    # share the y-axes
    for label, ax in g.axes_dict.items():
        if label in sharey_labels:
            ax.sharey(share_ax)

    # toggle autoscaling to adjust share_ax's scaling to include all shared axes
    share_ax.autoscale()


def plot_metrics_for_one_run(
    metrics: pd.DataFrame,
    kind: str = "box",
    task_aggregate_op: Optional[str] = None,
):
    """
    Plots metrics for a single run.
    """
    assert kind in ["box", "line"]

    # plot metrics
    golden = (1 + np.sqrt(5)) / 2
    if kind == "box":
        assert task_aggregate_op is None
        g = sns.catplot(
            data=metrics,
            x="n_ctx",
            y="metric_value",
            col="metric_type",
            kind="box",
            height=4 / golden,
            aspect=golden,
            sharey=False,
            showfliers=False,
        )
    else:
        assert task_aggregate_op in ["median", "mean"]
        g = sns.relplot(
            data=metrics,
            x="n_ctx",
            y="metric_value",
            col="metric_type",
            kind="line",
            marker="o",
            height=4 / golden,
            aspect=golden,
            estimator=np.median if task_aggregate_op == "median" else np.mean,
            facet_kws={"sharey": False},
        )

    # finalize plot
    set_log_yscale_for_facet_grid(g=g, log_scale_labels=["MSE"])
    set_sharey_for_facet_grid(
        g=g, sharey_labels=["LMLHD_NAIVE_MC", "LMLHD_IW_MC", "ELBO"]
    )
    g.fig.tight_layout()
