from itertools import product
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from metalearning_benchmarks import MetaLearningBenchmark


def assert_shape(t: tf.Tensor, shape: Tuple[int]):
    if tf.executing_eagerly():
        assert tf.reduce_all(tf.shape(t) == shape)


def visualize_latent_space_structure(
    np_model,
    x_bounds: List[float],
    z_bounds: List[float],
    n_ticks: int = 10,
    n_samples: int = 10,
):
    ## prepare plot
    x_size = 16
    fig, axes = plt.subplots(
        nrows=n_ticks, ncols=n_ticks, figsize=(x_size, x_size), sharex=True, sharey=True
    )

    ## prepare z-grid
    z1 = np.linspace(z_bounds[0][0], z_bounds[0][1], n_ticks)
    z2 = np.linspace(z_bounds[1][0], z_bounds[1][1], n_ticks)
    delta_z1 = z_bounds[0][1] - z_bounds[0][0]
    delta_z2 = z_bounds[1][1] - z_bounds[1][0]
    assert (
        delta_z2 == delta_z1
    )  # otherwise sampling z requires different stddevs for each ax

    ## prepare x_plt
    x_min = x_bounds[0, 0]
    x_max = x_bounds[0, 1]
    x_plt_min = x_min - 0.25 * (x_max - x_min)
    x_plt_max = x_max + 0.25 * (x_max - x_min)
    n_plt = 128
    x_plt = np.linspace(x_plt_min, x_plt_max, n_plt)
    x_plt = x_plt[None, :, None]

    ## plot
    for (i, j) in product(range(n_ticks), range(n_ticks)):
        ax = axes[-(j + 1), i]  # z1 horizontal, z2 vertical
        # prepare current z
        cur_z1 = z1[i]
        cur_z2 = z2[j]
        cur_z = np.array([cur_z1, cur_z2])
        cur_z = np.random.normal(
            loc=cur_z, scale=delta_z1 / n_ticks, size=(n_samples, 1, 2)
        )
        # obtain prediction
        y_pred, _ = np_model.predict_at_z(x=x_plt, z=cur_z)
        # plot
        ax.plot(np.squeeze(x_plt), np.squeeze(y_pred).T, color="b", alpha=0.3)
        ax.tick_params(
            which="both", bottom=False, labelbottom=False, labelleft=False, left=False
        )
        if i == 0:
            ax.set_ylabel(f"{cur_z2:.1f}")
        if j == 0:
            ax.set_xlabel(f"{cur_z1:.1f}")
        # ax.grid(False)

    fig.tight_layout(pad=0)
    return fig


def plot_distributions(
    axes,
    np_model,
    x: tf.Tensor,
    y: tf.Tensor,
    ctx_mask: tf.Tensor,
    task_id: tf.Tensor,
    z_bounds: List = None,
    logspace: bool = False,
    plot_prior: bool = False,
):
    def plot_gaussian_ellipse(ax, mean, scale_tril, color, linestyle="-", alpha=1.0):
        n_plot = 50
        evals, evecs = np.linalg.eig(scale_tril @ scale_tril.T)
        theta = np.linspace(0, 2 * np.pi, n_plot)
        ellipsis = (np.sqrt(evals[None, :]) * evecs) @ [np.sin(theta), np.cos(theta)]
        ellipsis = ellipsis + mean[:, None]
        ax.plot(
            ellipsis[0, :],
            ellipsis[1, :],
            color=color,
            linestyle=linestyle,
            alpha=alpha,
        )

    ## condition np model on current task
    np_model.reset_batch(task_ids=task_id)

    ## plot all posterior GMMs
    ax = axes["posterior_gmms"]
    means = np_model.post_loc.numpy()
    means = np.reshape(means, (-1, np_model.d_z))  # flatten task an components
    ax.scatter(x=means[:, 0], y=means[:, 1], color="gainsboro", alpha=0.1)
    ax.set_title("Aggregated Posterior")
    ax.set_xlabel("z_1")
    ax.set_ylabel("z_2")
    ax.grid(visible=False)

    ## plot target density and posterior GMM of current task
    ax = axes["target_density"]
    colors = []
    if plot_prior:
        for k in range(np_model.prior_n_comps):
            color = "w"
            scale_tril = np_model.prior.scale_tril[0, k].numpy()
            mean = np_model.prior.loc[0, k].numpy()
            ax.scatter(x=mean[0], y=mean[1], color=color, alpha=1.0)
            axes["posterior_gmms"].scatter(x=mean[0], y=mean[1], color=color, alpha=1.0)
            plot_gaussian_ellipse(
                ax=ax,
                mean=mean,
                scale_tril=scale_tril,
                color=color,
                alpha=0.2,
                linestyle="--",
            )
            plot_gaussian_ellipse(
                ax=axes["posterior_gmms"],
                mean=mean,
                scale_tril=scale_tril,
                color=color,
                alpha=0.2,
                linestyle="--",
            )
    for k in range(np_model.post_n_comps):
        color = next(ax._get_lines.prop_cycler)["color"]
        colors.append(color)
        scale_tril = np_model.post_batch.scale_tril[0, k].numpy()
        mean = np_model.post_batch.loc[0, k].numpy()
        ax.scatter(x=mean[0], y=mean[1], color=color)
        axes["posterior_gmms"].scatter(x=mean[0], y=mean[1], color=color)
        plot_gaussian_ellipse(
            ax=ax, mean=mean, scale_tril=scale_tril, alpha=1.0, color=color
        )
        plot_gaussian_ellipse(
            ax=axes["posterior_gmms"],
            mean=mean,
            scale_tril=scale_tril,
            alpha=1.0,
            color=color,
        )

    ## prepare z-grid
    n_ticks = 500
    if z_bounds is None:
        # determine z_bounds from axis bounds (contains full GMM ellipses)
        z1_min_pre = ax.get_xlim()[0]
        z1_max_pre = ax.get_xlim()[1]
        z2_min_pre = ax.get_ylim()[0]
        z2_max_pre = ax.get_ylim()[1]
        z1_min = z1_min_pre - 0.33 * (z1_max_pre - z1_min_pre)
        z1_max = z1_max_pre + 0.33 * (z1_max_pre - z1_min_pre)
        z2_min = z2_min_pre - 0.33 * (z2_max_pre - z2_min_pre)
        z2_max = z2_max_pre + 0.33 * (z2_max_pre - z2_min_pre)
        z_bounds = [[z1_min, z1_max], [z2_min, z2_max]]
    z1 = np.linspace(z_bounds[0][0], z_bounds[0][1], n_ticks)
    z2 = np.linspace(z_bounds[1][0], z_bounds[1][1], n_ticks)
    zz1, zz2 = np.meshgrid(z1, z2)
    z_grid = np.vstack([zz1.ravel(), zz2.ravel()]).T
    z_grid = tf.convert_to_tensor(z_grid, dtype=np.float32)
    z_grid = z_grid[:, None, :]

    ## evaluate distributions on z-grid
    p_tgt = np_model._log_unnormalized_posterior_density(
        x=x, y=y, z=z_grid, ctx_mask=ctx_mask
    ).numpy()
    if not logspace:
        p_tgt = np.exp(p_tgt)
    gmm_post_weights = np.exp(np_model.post_batch.log_w.numpy())
    gmm_prior_weights = np.exp(np_model.prior.log_w.numpy())

    ## plot target density
    ax.contourf(
        zz1,
        zz2,
        p_tgt.reshape(n_ticks, n_ticks),
        levels=100,
        zorder=-100,
        cmap="viridis",
    )
    axes["posterior_gmms"].contourf(
        zz1,
        zz2,
        p_tgt.reshape(n_ticks, n_ticks),
        levels=100,
        zorder=-100,
        cmap="viridis",
    )

    ## adjust axes
    # ax.axis("scaled")
    ax.set_title("Target density")
    # if custom z_bounds were given, adjust axis limits
    ax.set_xlim(z_bounds[0])
    ax.set_ylim(z_bounds[1])
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.grid(visible=False)

    ## plot weights
    ax = axes["prior_mixture_weights"]
    if plot_prior:
        ax.pie(
            gmm_prior_weights[0],
            labels=[f"{w*100:.2f}%" for w in gmm_prior_weights[0]],
            colors=colors,
        )
        ax.axis("scaled")
        ax.set_title("Prior weights")
    else:
        ax.remove()
    ax = axes["posterior_mixture_weights"]
    ax.pie(
        gmm_post_weights[0],
        labels=[f"{w*100:.2f}%" for w in gmm_post_weights[0]],
        colors=colors,
    )
    ax.axis("scaled")
    ax.set_title("Posterior weights")

    # return some data for further evaluation
    data = {"z_grid": z_grid.numpy(), "p_tgt": p_tgt}
    return data


def plot_predictions(
    axes,
    np_model,
    x: tf.Tensor,
    y: tf.Tensor,
    ctx_mask: tf.Tensor,
    x_test: tf.Tensor,
    y_test: tf.Tensor,
    task_id: tf.Tensor,
    benchmark: MetaLearningBenchmark,
    n_samples: int,
    plot_prior: bool,
    z_explicit: Optional[np.ndarray] = None,
):
    # prepare axis
    ax = axes["predictions"]

    # prepare x_plt
    x_min = benchmark.x_bounds[0, 0]
    x_max = benchmark.x_bounds[0, 1]
    x_plt_min = x_min - 0.05 * (x_max - x_min)
    x_plt_max = x_max + 0.05 * (x_max - x_min)
    n_plt = 128
    x_plt = np.linspace(x_plt_min, x_plt_max, n_plt)
    x_plt = tf.convert_to_tensor(x_plt, dtype=tf.float32)
    x_plt = x_plt[None, :, None]

    # condition NP model on current task
    np_model.reset_batch(task_ids=task_id)

    ## evaluate predictions
    # posterior
    mu_post, _ = np_model.predict(
        x=x_plt, n_samples=n_samples, sample_from="approximate_posterior"
    )
    assert mu_post.shape == (n_samples, 1, n_plt, 1)
    mu_post = np.squeeze(mu_post, axis=(1, 3))  # (n_samples, n_plt)
    # prior
    if plot_prior:
        mu_prior, _ = np_model.predict(
            x=x_plt, n_samples=n_samples, sample_from="prior"
        )
        assert mu_prior.shape == (n_samples, 1, n_plt, 1)
        mu_prior = np.squeeze(mu_prior, axis=(1, 3))  # (n_samples, n_plt)
    # explicit z
    if z_explicit is not None:
        # prepare z
        n_explicit_samples = z_explicit.shape[0]
        assert z_explicit.shape == (n_explicit_samples, 2)
        z_explicit = z_explicit[:, None, :]
        # obtain predictions
        mu_explicit, _ = np_model.predict_at_z(x=x_plt, z=z_explicit)
        assert mu_explicit.shape == (n_explicit_samples, 1, n_plt, 1)
        mu_explicit = np.squeeze(mu_explicit, axis=(1, 3))

    ## plot predictions
    x_plt = np.squeeze(x_plt)  # (n_plt,)
    # prior
    if plot_prior:
        ax.plot(x_plt, mu_prior.T, color="C1", alpha=0.1, label="prior")
    # posterior
    ax.plot(x_plt, mu_post.T, color="C0", alpha=0.1, label="posterior")
    # explicit samples
    if z_explicit is not None:
        ax.plot(x_plt, mu_explicit.T, color="C2", alpha=0.5, label="explicit")
    ax.scatter(x_test, y_test, marker=".", s=25, color="k", zorder=1000)
    ax.scatter(x, y, marker="x", s=100, color="k", zorder=1000)
    ax.scatter(x[ctx_mask], y[ctx_mask], marker="x", s=250, color="C3", zorder=1000)
    ax.set_title("Predictions")


def plot_per_sample_metrics(
    axes,
    np_model,
    x: tf.Tensor,
    y: tf.Tensor,
    x_test: tf.Tensor,
    y_test: tf.Tensor,
    task_id: tf.Tensor,
    n_samples: int,
    plot_prior: bool,
    z_explicit: np.ndarray = None,
    show_outliers: bool = True,
):
    # condition NP model on current task
    np_model.reset_batch(task_ids=task_id)

    # prepare data
    x_all = np.concatenate((x, x_test), axis=1)
    y_all = np.concatenate((y, y_test), axis=1)

    ## evaluate predictions
    # posterior
    mu_post, var_post = np_model.predict(
        x=x_all, n_samples=n_samples, sample_from="approximate_posterior"
    )
    # prior
    if plot_prior:
        mu_prior, var_prior = np_model.predict(
            x=x_all, n_samples=n_samples, sample_from="prior"
        )
    # explicit z
    if z_explicit is not None:
        # prepare z
        n_explicit_samples = z_explicit.shape[0]
        assert z_explicit.shape == (n_explicit_samples, 2)
        z_explicit = z_explicit[:, None, :]
        # obtain predictions
        mu_explicit, var_explicit = np_model.predict_at_z(x=x_all, z=z_explicit)

    # compute metrics
    mse_post = (
        compute_per_datapoint_mse(y_true=y_all, y_pred=mu_post).sum(axis=-1).squeeze()
    )
    llhd_post = (
        compute_per_datapoint_log_likelihood_mc(
            y_true=y_all,
            y_pred=mu_post,
            sigma_pred=np.sqrt(var_post),
        )
        .sum(axis=-1)
        .squeeze()
    )
    lmlhd_post = compute_log_marginal_likelihood_mc(
        y_true=y_all, y_pred=mu_post, sigma_pred=np.sqrt(var_post)
    )
    if plot_prior:
        mse_prior = (
            compute_per_datapoint_mse(y_true=y_all, y_pred=mu_prior)
            .sum(axis=-1)
            .squeeze()
        )
        llhd_prior = (
            compute_per_datapoint_log_likelihood_mc(
                y_true=y_all,
                y_pred=mu_prior,
                sigma_pred=np.sqrt(var_prior),
            )
            .sum(axis=-1)
            .squeeze()
        )
        lmlhd_prior = compute_log_marginal_likelihood_mc(
            y_true=y_all, y_pred=mu_prior, sigma_pred=np.sqrt(var_prior)
        )
    if z_explicit is not None:
        mse_explicit = (
            compute_per_datapoint_mse(y_true=y_all, y_pred=mu_explicit)
            .sum(axis=-1)
            .squeeze()
        )
        llhd_explicit = (
            compute_per_datapoint_log_likelihood_mc(
                y_true=y_all,
                y_pred=mu_explicit,
                sigma_pred=np.sqrt(var_explicit),
            )
            .sum(axis=-1)
            .squeeze()
        )
        lmlhd_explicit = compute_log_marginal_likelihood_mc(
            y_true=y_all, y_pred=mu_explicit, sigma_pred=np.sqrt(var_explicit)
        )

    # create boxplot
    data_mse = pd.DataFrame({"metric_name": "MSE Posterior", "metric_value": mse_post})
    data_lmlhd = pd.DataFrame(
        {"metric_name": "LLHD Posterior", "metric_value": llhd_post}
    )
    if plot_prior:
        data_mse = pd.concat(
            [
                data_mse,
                pd.DataFrame({"metric_name": "MSE Prior", "metric_value": mse_prior}),
            ],
            ignore_index=True,
        )
        data_lmlhd = pd.concat(
            [
                data_lmlhd,
                pd.DataFrame({"metric_name": "LLHD Prior", "metric_value": llhd_prior}),
            ],
            ignore_index=True,
        )
    if z_explicit is not None:
        data_mse = pd.concat(
            [
                data_mse,
                pd.DataFrame(
                    {"metric_name": "MSE Max LL", "metric_value": mse_explicit}
                ),
            ],
            ignore_index=True,
        )
        data_lmlhd = pd.concat(
            [
                data_lmlhd,
                pd.DataFrame(
                    {"metric_name": "LLHD Max LL", "metric_value": llhd_explicit}
                ),
            ],
            ignore_index=True,
        )
    b = sns.boxplot(
        y="metric_name",
        x="metric_value",
        data=data_mse,
        ax=axes["mse"],
        showfliers=show_outliers,
    )
    b.set(xlabel=None, ylabel=None)
    b = sns.boxplot(
        y="metric_name",
        x="metric_value",
        data=data_lmlhd,
        ax=axes["llhd"],
        showfliers=show_outliers,
    )
    b.set(xlabel=None, ylabel=None)

    # add log marginal likelihood values
    ax = axes["llhd"]
    text = ""
    text = text + f"LMLHD Posterior = {lmlhd_post:.2f}\n"
    if plot_prior:
        text = text + f"LMLHD Prior = {lmlhd_prior:.2f}\n"
    if z_explicit is not None:
        text = text + f"LMLHD Max LL = {lmlhd_explicit:.2f}"
    ax.text(x=0.05, y=0.65, transform=ax.transAxes, s=text, fontsize="x-small")
    ax.set_title("Per-sample metrics on current task")


def visualize_np(
    np_model,
    dataset,
    full_task_id,
    ctx_size,
    subtask_number=0,  # identifies which subtask with full_task_id, ctx_size is plotted
    plot_predictions_only=False,
    plot_prior_distribution=True,
    plot_distributions_in_logspace=False,
    n_samples=500,
    z_bounds=None,
    plot_prior_predictions=True,
    show_boxplot_outliers=True,
):
    ## prepare plot
    golden = (1 + 5**0.5) / 2
    x_size = 12
    if not plot_predictions_only:
        fig, axes = plt.subplot_mosaic(
            [
                [
                    "predictions",
                    "predictions",
                    "predictions",
                    "predictions",
                    "predictions",
                    "llhd",
                    "llhd",
                    "llhd",
                ],
                [
                    "predictions",
                    "predictions",
                    "predictions",
                    "predictions",
                    "predictions",
                    "mse",
                    "mse",
                    "mse",
                ],
                [
                    "target_density",
                    "target_density",
                    "target_density",
                    "posterior_gmms",
                    "posterior_gmms",
                    "posterior_gmms",
                    "prior_mixture_weights",
                    "prior_mixture_weights",
                ],
                [
                    "target_density",
                    "target_density",
                    "target_density",
                    "posterior_gmms",
                    "posterior_gmms",
                    "posterior_gmms",
                    "posterior_mixture_weights",
                    "posterior_mixture_weights",
                ],
            ],
            gridspec_kw={
                # "width_ratios": [1, 1, 1, 1, 1, 1],
                # "height_ratios": [1, 1, 1, 1],
            },
            figsize=(x_size, 0.75 * x_size),
        )
    else:
        fig, axes = plt.subplot_mosaic(
            [["predictions"]],
            figsize=(x_size, x_size / golden),
        )
    fig.suptitle(
        f"Task id = {full_task_id:03d}, Ctx size = {ctx_size:03d}, Subtask no. = {subtask_number:03d}"
    )

    ## gather data
    (
        x,
        y,
        ctx_mask,
        x_test,
        y_test,
        task_id,
    ) = dataset.get_subtasks_by_task_id_and_ctx_size(
        full_task_id=full_task_id, ctx_size=ctx_size
    )
    if subtask_number >= x.shape[0]:
        print(f"Subtask number {subtask_number:d} not available!")
        return None
    x, y, ctx_mask, x_test, y_test, task_id = (
        x[subtask_number : subtask_number + 1],
        y[subtask_number : subtask_number + 1],
        ctx_mask[subtask_number : subtask_number + 1],
        x_test[subtask_number : subtask_number + 1],
        y_test[subtask_number : subtask_number + 1],
        task_id[subtask_number : subtask_number + 1],
    )

    # plot distributions
    if not plot_predictions_only:
        data = plot_distributions(
            axes=axes,
            np_model=np_model,
            x=x,
            y=y,
            ctx_mask=ctx_mask,
            task_id=task_id,
            z_bounds=z_bounds,
            logspace=plot_distributions_in_logspace,
            plot_prior=plot_prior_distribution,
        )

    # plot predictions
    if not plot_predictions_only:
        n_explicit_samples = 10
        z_explicit_idx = np.argsort(data["p_tgt"][:, 0])[-n_explicit_samples:]
        z_explicit = data["z_grid"][z_explicit_idx][:, 0, :]
    else:
        z_explicit = None
    plot_predictions(
        axes=axes,
        np_model=np_model,
        x=x,
        y=y,
        ctx_mask=ctx_mask,
        x_test=x_test,
        y_test=y_test,
        task_id=task_id,
        benchmark=dataset.bm,
        n_samples=n_samples,
        plot_prior=plot_prior_predictions,
        z_explicit=z_explicit,
    )

    # plot metrics
    if not plot_predictions_only:
        plot_per_sample_metrics(
            axes=axes,
            np_model=np_model,
            x=x,
            y=y,
            x_test=x_test,
            y_test=y_test,
            task_id=task_id,
            n_samples=n_samples,
            plot_prior=plot_prior_predictions,
            z_explicit=z_explicit,
            show_outliers=show_boxplot_outliers,
        )

    fig.tight_layout()
    return fig


def plot_learning_curves(likelihood_losses, neg_mean_log_tgt_densities):
    # prepare plot
    golden = (1 + 5**0.5) / 2
    size = 8
    fig, ax = plt.subplots(figsize=(size, size / golden))
    # plot
    iters = np.arange(1, len(neg_mean_log_tgt_densities) + 1)
    ax.plot(iters, neg_mean_log_tgt_densities, label="avg. neg. log tgt. dens.")
    ax.plot(iters, likelihood_losses, label="likelihood loss")
    ax.set_ylim((-5.0, None))
    ax.legend()
    return fig


def plot_metrics(metrics: dict, show_outliers: bool = True):
    def boxplot_with_n_task(ax, data, x, y, showfliers):
        # https://python-graph-gallery.com/38-show-number-of-observation-on-boxplot
        # Calculate number of obs per group & median to position labels
        medians = data.groupby([x])[y].median().values
        n_task = data[x].value_counts()
        x_ticks = range(len(n_task))
        # Add it to the plot
        for tick in x_ticks:
            ax.text(
                x_ticks[tick],
                medians[tick] + 0.03,
                f"L = {n_task.iloc[tick]:d}",
                horizontalalignment="center",
                size="x-small",
                color="w",
                weight="semibold",
            )

        sns.boxplot(
            x=x,
            y=y,
            data=data,
            ax=ax,
            showfliers=showfliers,
        )

    ## prepare plot
    golden = (1 + 5**0.5) / 2
    size = 8
    fig, axes = plt.subplots(
        nrows=6, ncols=1, figsize=(size, 5 * size / golden), squeeze=False
    )

    ## plot metrics over context set size
    # MSE
    ax = axes[2, 0]
    ax.plot(metrics["n_ctx"], metrics["mse1"], label="MSE1")
    ax.plot(metrics["n_ctx"], metrics["mse2"], label="MSE2")
    ax.plot(metrics["n_ctx"], metrics["mse3"], label="MSE3")
    text = ""
    text = text + "\n" + f"AUC-MSE1: {metrics['mse1_auc']:.4f}"
    text = text + "\n" + f"AUC-MSE2: {metrics['mse2_auc']:.4f}"
    text = text + "\n" + f"AUC-MSE3: {metrics['mse3_auc']:.4f}"
    ax.text(x=0.25, y=0.75, transform=ax.transAxes, s=text)
    ax.set_xlabel("n_ctx")
    ax.legend()
    # LMLHD
    ax = axes[0, 0]
    ax.plot(metrics["n_ctx"], metrics["lmlhd"], label="LMLHD")
    text = ""
    text = text + "\n" + f"AUC-LMLHD: {metrics['lmlhd_auc']:.4f}"
    ax.text(x=0.25, y=0.75, transform=ax.transAxes, s=text)
    ax.set_xlabel("n_ctx")
    ax.legend()

    ## plot per-task metrics
    boxplot_with_n_task(
        x="n_ctx",
        y="MSE1",
        data=metrics["per_task_metrics"],
        ax=axes[3, 0],
        showfliers=show_outliers,
    )
    boxplot_with_n_task(
        x="n_ctx",
        y="MSE2",
        data=metrics["per_task_metrics"],
        ax=axes[4, 0],
        showfliers=show_outliers,
    )
    boxplot_with_n_task(
        x="n_ctx",
        y="MSE3",
        data=metrics["per_task_metrics"],
        ax=axes[5, 0],
        showfliers=show_outliers,
    )
    boxplot_with_n_task(
        x="n_ctx",
        y="LMLHD",
        data=metrics["per_task_metrics"],
        ax=axes[1, 0],
        showfliers=show_outliers,
    )
    axes[2, 0].set_title("Per-task metrics")

    fig.tight_layout()
    return fig
