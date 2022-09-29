import math

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm


def mse_per_datapoint(
    y: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    MSE for each datapoint.
    """
    ## check input
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(y, np.ndarray)
    n_tasks = y.shape[0]
    n_points = y.shape[1]
    d_y = y.shape[2]
    n_samples = y_pred.shape[0]
    assert y_pred.shape == (n_samples, n_tasks, n_points, d_y)
    assert y.shape == (n_tasks, n_points, d_y)

    ## compute MSE
    mse = (y_pred - y[None, ...]) ** 2
    assert mse.shape == (n_samples, n_tasks, n_points, d_y)
    # sum over data dimension
    mse = np.sum(mse, axis=-1)

    ## check output
    assert mse.shape == (n_samples, n_tasks, n_points)
    return mse


def mse1(
    y: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Mean of MSE of all samples per task.
    """
    ## check input
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(y, np.ndarray)
    n_tasks = y.shape[0]
    n_points = y.shape[1]
    d_y = y.shape[2]
    n_samples = y_pred.shape[0]
    assert y_pred.shape == (n_samples, n_tasks, n_points, d_y)
    assert y.shape == (n_tasks, n_points, d_y)

    ## compute MSE
    # compute MSE of all samples
    mse = mse_per_datapoint(y_pred=y_pred, y=y)
    assert mse.shape == (n_samples, n_tasks, n_points)
    # compute mean over samples and points per task
    mse = np.mean(mse, axis=(0, 2))

    # check output
    assert mse.shape == (n_tasks,)
    return mse


def mse2(
    y: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    MSE of mean sample per task.
    """
    ## check input
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(y, np.ndarray)
    n_tasks = y.shape[0]
    n_points = y.shape[1]
    d_y = y.shape[2]
    n_samples = y_pred.shape[0]
    assert y_pred.shape == (n_samples, n_tasks, n_points, d_y)
    assert y.shape == (n_tasks, n_points, d_y)

    ## compute MSE
    # compute mean sample
    y_pred_mean = np.mean(y_pred, axis=0, keepdims=True)
    assert y_pred_mean.shape == (1, n_tasks, n_points, d_y)
    # compute MSE of mean sample
    mse = mse_per_datapoint(y_pred=y_pred_mean, y=y)
    mse = np.squeeze(mse, axis=0)
    assert mse.shape == (n_tasks, n_points)
    # compute mean over points per task
    mse = np.mean(mse, axis=-1)

    ## check output
    assert mse.shape == (n_tasks,)
    return mse


def mse3(
    y: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Minimum MSE of samples per task.
    """
    ## check input
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(y, np.ndarray)
    n_tasks = y.shape[0]
    n_points = y.shape[1]
    d_y = y.shape[2]
    n_samples = y_pred.shape[0]
    assert y_pred.shape == (n_samples, n_tasks, n_points, d_y)
    assert y.shape == (n_tasks, n_points, d_y)

    ## compute MSE
    # compute MSE of all samples
    mse = mse_per_datapoint(y_pred=y_pred, y=y)
    assert mse.shape == (n_samples, n_tasks, n_points)
    # average mses over datapoints per task
    mse = np.mean(mse, axis=-1)
    assert mse.shape == (n_samples, n_tasks)
    # compute min over samples
    mse = np.min(mse, axis=0)

    ## check output
    assert mse.shape == (n_tasks,)
    return mse


def compute_log_likelihood(
    y: np.ndarray,
    y_pred: np.ndarray,
    var_pred: np.ndarray,
) -> np.ndarray:
    # check input
    n_samples = y_pred.shape[0]
    n_tasks = y.shape[0]
    n_points = y.shape[1]
    d_y = y.shape[2]
    assert y.shape == (n_tasks, n_points, d_y)
    assert y_pred.shape == (n_samples, n_tasks, n_points, d_y)
    assert var_pred.shape == (n_samples, n_tasks, n_points, d_y)

    ## compute log likelihood
    log_likelihood = norm.logpdf(y, loc=y_pred, scale=np.sqrt(var_pred))
    assert log_likelihood.shape == (n_samples, n_tasks, n_points, d_y)
    log_likelihood = np.sum(log_likelihood, axis=-1)  # data dimension
    assert log_likelihood.shape == (n_samples, n_tasks, n_points)
    log_likelihood = np.sum(log_likelihood, axis=-1)  # points per task dimension

    # check output
    assert log_likelihood.shape == (n_samples, n_tasks)
    return log_likelihood


def log_marginal_likelihood_from_log_prob(
    log_prob: np.ndarray,
) -> np.ndarray:
    # check input
    n_samples = log_prob.shape[0]
    n_tasks = log_prob.shape[1]
    assert log_prob.shape == (n_samples, n_tasks)

    # marginalize
    lmlhd = logsumexp(log_prob, axis=0) - math.log(n_samples)  # sample dimension

    # check output
    assert lmlhd.shape == (n_tasks,)
    return lmlhd


def log_marginal_likelihood_naive_mc(
    y: np.ndarray,
    y_pred_at_z_prior: np.ndarray,
    var_pred_at_z_prior: np.ndarray,
) -> np.ndarray:
    # check input
    n_samples = y_pred_at_z_prior.shape[0]
    n_tasks = y.shape[0]
    n_points = y.shape[1]
    d_y = y.shape[2]
    assert y.shape == (n_tasks, n_points, d_y)
    assert y_pred_at_z_prior.shape == (n_samples, n_tasks, n_points, d_y)
    assert var_pred_at_z_prior.shape == (n_samples, n_tasks, n_points, d_y)

    # compute log likelihood
    log_likelihood_at_z_prior = compute_log_likelihood(
        y=y,
        y_pred=y_pred_at_z_prior,
        var_pred=var_pred_at_z_prior,
    )
    assert log_likelihood_at_z_prior.shape == (n_samples, n_tasks)
    ## marginalize
    lmlhd_naive_mc = log_marginal_likelihood_from_log_prob(log_likelihood_at_z_prior)

    # check output
    assert lmlhd_naive_mc.shape == (n_tasks,)
    return lmlhd_naive_mc


def log_marginal_likelihood_iw_mc(
    y: np.ndarray,
    y_pred_at_z_proposal: np.ndarray,
    var_pred_at_z_proposal: np.ndarray,
    log_prob_prior_at_z_proposal: np.ndarray,
    log_prob_proposal_at_z_proposal: np.ndarray,
) -> np.ndarray:
    """
    Log marginal likelihood per task, estimated using importance-weighted Monte-Carlo.
    """
    # check input
    n_samples = y_pred_at_z_proposal.shape[0]
    n_tasks = y.shape[0]
    n_points = y.shape[1]
    d_y = y.shape[2]
    assert y.shape == (n_tasks, n_points, d_y)
    assert y_pred_at_z_proposal.shape == (n_samples, n_tasks, n_points, d_y)
    assert var_pred_at_z_proposal.shape == (n_samples, n_tasks, n_points, d_y)
    assert log_prob_prior_at_z_proposal.shape == (n_samples, n_tasks)
    assert log_prob_proposal_at_z_proposal.shape == (n_samples, n_tasks)

    # compute log likelihood
    log_likelihood_at_z_proposal = compute_log_likelihood(
        y=y,
        y_pred=y_pred_at_z_proposal,
        var_pred=var_pred_at_z_proposal,
    )
    assert log_likelihood_at_z_proposal.shape == (n_samples, n_tasks)
    # compute log importance weights
    log_iw = log_prob_prior_at_z_proposal - log_prob_proposal_at_z_proposal
    assert log_iw.shape == (n_samples, n_tasks)
    # marginalize
    lmlhd_iw_mc = log_marginal_likelihood_from_log_prob(
        log_likelihood_at_z_proposal + log_iw
    )

    # check output
    assert lmlhd_iw_mc.shape == (n_tasks,)
    return lmlhd_iw_mc


def elbo(
    y: np.ndarray,
    y_pred_at_z_proposal: np.ndarray,
    var_pred_at_z_proposal: np.ndarray,
    log_prob_prior_at_z_proposal: np.ndarray,
    log_prob_proposal_at_z_proposal: np.ndarray,
) -> np.ndarray:
    """
    Evidence lower bound per task, estimated using naive Monte Carlo.
    The proposal distribution is the variational posterior.
    """
    # check input
    n_samples = y_pred_at_z_proposal.shape[0]
    n_tasks = y.shape[0]
    n_points = y.shape[1]
    d_y = y.shape[2]
    assert y.shape == (n_tasks, n_points, d_y)
    assert y_pred_at_z_proposal.shape == (n_samples, n_tasks, n_points, d_y)
    assert var_pred_at_z_proposal.shape == (n_samples, n_tasks, n_points, d_y)
    assert log_prob_prior_at_z_proposal.shape == (n_samples, n_tasks)
    assert log_prob_proposal_at_z_proposal.shape == (n_samples, n_tasks)

    # compute log likelihood
    log_likelihood_at_z_proposal = compute_log_likelihood(
        y=y,
        y_pred=y_pred_at_z_proposal,
        var_pred=var_pred_at_z_proposal,
    )
    assert log_likelihood_at_z_proposal.shape == (n_samples, n_tasks)
    # compute log importance weights
    log_iw = log_prob_prior_at_z_proposal - log_prob_proposal_at_z_proposal
    assert log_iw.shape == (n_samples, n_tasks)
    ## average log_probs
    elbo = np.mean(log_likelihood_at_z_proposal + log_iw, axis=0)

    ## check output
    assert elbo.shape == (n_tasks,)
    return elbo


def log_marginal_likelihood_ais():
    raise NotImplementedError
