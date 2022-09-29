import os
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from experiment_util.metalearning_model import MetaLearningLVMParametric
from metalearning_benchmarks import MetaLearningBenchmark
from tqdm import tqdm

from metalearning_model_gmm_np.metalearning_gmm import MetaLearningGMM
from metalearning_model_gmm_np.model_learner import ModelLearner
from metalearning_model_gmm_np.np import NP
from metalearning_model_gmm_np.posterior_learner import (
    MultiDaftLearner,
)
from metalearning_model_gmm_np.subtask_generator import SubtaskGenerator


class MetaLearningModelGMMNP(MetaLearningLVMParametric):
    def __init__(self, d_x: int, d_y: int, cfg: dict):
        super().__init__(d_x=d_x, d_y=d_y, d_z=cfg["d_z"], cfg=cfg)
        self._cfg = cfg
        self._set_all_seeds()
        self._model = self._generate_model()

        # the subtask generator used for meta training
        self._subtask_generator_meta = None

    def _set_all_seeds(self):
        np.random.seed(self._cfg["seed"])
        tf.random.set_seed(self._cfg["seed"])

    def _generate_model(self):
        np_model = NP(
            d_x=self._d_x,
            d_y=self._d_y,
            d_z=self._cfg["d_z"],
            gmm_prior_scale=float(self._cfg["gmm_prior_scale"]),
            gmm_prior_n_components=self._cfg["gmm_prior_n_components"],
            gmm_posterior_n_components=self._cfg["gmm_posterior_n_components"],
            decoder_n_hidden=self._cfg["decoder_n_hidden"],
            decoder_d_hidden=self._cfg["decoder_d_hidden"],
            decoder_activation=self._cfg["decoder_activation"],
            decoder_std_y_features=self._cfg["decoder_output_scale_features"],
            decoder_fixed_std_y_value=self._cfg["decoder_output_scale"],
            decoder_std_y_lower_bound=self._cfg["decoder_output_scale_min"],
        )
        return np_model

    def _generate_posterior_learner(self):
        posterior_learner = MultiDaftLearner(
            np_model=self._model,
            n_samples_per_comp=self._cfg["gmm_learner_n_samples_per_comp"],
            component_kl_bound=float(self._cfg["gmm_learner_component_kl_bound"]),
            dual_conv_tol=float(self._cfg["gmm_learner_dual_conv_tol"]),
            global_upper_bound=float(self._cfg["gmm_learner_global_upper_bound"]),
            use_warm_starts=self._cfg["gmm_learner_use_warm_starts"],
            diagonal_cov=self._cfg["gmm_diagonal_cov"],
        )

        return posterior_learner

    def _reset(
        self,
        x: np.ndarray,
        y: np.ndarray,
        ctx_masks: np.ndarray,
        task_ids: np.ndarray,
    ):
        """
        Initializes new GMM posteriors for all tasks in input data (x, y).
        (i) Create new GMM posteriors.
        (ii) Initialize these posteriors to prior parameters, choosing the components
             that best explain the data.
        """
        # Create posteriors
        n_tasks = x.shape[0]
        self._model.reset_posteriors(n_tasks_total=n_tasks)

        # Initialize posteriors
        self._model.initialize_posteriors(  # called on all subtasks
            x=tf.convert_to_tensor(x),
            y=tf.convert_to_tensor(y),
            ctx_mask=tf.convert_to_tensor(ctx_masks),
            task_ids=tf.convert_to_tensor(task_ids),
            n_samples_comp=self._cfg["n_samples_posterior_init"],
        )

    def restore_latest_checkpoint(self, trackables: dict):
        checkpoint = tf.train.Checkpoint(**trackables)
        latest_checkpoint = tf.train.latest_checkpoint(self._cfg["checkpoint_path"])
        status = checkpoint.restore(latest_checkpoint)
        # sanity check
        try:
            status.assert_existing_objects_matched()
            print(f"Successfully loaded latest_checkpoint = {latest_checkpoint}!")
        except AssertionError:
            print(f"Warning! Restoring latest_checkpoint = {latest_checkpoint} failed!")
        return checkpoint

    def save_checkpoint(self, checkpoint):
        os.makedirs(self._cfg["checkpoint_path"], exist_ok=True)
        path = checkpoint.save(os.path.join(self._cfg["checkpoint_path"], "ckpt"))
        print(f"Saved checkpoint at {path}!")

    def _meta_train(
        self,
        benchmark: MetaLearningBenchmark,
        n_epochs: int,
    ) -> None:
        # generate training data
        self._subtask_generator_meta = SubtaskGenerator(
            benchmark=benchmark,
            seed=self._cfg["seed"],
            ctx_size_range=(
                self._cfg["min_context_size_meta"],
                self._cfg["max_context_size_meta"],
            ),
            n_ctx_sets_per_task=self._cfg["n_context_sets_per_task_meta"],
        )
        x, y, ctx_masks, task_ids = self._subtask_generator_meta.get_data()
        n_subtasks = x.shape[0]
        ds_tf = tf.data.Dataset.from_tensor_slices((x, y, ctx_masks, task_ids))
        ds_tf = ds_tf.shuffle(buffer_size=n_subtasks, reshuffle_each_iteration=True)
        # Sanity check (as we drop the remainder, otherwise batch is empty)
        assert self._cfg["batch_size"] <= n_subtasks
        # We have to set drop_remainder = True necessary for prior learning
        ds_tf = ds_tf.batch(batch_size=self._cfg["batch_size"], drop_remainder=True)

        # reset model
        self._reset(x=x, y=y, ctx_masks=ctx_masks, task_ids=task_ids)

        ## generate learners
        # model learner
        model_learner = ModelLearner(
            np_model=self._model,
            lr_likelihood=float(self._cfg["model_learner_lr"]),
            n_samples=self._cfg["model_learner_n_samples"],
            learn_prior=self._cfg["model_learner_learn_prior"],
            prior_diagonal_cov=self._cfg["gmm_diagonal_cov"],
            n_tasks_batch=self._cfg["batch_size"],
            n_tasks_total=n_subtasks,
        )
        # posterior learner
        posterior_learner = self._generate_posterior_learner()

        # restore checkpoint
        if self._cfg.get("do_save_and_load_for_meta_training", False):
            trackables = (
                model_learner.trackables
                | posterior_learner.trackables
                | self._model.trackables
            )
            checkpoint = self.restore_latest_checkpoint(trackables)

        # perform meta training
        for i in (pbar := tqdm(range(n_epochs), desc="Meta-train")):
            # perform one epoch
            mb_loss_model, mb_loss_posterior = [], []
            # go through minibatches
            for x_mb, y_mb, ctx_masks_mb, task_ids_mb in tqdm(
                ds_tf,
                desc="Meta-train mb",
                disable=not self._cfg["show_tqdm_for_minibatches"],
            ):
                # prepare minibatch
                self._model.set_batch(task_ids=task_ids_mb)
                # step model
                m1 = model_learner.step(x=x_mb, y=y_mb, ctx_mask=ctx_masks_mb)
                mb_loss_model.append(m1["loss_likelihood"])
                # step posterior
                m2 = posterior_learner.step(x=x_mb, y=y_mb, ctx_mask=ctx_masks_mb)
                # bookkeeping
                mb_loss_posterior.append(-m2["mean_log_tgt_density"])
            # bookkeeping
            log_dict = {
                "epoch": i,
                "lc/meta_loss_model": np.mean(mb_loss_model),
                "lc/meta_neg_mean_log_tgt_density": np.mean(mb_loss_posterior),
            }
            pbar.set_postfix(log_dict)

        # save checkpoint
        if self._cfg.get("do_save_and_load_for_meta_training", False):
            self.save_checkpoint(checkpoint)

        ## set batch to task_ids given by SubtaskGenerator
        # This call may lead to OOM errors for large numbers of tasks, and it is
        # not necessary if we do not want to have a look at the training tasks again.
        # Thus we allow to switch it off through the config.
        if self._cfg.get("set_batch_after_meta_training", True):
            self._model.set_batch(task_ids)

    def _adapt(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_epochs: int,
    ) -> None:
        n_tasks = x.shape[0]
        n_context = x.shape[1]

        # generate training data
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        ctx_masks = np.ones((n_tasks, n_context), dtype=bool)
        task_ids = np.arange(n_tasks, dtype=np.int32)
        ds_tf = tf.data.Dataset.from_tensor_slices((x, y, ctx_masks, task_ids))
        ds_tf = ds_tf.shuffle(buffer_size=n_tasks, reshuffle_each_iteration=True)
        ds_tf = ds_tf.batch(batch_size=self._cfg["batch_size"], drop_remainder=False)

        # reset model
        self._reset(x=x, y=y, ctx_masks=ctx_masks, task_ids=task_ids)

        if n_context > 0:  # if no context data provided, leave model in prior state
            # generate learner
            posterior_learner = self._generate_posterior_learner()

            # perform adaptation
            for _ in (pbar := tqdm(range(n_epochs), desc="Adapt")):
                # perform one epoch
                mb_loss_posterior = []
                # go through minibatches
                for x_mb, y_mb, ctx_masks_mb, task_ids_mb in ds_tf:
                    # prepare minibatch
                    self._model.set_batch(task_ids=task_ids_mb)
                    # step posterior
                    m = posterior_learner.step(x=x_mb, y=y_mb, ctx_mask=ctx_masks_mb)
                    # bookkeeping
                    mb_loss_posterior.append(-m["mean_log_tgt_density"])
                # bookkeeping
                log_dict = {
                    "adapt_neg_mean_log_tgt_density": np.mean(mb_loss_posterior)
                }
                pbar.set_postfix(log_dict)

        # set batch to user-defined task_ids
        self._model.set_batch(task_ids)

    def _predict(
        self,
        x: np.ndarray,
        n_samples: Optional[int] = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_pred, var_pred = self._model.predict(
            x=x,
            n_samples=n_samples,
            sample_from="approximate_posterior",  # == "prior" if n_ctx == 0
        )
        return y_pred, var_pred

    def _sample_z_deprecated(self, n_samples: int) -> np.ndarray:
        print("Warning: This method (_sample_z_deprecated()) is deprecated!")

        z = self._model.sample_z(
            n_samples=n_samples,
            sample_from="approximate_posterior",  # == "prior" if n_ctx == 0
        )
        return z

    def _predict_at_z(
        self,
        x: np.ndarray,
        z: np.ndarray,
        task_ids=None,  # unused
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_pred, var_pred = self._model.predict_at_z(x=x, z=z)
        return y_pred, var_pred

    def latent_distribution(self) -> MetaLearningGMM:
        # read out latent state and check consistency
        log_w = self._model.post_batch.log_w.numpy()
        mu = self._model.post_batch.loc.numpy()
        scale_tril = self._model.post_batch.scale_tril.numpy()

        # check consistency
        # prior might have differen n_comps than posterior, so read n_comps from log_w
        n_components = log_w.shape[1]
        assert log_w.shape == (self._n_tasks_adapt, n_components)
        assert mu.shape == (self._n_tasks_adapt, n_components, self._d_z)
        assert scale_tril.shape == (self._n_tasks_adapt, n_components, self._d_z, self._d_z)  # fmt: skip

        # return MetaLearningDistribution
        return MetaLearningGMM(log_w=log_w, mu=mu, scale_tril=scale_tril)
