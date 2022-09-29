from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from metalearning_benchmarks import MetaLearningBenchmark
from experiment_util.metalearning_distributions import MetaLearningDistribution


class MetaLearningModel(ABC):
    def __init__(self, d_x: int, d_y: int, cfg: Optional[dict]):
        self._d_x = d_x
        self._d_y = d_y
        self._n_tasks_adapt = None

    @abstractmethod
    def _meta_train(
        self,
        benchmark: MetaLearningBenchmark,
        n_epochs: Optional[int] = None,
    ) -> None:
        pass

    @abstractmethod
    def _adapt(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_epochs: Optional[int] = None,
    ) -> None:
        pass

    @abstractmethod
    def _predict(
        self, x: np.ndarray, n_samples: Optional[int] = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def meta_train(
        self,
        benchmark: MetaLearningBenchmark,
        n_epochs: Optional[int] = None,
    ) -> None:
        self._meta_train(
            benchmark=benchmark,
            n_epochs=n_epochs,
        )

    def adapt(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_epochs: Optional[int] = None,
    ) -> None:
        # check inputs
        n_tasks = x.shape[0]
        n_context = x.shape[1]
        assert x.shape == (n_tasks, n_context, self._d_x)
        assert y.shape == (n_tasks, n_context, self._d_y)

        # store n_tasks
        self._n_tasks_adapt = n_tasks

        # adapt
        self._adapt(x=x, y=y, n_epochs=n_epochs)

    def predict(
        self, x: np.ndarray, n_samples: Optional[int] = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        # check inputs
        n_tasks = x.shape[0]
        n_tgt = x.shape[1]
        assert x.shape == (n_tasks, n_tgt, self._d_x)
        assert n_tasks == self._n_tasks_adapt

        y_pred, var_pred = self._predict(x=x, n_samples=n_samples)

        # check outputs
        assert y_pred.shape == (n_samples, n_tasks, n_tgt, self._d_y)
        assert var_pred.shape == (n_samples, n_tasks, n_tgt, self._d_y)
        return y_pred, var_pred


class MetaLearningLVM(MetaLearningModel):
    """
    MetaLearningModel which allows to pre-compute a task encoding of the context set and
    to use this for predictions.
    """

    def __init__(self, d_x: int, d_y: int, d_z: int, cfg: Optional[dict]):
        super().__init__(d_x=d_x, d_y=d_y, cfg=cfg)
        self._d_z = d_z

    @abstractmethod
    def _predict_at_z(
        self,
        x: np.ndarray,
        z: np.ndarray,
        task_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs predictions at x for task encoded as z.
        - Returns the mean mu and **variance** var = sigma^2 of a factorized Gaussian
          likelihood, i.e.,
          p(y | x, z) = N(y | mu, var).
        - This method is in principle independent from the last adapt call, so the number
          of tasks for which this method is called is independent from the number of tasks
          it was adapted to (as this method receives z).
          Caveat: Some models (e.g., ANP) use the context set (stored internally) to
          compute predictions, even when z is known (e.g., to compute attention between x
          and x_ctx). Thus, if the number of tasks for which this method is called
          is different from the number of tasks the model was adapted to,
          task_id has to be provided. It tells the model which tasks from the context set
          have to be used.
        Inputs:
        - x.shape = (n_tasks, n_points, d_x)
        - z.shape = (n_samples, n_tasks, d_z)
        - Optional: task_ids.shape = (n_tasks,)
        - If task_ids is not given, n_tasks has to match the number of tasks from the
          most recent call to adapt().
        Returns: (mu, sigma^2)
        - mu.shape = (n_samples, n_tasks, n_points, d_y)
        - var.shape = (n_samples, n_tasks, n_points, d_y), positive
        """
        pass

    @abstractmethod
    def _sample_z(self, n_samples: int) -> np.ndarray:
        pass

    def sample_z(self, n_samples: int) -> np.ndarray:
        z = self._sample_z(n_samples=n_samples)

        # check output
        assert z.shape == (n_samples, self._n_tasks_adapt, self._d_z)
        return z

    def predict_at_z(
        self,
        x: np.ndarray,
        z: np.ndarray,
        task_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # check inputs
        n_tasks = x.shape[0]
        n_tgt = x.shape[1]
        n_samples = z.shape[0]
        assert x.shape == (n_tasks, n_tgt, self._d_x)
        assert z.shape == (n_samples, n_tasks, self._d_z)
        if task_ids is None:
            assert n_tasks == self._n_tasks_adapt
        else:
            assert task_ids.shape == (n_tasks,)

        # predict
        y_pred, var_pred = self._predict_at_z(x=x, z=z, task_ids=task_ids)

        # check outputs
        assert y_pred.shape == (n_samples, n_tasks, n_tgt, self._d_y)
        assert var_pred.shape == (n_samples, n_tasks, n_tgt, self._d_y)
        assert (var_pred > 0).all()
        return y_pred, var_pred


class MetaLearningLVMParametric(MetaLearningLVM):
    """
    MetaLearningLVM with a parametric latent distribution.
    """

    def __init__(self, d_x: int, d_y: int, d_z: int, cfg: Optional[dict]):
        super().__init__(d_x=d_x, d_y=d_y, d_z=d_z, cfg=cfg)
        self._d_z = d_z

    @abstractmethod
    def latent_distribution(self) -> MetaLearningDistribution:
        """
        Return the latent distribution, conditioned on the dataset D passed to the most
        recent call to adapt().
        If the last call to adapt was with an empty dataset i.e., n_points_per_task = 0,
        q is supposed to equal the prior.
        Inputs: None
        Returns:
        q(z | D) as a MetaLearningDistribution
        """
        pass

    def _sample_z(self, n_samples: int) -> np.ndarray:
        return self.latent_distribution().sample(n_samples)
