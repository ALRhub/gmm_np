from itertools import cycle
from typing import Tuple, Union

import numpy as np
from metalearning_benchmarks.metalearning_benchmark import MetaLearningBenchmark
from scipy.special import comb


class SubtaskGenerator:
    def __init__(
        self,
        benchmark: MetaLearningBenchmark,
        seed: int,
        ctx_size_range: Tuple[int],
        n_ctx_sets_per_task: int,
    ):
        self.bm = benchmark
        self.rng = np.random.default_rng(seed=seed)
        self.n_ctx_sets_per_task = n_ctx_sets_per_task
        self.n_points = self.bm.n_datapoints_per_task
        self.ctx_size_range = ctx_size_range
        (
            self.x,
            self.y,
            self.ctx_mask,
            self.ctx_size,
            self.subtask_ids,
            self.task_id,
        ) = self.generate_subtasks()

    @property
    def n_subtasks(self):
        return self.x.shape[0]

    def get_data(self):
        return self.x, self.y, self.ctx_mask, self.subtask_ids

    def get_subtasks_by_subtask_ids(self, subtask_ids: Union[int, np.ndarray]):
        # subtask_id is the id of the task in the dataset
        return (
            self.x[subtask_ids],
            self.y[subtask_ids],
            self.ctx_mask[subtask_ids],
            self.subtask_ids[subtask_ids],
        )

    def get_subtasks_by_task_id(self, task_id: int):
        # task_id is the id of the task in the benchmark
        idx = self.task_id == task_id
        if not idx.any():
            raise ValueError(f"Task id {task_id:d} not found!")
        return self.get_subtasks_by_subtask_ids(subtask_ids=self.subtask_ids[idx])

    def get_subtasks_by_ctx_size(self, ctx_size: int):
        idx = self.ctx_size == ctx_size
        if not idx.any():
            raise ValueError(f"Task size {ctx_size:d} not available!")
        return self.get_subtasks_by_subtask_ids(subtask_ids=self.subtask_ids[idx])

    def get_subtasks_by_task_id_and_ctx_size(self, full_task_id: int, ctx_size: int):
        idx = np.logical_and(self.task_id == full_task_id, self.ctx_size == ctx_size)
        if not idx.any():
            raise ValueError(
                f"Task id {full_task_id:d} has no subtask of size {ctx_size:d}!"
            )
        return self.get_subtasks_by_subtask_ids(subtask_ids=self.subtask_ids[idx])

    def get_subtask_ids_by_task_id_and_ctx_size(self, full_task_id: int, ctx_size: int):
        idx = np.logical_and(self.task_id == full_task_id, self.ctx_size == ctx_size)
        if not idx.any():
            raise ValueError(
                f"Task id {full_task_id:d} has no subtask of size {ctx_size:d}!"
            )
        return self.get_subtasks_by_subtask_ids(subtask_ids=self.subtask_ids[idx])[3]

    def get_context_set(self, subtask_ids):
        subtask_ids = np.atleast_1d(subtask_ids)
        x, y, ctx_masks, _ = self.get_subtasks_by_subtask_ids(subtask_ids)
        n_tasks = x.shape[0]
        n_ctx = np.sum(ctx_masks[0])
        d_x = x.shape[2]
        d_y = y.shape[2]

        x_ctx = np.zeros((n_tasks, n_ctx, d_x))
        y_ctx = np.zeros((n_tasks, n_ctx, d_y))
        for l in range(n_tasks):
            assert (
                sum(ctx_masks[l]) == n_ctx
            ), "All tasks have to have the same context size!"
            x_ctx[l] = x[l][ctx_masks[l]]
            y_ctx[l] = y[l][ctx_masks[l]]

        return x_ctx, y_ctx

    def iter_ctx_sizes(self):
        for ctx_size in np.unique(self.ctx_size):
            yield ctx_size

    def iter_subtasks_by_ctx_size(self):
        for ctx_size in np.unique(self.ctx_size):
            yield ctx_size, self.get_subtasks_by_ctx_size(ctx_size)

    def iter_subtask_ids_by_ctx_size(self):
        for ctx_size in np.unique(self.ctx_size):
            yield ctx_size, self.get_subtasks_by_ctx_size(ctx_size)[-1]

    def generate_subtasks(self):
        """
        Generates subtasks consisting of subsets of the full task.
        To be used as context sets for non-amortizing models.
        """

        ## helpers
        def get_next_ctx_size():
            ct = 0
            for k in ctx_size_iterator:
                if size_counts[k] < comb(self.n_points, k):
                    size_counts[k] += 1
                    return k

                # assure that function returns
                ct += 1
                if ct >= self.n_ctx_sets_per_task:
                    raise ValueError(
                        f"There are not enough distinct context sets available!"
                    )

        def get_next_mask_idx(ctx_size, assure_unique=True):
            # -> is there a better way than rejection sampling
            while True:
                mask_idx_cand = sorted(
                    self.rng.choice(range(self.n_points), size=ctx_size, replace=False)
                )
                if assure_unique:
                    if tuple(mask_idx_cand) not in used_mask_idx[ctx_size]:
                        used_mask_idx[ctx_size].append(tuple(mask_idx_cand))
                        return mask_idx_cand
                else:
                    return mask_idx_cand

        ## initialize arrays
        n_subtasks = self.bm.n_task * self.n_ctx_sets_per_task
        x = np.zeros((n_subtasks, self.n_points, self.bm.d_x), dtype=np.float32)
        y = np.zeros((n_subtasks, self.n_points, self.bm.d_y), dtype=np.float32)
        ctx_mask = np.zeros((n_subtasks, self.n_points), dtype=np.int32)
        ctx_size = np.zeros((n_subtasks,), dtype=np.int32)
        subtask_ids = np.zeros((n_subtasks,), dtype=np.int32)
        task_ids = np.zeros((n_subtasks,), dtype=np.int32)

        ## keep track of how many context sets of size k we already added to
        # avoid obvious redundancy
        ctx_size_iterator = cycle(
            np.unique(
                np.linspace(
                    self.ctx_size_range[0],
                    self.ctx_size_range[1],
                    num=self.n_ctx_sets_per_task,
                    dtype=np.int32,
                )
            )
        )
        size_counts = {
            k: 0 for k in range(self.ctx_size_range[0], self.ctx_size_range[1] + 1)
        }
        used_mask_idx = {
            k: [] for k in range(self.ctx_size_range[0], self.ctx_size_range[1] + 1)
        }

        ## generate subtasks
        task_id = -1
        for _ in range(self.n_ctx_sets_per_task):
            cur_ctx_size = get_next_ctx_size()
            cur_mask_idx = get_next_mask_idx(ctx_size=cur_ctx_size)
            for l, task in enumerate(self.bm):
                task_id += 1
                x[task_id], y[task_id] = task.x, task.y
                ctx_size[task_id] = cur_ctx_size
                ctx_mask[task_id][cur_mask_idx] = 1
                subtask_ids[task_id] = task_id
                task_ids[task_id] = l

        ## convert to boolean ctx masks
        ctx_mask = ctx_mask.astype(bool)

        return x, y, ctx_mask, ctx_size, subtask_ids, task_ids
