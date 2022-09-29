from metalearning_benchmarks import benchmark_dict as BM_DICT
from metalearning_benchmarks import MetaLearningBenchmark
from dataclasses import dataclass, field


@dataclass
class ExperimentData:
    """
    Generate benchmarks in a standardized way.
    """

    config: dict

    benchmark_meta_train: MetaLearningBenchmark = field(init=False)
    benchmark_meta_test: MetaLearningBenchmark = field(init=False)
    benchmark_test: MetaLearningBenchmark = field(init=False)

    def __post_init__(self):
        # define seeds
        seed_task_meta = self.config["seed_data"]
        seed_task_test = self.config["seed_data"] + 1
        seed_x_meta_train = self.config["seed_data"] + 2
        seed_x_meta_test = self.config["seed_data"] + 3
        seed_x_test = self.config["seed_data"] + 4
        seed_noise_meta_train = self.config["seed_data"] + 5
        seed_noise_meta_test = self.config["seed_data"] + 6
        seed_noise_test = self.config["seed_data"] + 7

        # generate benchmarks
        self.benchmark_meta_train = BM_DICT[self.config["benchmark_meta"]](
            n_task=self.config["n_tasks_meta"],
            n_datapoints_per_task=self.config["n_datapoints_per_task_meta_train"],
            output_noise=self.config["data_noise_std"],
            seed_task=seed_task_meta,
            seed_x=seed_x_meta_train,
            seed_noise=seed_noise_meta_train,
        )
        self.benchmark_meta_test = BM_DICT[self.config["benchmark_meta"]](
            n_task=self.config["n_tasks_meta"],
            n_datapoints_per_task=self.config["n_datapoints_per_task_meta_test"],
            output_noise=self.config["data_noise_std"],
            seed_task=seed_task_meta,
            seed_x=seed_x_meta_test,
            seed_noise=seed_noise_meta_test,
        )
        self.benchmark_test = BM_DICT[self.config["benchmark_test"]](
            n_task=self.config["n_tasks_test"],
            n_datapoints_per_task=self.config["n_datapoints_per_task_test"],
            output_noise=self.config["data_noise_std"],
            seed_task=seed_task_test,
            seed_x=seed_x_test,
            seed_noise=seed_noise_test,
        )
