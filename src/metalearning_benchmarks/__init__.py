from metalearning_benchmarks.metalearning_task import MetaLearningTask
from metalearning_benchmarks.metalearning_benchmark import MetaLearningBenchmark
from metalearning_benchmarks.sinusoid1d_benchmark import Sinusoid1D
from metalearning_benchmarks.line_sine1d_benchmark import LineSine1D

benchmark_dict = {
    "Sinusoid1D": Sinusoid1D,
    "LineSine1D": LineSine1D,
}
