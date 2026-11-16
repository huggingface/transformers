import torch

from benchmark_v2.framework.benchmark_runner import BenchmarkRunner
from benchmark_v2.framework.data_classes import BenchmarkResult


def test_startup_metrics_round_trip():
    result = BenchmarkResult()
    result.startup = {"load_time": 1.5, "first_generate_latency": 9.0, "measurement_compiled_graphs": 0}
    for summarized in (True, False):
        data = result.to_dict(summarized=summarized)
        data["e2e_latency"] = []
        assert BenchmarkResult.from_dict(data).startup == result.startup


def test_results_without_startup_still_load():
    data = BenchmarkResult().to_dict()
    del data["startup"]
    assert BenchmarkResult.from_dict(data).startup == {}


def test_compile_counters_count_recompiles():
    torch._dynamo.reset()
    f = torch.compile(lambda x: x * 2, backend="eager")
    before = BenchmarkRunner._compile_counters()
    f(torch.ones(3))
    f(torch.ones(3))  # same shape: no new graph
    f(torch.ones(3, 2))  # new rank: recompile
    after = BenchmarkRunner._compile_counters()
    assert after["compiled_graphs"] - before["compiled_graphs"] == 2
