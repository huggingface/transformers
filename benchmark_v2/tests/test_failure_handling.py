import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from benchmark_v2.framework.benchmark_config import BenchmarkConfig
from benchmark_v2.framework.benchmark_runner import BenchmarkRunner
from benchmark_v2.framework.data_classes import BenchmarkResult
from benchmark_v2.framework.validation import validate_generation_outputs
from transformers import ContinuousBatchingConfig, GenerationConfig
from transformers.generation.continuous_batching.requests import GenerationOutput, RequestStatus


def output(request_id="req_0", prompt=None, **kwargs):
    return GenerationOutput(
        request_id=request_id,
        prompt_ids=[10, 11] if prompt is None else prompt,
        status=kwargs.pop("status", RequestStatus.FINISHED),
        generated_tokens=[5, 6],
        created_time=1.0,
        lifespan=(1.0, 2.0),
        **kwargs,
    )


@pytest.mark.parametrize(
    "bad_results",
    [
        {},
        {"req_1": output("req_1")},
        {"req_0": output(error="out of memory")},
        {"req_0": output(status=RequestStatus.FAILED)},
        {"req_0": output(status=RequestStatus.DECODING)},
        {"req_0": output(prompt=[99])},
        {"req_0": output("req_9")},
    ],
)
def test_reject_incomplete_or_failed_results(bad_results):
    with pytest.raises(RuntimeError):
        validate_generation_outputs(bad_results, [[10, 11]])


def test_multiple_return_sequences_and_duplicate_prompts():
    outputs = {key: output(key) for key in ["req_1", "req_0__child#0", "req_0", "req_1__child#0"]}
    validate_generation_outputs(outputs, [[10, 11], [10, 11]], 2)
    del outputs["req_1__child#0"]
    with pytest.raises(RuntimeError, match="req_1__child#0"):
        validate_generation_outputs(outputs, [[10, 11], [10, 11]], 2)


@pytest.fixture
def suite(monkeypatch):
    # These tests exercise orchestration, not the optional scoring libraries.
    for name in [
        "tabulate",
        "lighteval.models.model_output",
        "lighteval.tasks.lighteval_task",
        "lighteval.tasks.prompt_manager",
        "lighteval.tasks.registry",
        "lighteval.tasks.requests",
    ]:
        monkeypatch.setitem(sys.modules, name, Mock())
    monkeypatch.setattr(sys, "path", sys.path[:])
    path = Path(__file__).parents[1] / "benchmark_scripts" / "continuous_batching_overall.py"
    spec = importlib.util.spec_from_file_location("_failure_suite", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failure", ["setup", "generation", "missing", "failed"])
def test_suite_records_failures_without_scoring(suite, failure):
    results = suite.BenchmarkResults("test", "sdpa")
    results.cleanup = Mock()
    model = Mock(generation_config=GenerationConfig())
    results._get_model = Mock(return_value=model)
    if failure == "setup":
        results._get_model.side_effect = RuntimeError("load failed")
    elif failure == "generation":
        model.generate_batch.side_effect = RuntimeError("generation failed")
    else:
        model.generate_batch.return_value = {} if failure == "missing" else {"req_0": output(error="failed")}
    scorer = Mock()
    results.add_benchmark([[10, 11]], 2, ContinuousBatchingConfig(), score_fn=scorer)
    entry = results.entries[0]
    assert entry.error
    assert entry.time_seconds is None
    assert entry.throughput_tok_per_sec is None
    assert entry.accuracy is None
    scorer.assert_not_called()
    results.cleanup.assert_called()


def test_scorer_matches_request_ids_and_rejects_missing_outputs(suite):
    docs = [SimpleNamespace(name="first"), SimpleNamespace(name="second")]
    metric = Mock()
    metric.sample_level_fn.compute.side_effect = [1.0, 0.0]
    registry = suite.Registry.return_value
    registry.task_to_configs = {"task": [SimpleNamespace(metrics=[metric])]}
    task = Mock()
    task.get_docs.return_value = docs
    registry.load_tasks.return_value = {"task": task}
    tokenizer = Mock(return_value={"input_ids": [[10, 11], [20, 21]]})
    tokenizer.decode.side_effect = lambda tokens, **kwargs: str(tokens)
    _, score = suite._build_lighteval_inputs_scorer(
        tokenizer, task_spec="task|0", task_name="task", use_chat_template=False
    )
    outputs = {"req_1": output("req_1", [20, 21]), "req_0": output()}
    outputs["req_1"].generated_tokens = [7, 8]
    assert score(outputs) == 0.5
    assert [call.args[0] for call in tokenizer.decode.call_args_list] == [[5, 6], [7, 8]]
    assert [call.args[0] for call in metric.sample_level_fn.compute.call_args_list] == docs
    del outputs["req_0"]
    metric.sample_level_fn.compute.reset_mock()
    with pytest.raises(RuntimeError, match="missing"):
        score(outputs)
    metric.sample_level_fn.compute.assert_not_called()


def runner_for_scenarios(tmp_path):
    runner = BenchmarkRunner(Mock(), output_dir=str(tmp_path), commit_id="test")
    runner.setup_benchmark = Mock()
    runner.cleanup = Mock()
    runner.save_results = Mock()
    return runner


def successful_result():
    result = BenchmarkResult()
    result.accumulate(1.0, [[0.1, 0.2]], "output", None)
    return result


@pytest.mark.parametrize("failure", ["setup", "generation", "empty"])
def test_main_runner_continues_and_reports_failure(tmp_path, capsys, failure):
    runner = runner_for_scenarios(tmp_path)
    runner.run_benchmark = Mock(return_value=successful_result())
    if failure == "setup":
        runner.setup_benchmark.side_effect = [RuntimeError("load failed"), None]
    else:
        runner.run_benchmark.side_effect = [
            RuntimeError("generate failed") if failure == "generation" else BenchmarkResult(),
            successful_result(),
        ]
    configs = [BenchmarkConfig(name=name, skip_validity_check=True) for name in ["bad", "good"]]
    _, results = runner.run_benchmarks("test", configs)
    failed = results[configs[0].hash]
    assert failed["metadata"].success is False
    assert failed["metadata"].to_dict()["error"]
    assert results[configs[1].hash]["metadata"].success is True
    assert runner.cleanup.call_count == 2
    assert runner.save_results.call_count == 2
    assert "FAILED:" in capsys.readouterr().out


def test_all_failed_scenarios_are_saved_then_raise(tmp_path):
    runner = runner_for_scenarios(tmp_path)
    runner.run_benchmark = Mock(side_effect=RuntimeError("failed"))
    with pytest.raises(RuntimeError, match="No benchmark was run successfully"):
        runner.run_benchmarks("test", [BenchmarkConfig(skip_validity_check=True)])
    runner.save_results.assert_called_once()


def test_empty_measurements_do_not_print_zero_statistics(capsys):
    BenchmarkResult().pprint(batch_size=1, num_generated_tokens=2)
    assert capsys.readouterr().out.strip() == "No measurements available"


def test_monitor_stopped_when_generation_raises():
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.logger = Mock()
    runner.inputs = {"input_ids": torch.tensor([[10, 11]])}
    runner.model = Mock()
    runner.model.generate.side_effect = RuntimeError("generation failed")
    config = BenchmarkConfig(gpu_monitoring=True, skip_validity_check=True)
    with patch("benchmark_v2.framework.benchmark_runner.GPUMonitor") as monitor:
        with pytest.raises(RuntimeError, match="generation failed"):
            runner.time_generate(config, warmup=False)
        monitor.return_value.stop_and_collect.assert_called_once()


@pytest.mark.parametrize("bad_results", [{}, {"req_0": output(error="failed")}])
def test_main_runner_rejects_partial_generation_before_decoding(bad_results):
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.logger = Mock()
    runner.inputs = {"input_ids": torch.tensor([[10, 11]])}
    runner.model = Mock()
    runner.model.generate_batch.return_value = bad_results
    runner.continuous_batching_config = ContinuousBatchingConfig()
    runner.tokenizer = Mock()
    config = BenchmarkConfig(continuous_batching=True, gpu_monitoring=False, skip_validity_check=True)
    with pytest.raises(RuntimeError):
        runner.time_generate(config, warmup=False)
    runner.tokenizer.decode.assert_not_called()
