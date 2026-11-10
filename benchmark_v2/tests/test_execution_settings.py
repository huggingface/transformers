import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from benchmark_v2.framework.benchmark_config import BenchmarkConfig
from benchmark_v2.framework.benchmark_runner import BenchmarkRunner
from benchmark_v2.framework.data_classes import BenchmarkResult
from transformers import BatchEncoding
from transformers.generation.continuous_batching.requests import GenerationOutput, RequestStatus


@pytest.mark.parametrize("continuous_batching", [False, True])
@pytest.mark.parametrize("sequence_length", [2, 8])
def test_setup_exact_inputs_and_explicit_settings(continuous_batching, sequence_length):
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner._setup_for = ""
    runner.logger = Mock()
    tokenizer = Mock()
    tokenizer.return_value = BatchEncoding(
        {
            "input_ids": torch.tensor([[10, 11, 12]] * 2),
            "attention_mask": torch.ones((2, 3), dtype=torch.long),
        }
    )
    model = Mock()
    model.eval.return_value = model
    model.device = torch.device("cpu")
    model.dtype = torch.bfloat16
    model.config = SimpleNamespace(_attn_implementation="sdpa")

    def load_model(*args, **kwargs):
        model.generation_config = kwargs["generation_config"]
        return model

    config = BenchmarkConfig(
        continuous_batching=continuous_batching,
        sequence_length=sequence_length,
        batch_size=2,
        attn_implementation="sdpa",
        compile_kwargs={},
        skip_validity_check=True,
    )
    with (
        patch("benchmark_v2.framework.benchmark_runner.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("benchmark_v2.framework.benchmark_runner.AutoModelForCausalLM.from_pretrained", side_effect=load_model),
    ):
        runner.setup_benchmark("test-model", config)

    assert runner.inputs["input_ids"].tolist() == [[10, 11, 12, 10, 11, 12, 10, 11][:sequence_length]] * 2
    assert runner.execution_settings["input_token_lengths"] == [sequence_length] * 2
    assert model.generation_config.eos_token_id == -1
    assert model.generation_config.cache_implementation == (None if continuous_batching else "static")
    if continuous_batching:
        assert runner.continuous_batching_config.allow_block_sharing is False
        assert runner.continuous_batching_config.auto_switch_to_flash is False
        model.generate_batch.return_value = {
            f"req_{i}": GenerationOutput(
                request_id=f"req_{i}",
                prompt_ids=prompt,
                status=RequestStatus.FINISHED,
                timestamps=[1.0, 2.0],
                generated_tokens=[1, 2],
            )
            for i, prompt in enumerate(runner.inputs["input_ids"].tolist())
        }
        runner.torch_accelerator_module = Mock()
        runner.torch_accelerator_module.memory_allocated.return_value = 0
        config.num_tokens_to_generate = 2
        config.gpu_monitoring = False
        with patch("benchmark_v2.framework.benchmark_runner.flush_memory"):
            runner.time_generate(config, warmup=False)
        assert model.generate_batch.call_args.kwargs["continuous_batching_config"] is runner.continuous_batching_config
        assert model.generate_batch.call_args.kwargs["persistent_manager"] is False
    json.dumps(runner.execution_settings)
    result = BenchmarkResult()
    result.execution_settings = runner.execution_settings
    assert BenchmarkResult.from_dict(result.to_dict()).execution_settings == runner.execution_settings


def test_legacy_results_without_execution_settings():
    data = BenchmarkResult().to_dict()
    del data["execution_settings"]
    assert BenchmarkResult.from_dict(data).execution_settings == {}


def test_config_round_trip_preserves_execution_policy():
    config = BenchmarkConfig(continuous_batching=True, skip_validity_check=True)
    assert BenchmarkConfig.from_dict(config.to_dict(), skip_validity_check=True).hash == config.hash
    assert config.to_dict()["prompt_length_policy"] == "repeat_tokens_to_exact_length"


@pytest.mark.parametrize("disable_eos", [False, True])
def test_suite_preserves_eos_settings(disable_eos):
    import importlib.util
    import sys
    from pathlib import Path

    from transformers import ContinuousBatchingConfig, GenerationConfig

    # Scoring dependencies are unused by this generation/configuration test.
    modules = {
        name: Mock()
        for name in [
            "tabulate",
            "lighteval.models.model_output",
            "lighteval.tasks.lighteval_task",
            "lighteval.tasks.prompt_manager",
            "lighteval.tasks.registry",
            "lighteval.tasks.requests",
        ]
    }
    path = Path(__file__).parents[1] / "benchmark_scripts" / "continuous_batching_overall.py"
    spec = importlib.util.spec_from_file_location("_cb_suite_under_test", path)
    suite = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**modules, spec.name: suite}):
        spec.loader.exec_module(suite)
        results = suite.BenchmarkResults("test-model", "sdpa")
        results.cleanup = Mock()
        model = Mock()
        model.generation_config = GenerationConfig(eos_token_id=[2, 3])
        model.generate_batch.return_value = {
            "req_0": GenerationOutput(
                request_id="req_0",
                prompt_ids=[10, 11],
                status=RequestStatus.FINISHED,
                created_time=1.0,
                lifespan=(1.0, 2.0),
                generated_tokens=[1, 2],
            )
        }
        results._get_model = Mock(return_value=model)
        override = GenerationConfig(eos_token_id=-1) if disable_eos else None
        with patch.object(torch.cuda, "max_memory_allocated", return_value=0):
            results.add_benchmark([[10, 11]], 8, ContinuousBatchingConfig(), override)
        entry = results.entries[0]
        assert entry.error is None
        passed_config = model.generate_batch.call_args.kwargs["generation_config"]
        assert passed_config.eos_token_id == (-1 if disable_eos else [2, 3])
        assert passed_config.max_new_tokens == 8
        assert model.generation_config.max_new_tokens is None
        if override is not None:
            assert override.max_new_tokens is None
        assert entry.gen_config["eos_token_id"] == passed_config.eos_token_id
        json.dumps(entry.gen_config)
