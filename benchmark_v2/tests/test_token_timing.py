from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from benchmark_v2.framework.benchmark_runner import BenchmarkRunner, BenchmarkStreamer
from benchmark_v2.framework.data_classes import BenchmarkResult
from transformers.generation.continuous_batching.requests import GenerationOutput, RequestStatus


def test_stream_completion_is_not_a_token():
    streamer = BenchmarkStreamer()
    with patch("benchmark_v2.framework.benchmark_runner.time.perf_counter", side_effect=[1.0, 2.0]):
        streamer.put(10)
        streamer.put(20)
        streamer.end()
    assert streamer.timestamps == [1.0, 2.0]
    assert list(streamer) == [10, 20]


@pytest.mark.parametrize("continuous_batching", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 3])
def test_generation_token_timing(continuous_batching, num_tokens):
    # Large absolute clock values must preserve millisecond-scale token gaps.
    start = 1_000_000.0
    token_times = [start + offset for offset in [0.01, 0.02, 0.04][:num_tokens]]
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.inputs = {"input_ids": torch.tensor([[10, 11]])}
    runner.logger = Mock()
    runner.tokenizer = Mock()
    runner.torch_accelerator_module = Mock()
    runner.torch_accelerator_module.memory_allocated.return_value = 0
    runner.model = Mock()
    runner.continuous_batching_config = None

    def generate(input_ids, streamer):
        streamer.put(input_ids)
        for token in range(num_tokens):
            streamer.put(torch.tensor([token]))
        streamer.end()
        return torch.tensor([[10, 11] + list(range(num_tokens))])

    runner.model.generate.side_effect = generate
    runner.model.generate_batch.return_value = {
        "req_0": GenerationOutput(
            request_id="req_0",
            prompt_ids=[10, 11],
            status=RequestStatus.FINISHED,
            timestamps=token_times,
            generated_tokens=list(range(num_tokens)),
        )
    }
    config = SimpleNamespace(
        continuous_batching=continuous_batching, gpu_monitoring=False, num_tokens_to_generate=num_tokens, batch_size=1
    )
    # Completion occurs well after the final token and must only affect end-to-end latency.
    clock = [start, start + 0.5] if continuous_batching else [start, start + 0.001, *token_times, start + 0.5]
    with (
        patch("benchmark_v2.framework.benchmark_runner.time.perf_counter", side_effect=clock),
        patch("benchmark_v2.framework.benchmark_runner.flush_memory"),
    ):
        latency, timestamps, output, gpu_metrics = runner.time_generate(config, warmup=False)

    np.testing.assert_allclose(timestamps, [[0.01, 0.02, 0.04][:num_tokens]], atol=1e-9)
    assert latency == 0.5
    result = BenchmarkResult()
    result.accumulate(latency, timestamps, output, gpu_metrics)
    assert result.time_to_first_token[0] == pytest.approx(0.01)
    assert result.inter_token_latency[0] == pytest.approx(0.015 if num_tokens == 3 else 0.0)
