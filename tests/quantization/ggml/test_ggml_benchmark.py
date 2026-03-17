"""Benchmarks for GGUF checkpoint loading.

Run with:
    pip install pytest-benchmark
    pytest tests/quantization/ggml/test_ggml_benchmark.py --benchmark-only -v
"""
import pytest
from huggingface_hub import hf_hub_download

from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint
from transformers.testing_utils import require_gguf, slow


GGUF_MODEL_ID = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
GGUF_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"


@pytest.fixture(scope="module")
def gguf_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gguf")
    return hf_hub_download(GGUF_MODEL_ID, filename=GGUF_FILENAME, local_dir=str(tmp))


@slow
@require_gguf
def test_bench_load_gguf_metadata_only(benchmark, gguf_path):
    """Benchmark metadata-only load (no tensor deserialization)."""
    benchmark(load_gguf_checkpoint, gguf_path, return_tensors=False)


@slow
@require_gguf
def test_bench_load_gguf_with_tensors(benchmark, gguf_path):
    """Benchmark full load including static-table converter construction."""
    benchmark(load_gguf_checkpoint, gguf_path, return_tensors=True)
