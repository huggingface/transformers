"""Benchmarks for GGUF checkpoint loading.

Measures end-to-end AutoModelForCausalLM.from_pretrained() time for GGUF files
so the weight-map construction overhead (old: get_gguf_hf_weights_map full
state-dict walk; new: _GGUF_ARCH_CONVERTERS O(1) lookup) is captured.

Run with:
    pip install pytest-benchmark
    RUN_SLOW=1 pytest tests/quantization/ggml/test_ggml_benchmark.py --benchmark-only -v \
        --benchmark-min-rounds=3 --benchmark-max-time=120
"""
import gc

import torch

from transformers import AutoModelForCausalLM
from transformers.testing_utils import require_gguf, slow


# --- models ---
TINYLLAMA_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
TINYLLAMA_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

QWEN_MOE_REPO  = "gdax/Qwen1.5-MoE-A2.7B_gguf"
QWEN_MOE_FILE  = "Qwen1.5-MoE-A2.7B_q4_k_m.gguf"

QWEN3_MOE_REPO = "unsloth/Qwen3-30B-A3B-GGUF"
QWEN3_MOE_FILE = "Qwen3-30B-A3B-Q4_K_M.gguf"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def _load_and_free(repo_id, filename, device):
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        gguf_file=filename,
        device_map=device,
    )
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


# ── TinyLlama (dense, ~669 MB Q4_K_M) ──────────────────────────────────────

@slow
@require_gguf
def test_bench_tinyllama(benchmark):
    """Dense 1.1B model — baseline."""
    benchmark(_load_and_free, TINYLLAMA_REPO, TINYLLAMA_FILE, DEVICE)


# ── Qwen1.5-MoE-A2.7B (~1.7 GB Q4_K_M, 60 layers × 60 experts) ─────────────

@slow
@require_gguf
def test_bench_qwen_moe(benchmark):
    """Qwen1.5-MoE-A2.7B — medium MoE."""
    benchmark(_load_and_free, QWEN_MOE_REPO, QWEN_MOE_FILE, DEVICE)


# ── Qwen3-30B-A3B (MoE, Q4_K_M ~17 GB) ─────────────────────────────────────

@slow
@require_gguf
def test_bench_qwen3_moe_30b(benchmark):
    """Qwen3-30B-A3B — large MoE, main target of the refactor."""
    benchmark(_load_and_free, QWEN3_MOE_REPO, QWEN3_MOE_FILE, DEVICE)
