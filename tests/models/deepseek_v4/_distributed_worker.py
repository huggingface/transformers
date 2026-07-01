"""EP=8 distributed worker for the DSv4-Flash integration tests.

Invoked under ``torchrun`` with a ``--test=<name>`` argument by
``_run_distributed_worker`` in ``test_modeling_deepseek_v4.py``. Each test is a
plain function in this module that builds a :class:`Worker` and calls
``worker.run(...)`` once or twice with different generate kwargs.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist

from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig
from transformers.distributed import DistributedConfig


V4_INSTRUCT = "deepseek-ai/DeepSeek-V4-Flash"
V4_BASE = "deepseek-ai/DeepSeek-V4-Flash-Base"
EXPECTED_PRIMES = "2, 3, 5, 7, 11, 13, 17, 19, 23, 29"
V4_PROMPT_BASE = "Here is the list of the first ten prime numbers, separated by commas:"
V4_PROMPT_INSTRUCT = "<｜begin▁of▁sentence｜><｜User｜>List the first ten prime numbers:<｜Assistant｜></think>"


class Worker:
    """One distributed worker process: holds the loaded model + per-rank state
    and exposes ``run`` to sweep a generate call across a tuple of dispatches.
    """

    def __init__(self, model_id, prompt, add_special_tokens, loadtime_dispatch):
        self.rank = int(os.environ["RANK"])
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype="auto",
            attn_implementation="eager",
            experts_implementation=loadtime_dispatch,
            distributed_config=DistributedConfig(enable_expert_parallel=True),
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens).to(
            self.model.device
        )
        self.is_fp8 = getattr(self.model.config, "expert_dtype", "fp8") != "fp4"
        self.current_dispatch = loadtime_dispatch
        self.failed: list[str] = []

    def _switch(self, dispatch):
        # ``set_experts_implementation`` raises when crossing in/out of
        # ``deepgemm_megamoe`` (locked at load time) — only switch on real
        # transitions. FP8 only: Triton multi-expert kernels generate wrong
        # tokens after DeepGEMM has run on the same module — force Triton-only
        # for batched_mm / grouped_mm. Without
        # ``TRANSFORMERS_DISABLE_EXPERTS_DECODE_OPTIMIZATION=1``,
        # ``_optimize_model_for_decode`` silently swaps ``grouped_mm`` →
        # ``batched_mm`` during decode and the ``grouped_mm`` entry would never
        # run end-to-end.
        if dispatch != self.current_dispatch:
            self.model.set_experts_implementation(dispatch)
            self.current_dispatch = dispatch
        if self.is_fp8 and dispatch in ("batched_mm", "grouped_mm"):
            os.environ["TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR"] = "1"
        else:
            os.environ.pop("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", None)
        os.environ["TRANSFORMERS_DISABLE_EXPERTS_DECODE_OPTIMIZATION"] = "1"

    def run(self, runtime_dispatches, gen_kwargs, *, label, expected=None):
        """Sweep ``model.generate(**gen_kwargs)`` across all dispatches.
        If ``expected`` is set, the substring must appear in the decoded output.
        """
        for dispatch in runtime_dispatches:
            self._switch(dispatch)
            dist.barrier()
            try:
                with torch.no_grad():
                    out = self.model.generate(
                        **self.inputs,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        **gen_kwargs,
                    )
                ok = True
                if expected is not None:
                    decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
                    if expected not in decoded:
                        self.failed.append(f"[{label} {dispatch}] {decoded!r} does not contain {expected!r}")
                        ok = False
            except Exception as e:
                self.failed.append(f"[{label} {dispatch}] {type(e).__name__}: {e}")
                ok = False
            dist.barrier()
            if self.rank == 0:
                print(f"[{label} {dispatch}] {'OK' if ok else 'FAIL'}", flush=True)

    def truncate_decoder(self, n_layers):
        """Keep only the first ``n_layers`` decoder blocks and clear any
        stale cache. Used by the compile-generate tests so each dispatch only
        pays the (small) per-shape compile cost, not 43× the per-layer cost.
        Output of the post-truncate sweep can't be content-checked — 2 layers
        can't produce the reference completion."""
        self.model.model.layers = torch.nn.ModuleList(list(self.model.model.layers)[:n_layers])
        if hasattr(self.model.config, "num_hidden_layers"):
            self.model.config.num_hidden_layers = n_layers
        if hasattr(self.model, "_cache"):
            del self.model._cache

    def finalize(self) -> int:
        dist.barrier()
        dist.destroy_process_group()
        if self.rank == 0 and self.failed:
            print("FAILED:\n" + "\n".join(self.failed), flush=True)
            return 1
        return 0


# ── Tests ─────────────────────────────────────────────────────────────────────


def fp4_generation():
    w = Worker(V4_INSTRUCT, V4_PROMPT_INSTRUCT, add_special_tokens=False, loadtime_dispatch=None)
    w.run(
        ("eager", "batched_mm", "grouped_mm", "deepgemm"),
        {"max_new_tokens": 64},
        label="eager",
        expected=EXPECTED_PRIMES,
    )
    return w.finalize()


def fp4_generation_megamoe():
    w = Worker(V4_INSTRUCT, V4_PROMPT_INSTRUCT, add_special_tokens=False, loadtime_dispatch="deepgemm_megamoe")
    w.run(
        ("deepgemm_megamoe",),
        {"max_new_tokens": 64},
        label="eager",
        expected=EXPECTED_PRIMES,
    )
    return w.finalize()


def fp4_generation_compile_static():
    w = Worker(V4_INSTRUCT, V4_PROMPT_INSTRUCT, add_special_tokens=False, loadtime_dispatch=None)
    dispatches = ("batched_mm", "grouped_mm", "deepgemm")
    # 1. Full model + static cache, eager — correctness baseline.
    w.run(
        dispatches,
        {"max_new_tokens": 64, "cache_implementation": "static"},
        label="eager",
        expected=EXPECTED_PRIMES,
    )
    # 2. Trim to 2 layers + compile — smoke-test the compile path.
    w.truncate_decoder(2)
    w.run(
        dispatches,
        {"max_new_tokens": 8, "cache_implementation": "static", "compile_config": CompileConfig(fullgraph=True)},
        label="compile-generate",
    )
    return w.finalize()


def fp4_generation_compile_static_megamoe():
    w = Worker(V4_INSTRUCT, V4_PROMPT_INSTRUCT, add_special_tokens=False, loadtime_dispatch="deepgemm_megamoe")
    w.run(
        ("deepgemm_megamoe",),
        {"max_new_tokens": 64, "cache_implementation": "static"},
        label="eager",
        expected=EXPECTED_PRIMES,
    )
    w.truncate_decoder(2)
    w.run(
        ("deepgemm_megamoe",),
        {"max_new_tokens": 8, "cache_implementation": "static", "compile_config": CompileConfig(fullgraph=True)},
        label="compile-generate",
    )
    return w.finalize()


def fp8_base_generation():
    # DeepGEMM is CUDA-only — drop it on non-CUDA backends (e.g. XPU).
    dispatches = (
        ("eager", "batched_mm", "grouped_mm", "deepgemm")
        if torch.cuda.is_available()
        else ("eager", "batched_mm", "grouped_mm")
    )
    w = Worker(V4_BASE, V4_PROMPT_BASE, add_special_tokens=True, loadtime_dispatch=None)
    w.run(dispatches, {"max_new_tokens": 64}, label="eager", expected=EXPECTED_PRIMES)
    return w.finalize()


def fp8_base_generation_compile_static():
    dispatches = (
        ("batched_mm", "grouped_mm", "deepgemm") if torch.cuda.is_available() else ("batched_mm", "grouped_mm")
    )
    w = Worker(V4_BASE, V4_PROMPT_BASE, add_special_tokens=True, loadtime_dispatch=None)
    w.run(
        dispatches,
        {"max_new_tokens": 64, "cache_implementation": "static"},
        label="eager",
        expected=EXPECTED_PRIMES,
    )
    w.truncate_decoder(2)
    w.run(
        dispatches,
        {"max_new_tokens": 8, "cache_implementation": "static", "compile_config": CompileConfig(fullgraph=True)},
        label="compile-generate",
    )
    return w.finalize()


TESTS = {
    fn.__name__: fn
    for fn in (
        fp4_generation,
        fp4_generation_megamoe,
        fp4_generation_compile_static,
        fp4_generation_compile_static_megamoe,
        fp8_base_generation,
        fp8_base_generation_compile_static,
    )
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, choices=list(TESTS))
    args = parser.parse_args()
    return TESTS[args.test]()


if __name__ == "__main__":
    sys.exit(main())
