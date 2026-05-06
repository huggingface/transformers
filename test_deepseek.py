"""End-to-end DeepGEMM EP test on real DeepSeek checkpoints.

Drives both DeepGEMM dispatches end-to-end through `from_pretrained`:

  1. `deepseek-ai/DeepSeek-V3.2` → `experts_implementation="deepgemm"`
     (block-quantized FP8 weights, float SF, DSv3 recipe).
  2. `deepseek-ai/DeepSeek-V4-Flash` → `experts_implementation="deepgemm_megamoe"`
     (per-row UE8M0 SF, FP4 weights, fused EP + L1 + SwiGLU + L2 path; SM100+ only).

Run on B200 with the local HF cache and torchrun:

    HF_HOME=/raid/arthur \\
    CUDA_HOME=$HOME/cuda-12.9 \\
    torchrun --nproc_per_node=8 test_deepgemm_real.py

Generates a short continuation per checkpoint and asserts the output is finite.
Expensive (loads hundreds of GB per checkpoint).
"""

from __future__ import annotations

import gc
import os
import sys

import torch
import torch.distributed as dist

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed import DistributedConfig


_CHECKPOINTS = [
    ("deepseek-ai/DeepSeek-V3.2", "deepgemm", 9),
    ("deepseek-ai/DeepSeek-V4-Flash", "deepgemm_megamoe", 10),
]
_PROMPT = "DeepGEMM tests: list three properties of UE8M0 scale factors."


def _rank0_print(msg: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(msg, flush=True)


def _run_one(model_id: str, dispatch: str, rank: int) -> None:
    _rank0_print(f"\n=== {model_id}  (dispatch={dispatch}) ===")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        tp_plan="auto",
        distributed_config=DistributedConfig(enable_expert_parallel=True),
        experts_implementation=dispatch,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_id)
    inputs = tok(_PROMPT, return_tensors="pt").to(model.device)

    dist.barrier()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    if rank == 0:
        finite = torch.isfinite(out_ids.float()).all().item()
        new_tokens = out_ids[0, inputs.input_ids.size(1):]
        completion = tok.decode(new_tokens, skip_special_tokens=True)
        print(f"[{model_id}] generated {new_tokens.numel()} tokens (finite={finite}):", flush=True)
        print(f"  prompt:     {_PROMPT}", flush=True)
        print(f"  completion: {completion}", flush=True)
        if not finite or new_tokens.numel() == 0:
            raise RuntimeError(f"{model_id}: generation failed (finite={finite}, n={new_tokens.numel()})")

    dist.barrier()
    del model, tok, inputs, out_ids
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        sys.exit("EP test needs >=2 ranks (run with `torchrun --nproc_per_node=N`).")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))

    cap_major = torch.cuda.get_device_capability()[0]
    _rank0_print(f"device cap: SM{cap_major}0, world_size={world_size}")

    failures: list[tuple[str, BaseException]] = []
    for model_id, dispatch, min_sm in _CHECKPOINTS:
        if cap_major < min_sm:
            _rank0_print(f"[{model_id}] SKIP: needs SM{min_sm}0+, got SM{cap_major}0")
            continue
        try:
            _run_one(model_id, dispatch, rank)
        except BaseException as exc:
            if rank == 0:
                failures.append((model_id, exc))
                print(f"[{model_id}] FAIL — {type(exc).__name__}: {exc}", flush=True)

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        if failures:
            print(f"\n=== {len(failures)} model(s) failed ===", flush=True)
            for name, exc in failures:
                print(f"  - {name}: {type(exc).__name__}: {exc}", flush=True)
            return 1
        print("\n=== all models passed ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
