"""End-to-end DeepGEMM EP test on a real DeepSeek checkpoint.

Drives both DeepGEMM dispatches against `deepseek-ai/DeepSeek-V4-Flash`
(per-row UE8M0 SF, FP4 weights):

  1. `experts_implementation="deepgemm"`         — M-grouped FP8/FP4 path.
  2. `experts_implementation="deepgemm_megamoe"` — fused EP + L1 + SwiGLU + L2
                                                   (SM100+ only).

Run on B200 with a writable HF cache on the raid mount and torchrun. First run
downloads the checkpoint (hundreds of GB).

    HF_HOME=/raid/ilyas \\
    CUDA_HOME=$HOME/cuda-12.9 \\
    torchrun --nproc_per_node=8 test_deepseek.py

DeepSeek-V3.2 is intentionally not included: this transformers checkout only
registers `deepseek_v3` / `deepseek_v4`, not `deepseek_v32`. Add it back here
once the architecture lands.

Generates a short continuation per dispatch and asserts the output is finite.
"""

from __future__ import annotations

import gc
import os
import sys

import torch
import torch.distributed as dist

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed import DistributedConfig


_CHECKPOINT = "deepseek-ai/DeepSeek-V4-Flash"
_RUNS = [
    ("deepgemm", 9),           # M-grouped FP8/FP4 path
    ("deepgemm_megamoe", 10),  # fused Mega MoE (SM100+)
]
_PROMPT = "DeepGEMM tests: list three properties of UE8M0 scale factors."


def _rank0_print(msg: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(msg, flush=True)


def _run_one(dispatch: str, rank: int) -> None:
    _rank0_print(f"\n=== {_CHECKPOINT}  (dispatch={dispatch}) ===")

    model = AutoModelForCausalLM.from_pretrained(
        _CHECKPOINT,
        tp_plan="auto",
        distributed_config=DistributedConfig(enable_expert_parallel=True),
        experts_implementation=dispatch,
        dtype=torch.bfloat16,
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(_CHECKPOINT)
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
        print(f"[{dispatch}] generated {new_tokens.numel()} tokens (finite={finite}):", flush=True)
        print(f"  prompt:     {_PROMPT}", flush=True)
        print(f"  completion: {completion}", flush=True)
        if not finite or new_tokens.numel() == 0:
            raise RuntimeError(f"{dispatch}: generation failed (finite={finite}, n={new_tokens.numel()})")

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
    for dispatch, min_sm in _RUNS:
        if cap_major < min_sm:
            _rank0_print(f"[{dispatch}] SKIP: needs SM{min_sm}0+, got SM{cap_major}0")
            continue
        try:
            _run_one(dispatch, rank)
        except BaseException as exc:
            if rank == 0:
                failures.append((dispatch, exc))
                print(f"[{dispatch}] FAIL — {type(exc).__name__}: {exc}", flush=True)

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        if failures:
            print(f"\n=== {len(failures)} dispatch(es) failed ===", flush=True)
            for name, exc in failures:
                print(f"  - {name}: {type(exc).__name__}: {exc}", flush=True)
            return 1
        print("\n=== all dispatches passed ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
