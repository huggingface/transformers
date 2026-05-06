"""End-to-end DeepGEMM EP test on a real DeepSeek checkpoint.

Drives every relevant experts dispatch against `deepseek-ai/DeepSeek-V4-Flash`
using **two model loads** total — same weights are reused across dispatches
via `model.set_experts_implementation`:

  1. Dequantized load (`FineGrainedFP8Config(dequantize=True)`,
     `dtype=torch.bfloat16`) → bf16 weights. Cycles:
       - `grouped_mm`  (torch grouped GEMM)
       - `sonicmoe`    (sonicmoe kernel)
       - `deepgemm`    (`deepgemm_bf16_experts_forward`)
  2. Native quantized load (`dtype="auto"`) → FP8/FP4 weights kept on disk.
     Cycles:
       - `deepgemm`         (`deepgemm_fp8_fp4_experts_forward`)
       - `deepgemm_megamoe` (fused Mega MoE; skipped on < SM100)

Run on B200 with a writable HF cache on the raid mount and torchrun. First run
downloads the checkpoint (hundreds of GB).

    HF_HOME=/raid/ilyas \\
    CUDA_HOME=$HOME/cuda-12.9 \\
    torchrun --nproc_per_node=8 test_deepseek.py

DeepSeek-V3.2 is intentionally not included: this transformers checkout only
registers `deepseek_v3` / `deepseek_v4`, not `deepseek_v32`.
"""

from __future__ import annotations

import gc
import os
import sys

import torch
import torch.distributed as dist

from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config
from transformers.distributed import DistributedConfig


_CHECKPOINT = "deepseek-ai/DeepSeek-V4-Flash"
_PROMPT = "DeepGEMM tests: list three properties of UE8M0 scale factors."

# (label, dispatch, min_sm). All entries in a phase share one model load.
_DEQUANTIZED_DISPATCHES = [
    ("dequantized + grouped_mm", "grouped_mm",  9),
    ("dequantized + sonicmoe",   "sonicmoe",    9),
    ("dequantized + deepgemm",   "deepgemm",    9),
]
_QUANTIZED_DISPATCHES = [
    ("quantized + deepgemm",         "deepgemm",         9),
    ("quantized + deepgemm_megamoe", "deepgemm_megamoe", 10),
]


def _rank0_print(msg: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(msg, flush=True)


def _generate_and_check(model, tok, label: str, rank: int) -> None:
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
        print(f"[{label}] generated {new_tokens.numel()} tokens (finite={finite}):", flush=True)
        print(f"  prompt:     {_PROMPT}", flush=True)
        print(f"  completion: {completion}", flush=True)
        if not finite or new_tokens.numel() == 0:
            raise RuntimeError(f"{label}: generation failed (finite={finite}, n={new_tokens.numel()})")
    dist.barrier()


def _run_phase(load_kwargs: dict, dispatches, cap_major: int, rank: int, results: list) -> None:
    runnable = [(lab, d) for (lab, d, sm) in dispatches if cap_major >= sm]
    skipped = [(lab, d, sm) for (lab, d, sm) in dispatches if cap_major < sm]
    for lab, _, sm in skipped:
        _rank0_print(f"[{lab}] SKIP: needs SM{sm}0+, got SM{cap_major}0")
        results.append((lab, "SKIP", f"needs SM{sm}0+"))
    if not runnable:
        return

    _rank0_print(f"\n--- loading {_CHECKPOINT} (kwargs: {sorted(load_kwargs)}) ---")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            _CHECKPOINT,
            tp_plan="auto",
            distributed_config=DistributedConfig(enable_expert_parallel=True),
            **load_kwargs,
        )
        model.eval()
        tok = AutoTokenizer.from_pretrained(_CHECKPOINT)
    except BaseException as exc:
        if rank == 0:
            print(f"[load] FAIL — {type(exc).__name__}: {exc}", flush=True)
        for label, _ in runnable:
            results.append((label, "FAIL", f"load: {type(exc).__name__}: {exc}"))
        return

    try:
        for label, dispatch in runnable:
            _rank0_print(f"\n=== {label} ===")
            try:
                model.set_experts_implementation(dispatch)
                _generate_and_check(model, tok, label, rank)
                results.append((label, "PASS", ""))
            except BaseException as exc:
                if rank == 0:
                    print(f"[{label}] FAIL — {type(exc).__name__}: {exc}", flush=True)
                results.append((label, "FAIL", f"{type(exc).__name__}: {exc}"))
    finally:
        del model, tok
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

    results: list[tuple[str, str, str]] = []  # (label, status, detail)

    _run_phase(
        load_kwargs={
            "quantization_config": FineGrainedFP8Config(dequantize=True),
            "dtype": torch.bfloat16,
        },
        dispatches=_DEQUANTIZED_DISPATCHES,
        cap_major=cap_major,
        rank=rank,
        results=results,
    )
    _run_phase(
        load_kwargs={"dtype": "auto"},
        dispatches=_QUANTIZED_DISPATCHES,
        cap_major=cap_major,
        rank=rank,
        results=results,
    )

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        passed = [r for r in results if r[1] == "PASS"]
        failed = [r for r in results if r[1] == "FAIL"]
        skipped = [r for r in results if r[1] == "SKIP"]
        width = max((len(r[0]) for r in results), default=20)
        print("\n=== summary ===", flush=True)
        for label, status, detail in results:
            line = f"  {label.ljust(width)}  {status}"
            if detail:
                line += f"  ({detail})"
            print(line, flush=True)
        print(
            f"\n  totals: {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped",
            flush=True,
        )
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
