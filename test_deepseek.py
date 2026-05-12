"""End-to-end DeepGEMM EP test on a real DeepSeek checkpoint.

Drives the FP8/FP4 DeepGEMM dispatches against `deepseek-ai/DeepSeek-V4-Flash`
with one model load shared across dispatches via `model.set_experts_implementation`:

  - `deepgemm`         → `deepgemm_fp8_fp4_experts_forward`
  - `deepgemm_megamoe` → fused Mega MoE (skipped on < SM100)

Dequantize-on-load is intentionally NOT exercised here: V4-Flash is ~671B
parameters; dequantizing to bf16 needs ~1.3 TB across the world, which doesn't
fit in 8× B200 (178 GB each). For the bf16 / dequantized expert dispatches
(`grouped_mm`, `sonicmoe`, `deepgemm` bf16) use a smaller MoE checkpoint.

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
import traceback

import torch
import torch.distributed as dist

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed import DistributedConfig
from transformers.utils.quantization_config import FineGrainedFP8Config


_CHECKPOINT = "deepseek-ai/DeepSeek-V4-Flash"
_PROMPTS = [
    "The capital of France is",
    "List the first ten prime numbers:",
    "Translate to French: 'The quick brown fox jumps over the lazy dog.'",
    "Write a Python function fibonacci(n) that returns the nth Fibonacci number.",
    "What are the three properties of the UE8M0 scale factor format?",
    'Write a short story that begins with: "Once upon a time, in a forest far away, there lived a..."',
]


def _format_chat(prompt: str) -> str:
    """Wrap a single-turn user prompt in V4-Flash's chat-mode template.

    The tokenizer doesn't ship a Jinja `chat_template`; the canonical format is in
    `encoding/encoding_dsv4.py` on the model repo. Chat mode places `</think>`
    right after `<｜Assistant｜>` to skip the reasoning block and answer directly.
    """
    return f"<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜></think>"

# (label, dispatch, min_sm). All entries share one model load.
_QUANTIZED_DISPATCHES = [
    ("quantized + deepgemm",         "deepgemm",         9),
    ("quantized + deepgemm_megamoe", "deepgemm_megamoe", 10),
]


def _rank0_print(msg: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(msg, flush=True)


def _render_report(results: list[tuple[str, str, str]], completions: dict[str, list[str]]) -> None:
    """Render a side-by-side rich table comparing each dispatch's per-prompt completion."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("")

    # Per-dispatch status row first.
    status_table = Table(title="Run summary", show_lines=False)
    status_table.add_column("dispatch", style="bold")
    status_table.add_column("status")
    status_table.add_column("detail", overflow="fold")
    for label, status, detail in results:
        style = "green" if status == "PASS" else "yellow" if status == "SKIP" else "red"
        status_table.add_row(label, f"[{style}]{status}[/{style}]", detail)
    console.print(status_table)

    if not completions:
        return

    # Per-prompt completions side by side. Each dispatch gets its own column.
    dispatch_keys = list(completions.keys())
    title = "completions: " + " vs ".join(dispatch_keys)
    completion_table = Table(title=title, show_lines=True)
    completion_table.add_column("#", justify="right", no_wrap=True)
    completion_table.add_column("prompt", overflow="fold", max_width=30)
    for d in dispatch_keys:
        completion_table.add_column(d, overflow="fold", max_width=42)

    for i, prompt in enumerate(_PROMPTS):
        row = [str(i + 1), prompt]
        for d in dispatch_keys:
            comps = completions.get(d, [])
            row.append(comps[i] if i < len(comps) else "")
        completion_table.add_row(*row)
    console.print(completion_table)


def _generate_and_check(model, tok, label: str, rank: int, completions: list[str]) -> None:
    for i, prompt in enumerate(_PROMPTS):
        inputs = tok(_format_chat(prompt), return_tensors="pt", add_special_tokens=False).to(model.device)
        dist.barrier()
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        if rank == 0:
            finite = torch.isfinite(out_ids.float()).all().item()
            new_tokens = out_ids[0, inputs.input_ids.size(1):]
            completion = tok.decode(new_tokens, skip_special_tokens=True)
            completion_raw = tok.decode(new_tokens, skip_special_tokens=False)
            completions.append(completion)
            print(f"[{label}] prompt {i + 1}/{len(_PROMPTS)} — {new_tokens.numel()} tokens (finite={finite}):", flush=True)
            print(f"  completion:  {completion!r}", flush=True)
            print(f"  raw decode:  {completion_raw!r}", flush=True)
            print(f"  token ids:   {new_tokens.tolist()}", flush=True)
            if not finite or new_tokens.numel() == 0:
                raise RuntimeError(f"{label}: generation failed (finite={finite}, n={new_tokens.numel()})")
        dist.barrier()


def _run_dequant_phase(results: list, completions: dict[str, list[str]]) -> None:
    """Dequantized (bf16) baseline via FineGrainedFP8Config(dequantize=True) +
    `device_map="auto"` CPU offload. Runs only on rank 0 — the dequantized model
    is ~1.3 TB and we use the host's RAM (`max_memory`) to keep all 8 ranks'
    activations from contending with the model weights on GPU 0.
    """
    label = "dequantized (bf16)"
    dispatch = "dequantized"
    print(f"\n--- loading {_CHECKPOINT} (dequantize=True, device_map=auto across all GPUs) ---", flush=True)
    # Spread the bf16 model across all 8 GPUs (~1.36 TB total) instead of GPU 0 + CPU offload —
    # the latter forces every forward pass to stream weights over PCIe and is ~10× slower.
    # Other torchrun ranks have torn down by now, so GPUs 1..N are free.
    n_gpus = torch.cuda.device_count()
    max_memory = {i: "170GiB" for i in range(n_gpus)}
    max_memory["cpu"] = "1500GiB"  # fallback for any leftover
    qcfg = FineGrainedFP8Config(dequantize=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            _CHECKPOINT,
            device_map="auto",
            dtype="auto",
            quantization_config=qcfg,
            max_memory=max_memory,
        )
        model.eval()
        tok = AutoTokenizer.from_pretrained(_CHECKPOINT)
    except BaseException as exc:
        print(f"[load] FAIL — {type(exc).__name__}: {exc}", flush=True)
        results.append((label, "FAIL", f"load: {type(exc).__name__}: {exc}"))
        return

    completions[dispatch] = []
    print(f"\n=== {label} ===", flush=True)
    try:
        for i, prompt in enumerate(_PROMPTS):
            inputs = tok(_format_chat(prompt), return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            new_tokens = out_ids[0, inputs.input_ids.size(1):]
            completion = tok.decode(new_tokens, skip_special_tokens=True)
            completion_raw = tok.decode(new_tokens, skip_special_tokens=False)
            completions[dispatch].append(completion)
            print(f"[{label}] prompt {i + 1}/{len(_PROMPTS)} — {new_tokens.numel()} tokens:", flush=True)
            print(f"  completion:  {completion!r}", flush=True)
            print(f"  raw decode:  {completion_raw!r}", flush=True)
            print(f"  token ids:   {new_tokens.tolist()}", flush=True)
        results.append((label, "PASS", ""))
    except BaseException as exc:
        print(f"[{label}] FAIL — {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        results.append((label, "FAIL", f"{type(exc).__name__}: {exc}"))
    finally:
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()


def _run_phase(
    load_kwargs: dict,
    dispatches,
    cap_major: int,
    rank: int,
    results: list,
    completions: dict[str, list[str]],
) -> None:
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
                completions[dispatch] = []
                _generate_and_check(model, tok, label, rank, completions[dispatch])
                results.append((label, "PASS", ""))
            except BaseException as exc:
                if rank == 0:
                    print(f"[{label}] FAIL — {type(exc).__name__}: {exc}", flush=True)
                    traceback.print_exc()
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
    completions: dict[str, list[str]] = {}    # dispatch → per-prompt completions

    _run_phase(
        load_kwargs={"dtype": "auto"},
        dispatches=_QUANTIZED_DISPATCHES,
        cap_major=cap_major,
        rank=rank,
        results=results,
        completions=completions,
    )

    dist.barrier()
    dist.destroy_process_group()

    # Dequantized bf16 baseline: rank 0 only, spread across all GPUs once
    # ranks 1..N have exited and released their per-rank quantized weights.
    if rank == 0:
        import time
        time.sleep(5)
        torch.cuda.empty_cache()
        _run_dequant_phase(results, completions)

    if rank == 0:
        passed = [r for r in results if r[1] == "PASS"]
        failed = [r for r in results if r[1] == "FAIL"]
        skipped = [r for r in results if r[1] == "SKIP"]
        _render_report(results, completions)
        print(
            f"\n  totals: {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped",
            flush=True,
        )
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
