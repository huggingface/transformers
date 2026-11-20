# Task: >= 120 tok/s sustained batch-1 decode on this A100, Qwen3-30B-A3B, with transformers

You are an autonomous ML performance engineer working in a clone of huggingface/transformers, branch `stream-experts`
(PR #50997: MoE experts streamed from the checkpoint through a device slot cache; `src/transformers/distributed/offload.py`).
GPU: NVIDIA A100 80GB. Python env: `.venv` (activate it). Always run with `PYTHONPATH=$PWD/src`.

## Goal (definition of done)
Batch-1 greedy decode of Qwen3-30B-A3B at **>= 120 tok/s sustained** (mean over tokens 512-1024 of a 1024-token generation),
through transformers' public API (`from_pretrained` + `generate`), with:
1. the model fully on the GPU (bf16 `ArthurZ/Qwen3-30B-A3B-native` or 4-bit experts `ArthurZ/Qwen3-30B-A3B-metal-experts-4bit`), and
2. as a stretch, experts streamed (`device_map={"": "cuda", "experts": "disk"}`, `max_memory={0: <bytes>}`) at 50% / 25% of the experts.
Output quality must hold: greedy tokens must match the unoptimized model's for >= 32 tokens (or explain the numeric reason).

## Current numbers (A100, batch 1, decode tok/s)
- vLLM all on GPU (CUDA graphs): ~137
- transformers full bf16 + `generate(cache_implementation="static")` + torch.compile (reduce-overhead): 67.1
- transformers full bf16, eager: 13.45
- streamed bf16 experts, 50% cache + compile: 12.8 (IO bound before recent fixes); 4-bit experts streamed 50%: 7.7 (Triton `gather_qmv` kernel slow)
- 4-bit experts full + compile: 11.25 (kernel `src/transformers/integrations/triton_affine_moe.py` is the bottleneck)

## How to measure (never a hand-written decode loop)
```bash
python bench_experts_offload.py --model ArthurZ/Qwen3-30B-A3B-native --configs full+compile --new-tokens 128 --sustained 1024
python profile_offload.py ArthurZ/Qwen3-30B-A3B-native 0.25 sdpa      # streamed-experts breakdown
```
Use torch.profiler to find what dominates a decode step before changing anything.

## Rules
- One change at a time; benchmark it; keep it only if it measurably helps; record every run in `BENCH_LOG.md`
  (date, exact command, config, decode tok/s, sustained tok/s, prefill tok/s, notes). Never claim a win without a number.
- Keep changes minimal and in the library (not in the benchmark script) when they are general improvements.
- Run `PYTHONPATH=$PWD/src python -m pytest tests/distributed/test_offload.py -q` after library changes.
- Commit each kept improvement locally with a clear message (`git commit`); never push (there are no credentials).
- Ideas to evaluate (measure first): CUDA graphs through static cache + compile on every path; `experts_implementation`
  (`batched_mm` vs `grouped_mm` vs fused MoE kernels from the kernels hub, e.g. `kernels-community/*moe*`);
  `attn_implementation="kernels-community/flash-attn2"`; `model.set_use_kernels(True)`; removing host syncs and Python
  from the decode step; a faster 4-bit decode kernel (vectorized unpack, all routes of a token in one launch);
  fewer kernel launches per layer (fusions).
- Work until the goal is met or the time is up. When you stop, leave BENCH_LOG.md with a summary of what helped,
  what did not, and the best reproducible command.
