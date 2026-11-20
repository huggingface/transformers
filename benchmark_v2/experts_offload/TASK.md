# Task: >= 120 tok/s batch-1 decode of Qwen3.8-Flash-Next on this H200, with transformers

You are an autonomous ML performance engineer in a clone of huggingface/transformers, branch `stream-experts`
(PR #50997). GPU: NVIDIA H200 141 GB (FP8 native). Python env: `.venv` (activate it). Always run with `PYTHONPATH=$PWD/src`.

## Model
`ArthurZ/Qwen3.8-Flash-Next-metal-experts-4bit` (model_type `qwen4_exp`, code in `src/transformers/models/qwen4_exp/`):
48 layers (36 gated-delta-net linear attention + 12 full attention with a QSA sparse indexer), hidden 2560,
512 experts top-10 (4-bit affine experts, `MetalExperts`; on CUDA they run the Triton `gather_qmv` in
`src/transformers/integrations/triton_affine_moe.py`), FP8 dense layers, hyper-connections (`GatedResidual`),
a per-layer n-gram embedding (`ple`, ~95 GB table: stream it, `--stream-extra ngram_embedding`).
Edit `modular_qwen4_exp.py` then run `python utils/modular_model_converter.py --files <modular file>`; never edit the
generated `modeling_qwen4_exp.py` by hand.

## Goal (definition of done)
Greedy batch-1 decode >= 120 tok/s (median over 4 prompts x 64 tokens) through `from_pretrained` + `generate`, all
experts resident (`ours:1.0`), output unchanged (greedy tokens match the unoptimized model for >= 32 tokens, or explain
the numeric reason). Stretch: the same with experts streamed at 25% (`ours:0.25`).

## Current numbers (H200)
- `ours:1.0`: 6.70 tok/s decode (149 ms/token), prefill 36 tok/s; `ours:1.0+compile`: 4.83 (worse); `ours:0.25`: 4.42
- even at 1.0 the pool misses 16 experts/token (filled on demand) and each MoE layer syncs the host
- On an M3 Max the same architecture (synthetic weights) runs 29.6 tok/s compiled: the GPU path here is far from its roofline

## How to measure (never a hand-written decode loop)
```bash
python bench_experts_offload.py --model ArthurZ/Qwen3.8-Flash-Next-metal-experts-4bit --configs ours:1.0 \
    --new-tokens 64 --prompt-tokens 512 --stream-extra ngram_embedding --use-kernels
```
Profile first (torch.profiler with CUDA activity) and attack the largest cost.

## Rules
- One change at a time; benchmark it; keep it only if it measurably helps; record every run in `BENCH_LOG.md`
  (date, exact command, config, decode tok/s, prefill tok/s, notes). Never claim a win without a number.
- General fixes go in the library, not the benchmark script. Run `PYTHONPATH=$PWD/src python -m pytest
  tests/distributed/test_offload.py tests/models/qwen4_exp -q -x` after library changes (15 qwen4_exp failures pre-exist:
  compare against `git stash`-free baseline runs, not zero).
- Commit each kept improvement locally (`git commit`); never push (no credentials).
- Ideas (measure first): load all experts resident when the budget fits (no pool, no misses, no per-layer sync);
  CUDA graphs via static cache + compile once host syncs are gone; fused MoE kernels; FP8 dense GEMMs
  (`kernels-community/finegrained-fp8`); FLA kernels for the gated delta rule; vectorized indexer cost at decode.
- Work until the goal is met or time is up; leave BENCH_LOG.md with what helped, what did not, the best command.
