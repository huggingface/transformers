# Streamed MoE experts: benchmarks and diagnostics (PR #50997)

Run from this directory with `PYTHONPATH=../../src`. Results and exact commands: `BENCH_LOG.md`.

| Script | What |
|---|---|
| `bench_experts_offload.py` | decode / sustained / prefill tok/s via `generate` (full, streamed, accelerate, POC; `+compile`) |
| `verify_generate.py` | public-API check on MPS: static cache + compile, correctness vs eager, sustained |
| `quality_check.py`, `dense_quant.py` | perplexity / greedy agreement of always-on layer quantization specs |
| `profile_offload.py` | where a streamed decode token goes (host phases, IO, ops) |
| `diag_*.py` | targeted diagnostics (IO paths, compile graph breaks, host vs device time, kernels) |
| `convert_metal_experts.py` | FP8/bf16 checkpoint -> stacked Metal affine experts (transformers format) |
| `bench_vllm.py` | vLLM batch-1 reference |
| `run_agent.sh`, `TASK.md` | autonomous pi agent on HF Jobs working on GPU decode speed |
