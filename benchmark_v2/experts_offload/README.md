# Streamed MoE experts: benchmarks

- `bench_experts_offload.py`: decode / prefill tok/s and per-token streaming stats for `full`, `ours:<cache fraction>`
  (`+compile`), on any checkpoint (local dir or Hub repo). `--router-skew` simulates routing locality on synthetic weights.
- `make_qwen38_synthetic_ckpt.py`: a loadable Qwen3.8-Flash-Next with all 512 experts and random weights (~80 GB),
  to measure streaming on a Mac without the real checkpoint. `sim_routing.py`: the router skew.
- `bench_qwen38_synthetic.py`: the same architecture fully resident (64 experts, same work per token), the reference.
- `verify_generate.py`: public-API run on a real checkpoint (correctness vs eager, decode and sustained tok/s).
- `convert_metal_experts.py`: an FP8 / bf16 checkpoint -> 4-bit Metal experts in the transformers format.
- `dense_quant.py`, `quality_check.py`: 4-bit always-on layers on Metal and their perplexity.

Results and the experiment log: `BENCH_LOG.md`. Check the GPU is idle first
(`ioreg -r -c IOAccelerator | grep "Device Utilization"`): another app on the GPU skews every number.
